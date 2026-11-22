"""SD3 prompt encodings: the three-tower encoder, the per-shard text cache and the empty-prompt cache.

A prepared split stores one ``shard_NNNNN_text.safetensors`` per latent shard (``text_emb [n, 154, 4096]``, ``pooled [n, 2048]``,
``attn_mask [n, 154]``, row ``i`` aligned with row ``i`` of the latent shard) and one ``empty_prompt.safetensors`` (the encoding of ``""``, the
caption-dropout target and the negative prompt) with a fingerprint manifest (model snapshot, ``max_sequence_length``, dtype, sha256 of the
bytes), so a later run can tell the cache was produced under the conditions it is about to be used under.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, safe_open, save

from distill_to_loop.hub import explain_missing_model, missing_model_message
from distill_to_loop.io_utils import atomic_write, atomic_write_text

TEXT_EMB, POOLED, ATTN_MASK = "text_emb", "pooled", "attn_mask"
EMPTY_PROMPT_FILE = "empty_prompt.safetensors"
EMPTY_PROMPT_MANIFEST_FILE = "empty_prompt_manifest.json"

TextEncodeFn = Callable[[Sequence[str]], dict[str, torch.Tensor]]


# --------------------------------------------------------------------------- #
# the encoder
# --------------------------------------------------------------------------- #
class Sd3PromptEncoder:
    """CLIP-L + CLIP-G + T5 prompt encoding through ``StableDiffusion3Pipeline.encode_prompt``, so the cached embeddings are what the pipeline
    produces at sampling time. Loads only the text towers (``transformer=None, vae=None``). ``__call__`` returns, for
    ``max_sequence_length=77``, ``text_emb [B, 154, 4096]``, ``pooled [B, 2048]`` and an all-ones ``attn_mask`` (SD3 pads to a fixed length
    and does not mask)."""

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 77,
        local_files_only: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        from diffusers import StableDiffusion3Pipeline

        self.model_id = model_id
        self.max_sequence_length = int(max_sequence_length)
        self.device = torch.device(device)
        with explain_missing_model(model_id, cache_dir):
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id, transformer=None, vae=None, dtype=dtype, local_files_only=local_files_only, cache_dir=cache_dir
            ).to(self.device)
        # from_pretrained does not put the text encoders in eval mode; without this, dropout makes the encodings non-deterministic.
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self.pipe, name, None)
            if module is not None:
                module.eval()

    @torch.no_grad()
    def __call__(self, captions: Sequence[str]) -> dict[str, torch.Tensor]:
        prompts = list(captions)
        prompt_embeds, _, pooled, _ = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            do_classifier_free_guidance=False,
            max_sequence_length=self.max_sequence_length,
        )
        n, seq_len = prompt_embeds.shape[0], prompt_embeds.shape[1]
        return {TEXT_EMB: prompt_embeds, POOLED: pooled, ATTN_MASK: torch.ones(n, seq_len, dtype=torch.bool)}


def encode_captions(captions: Sequence[str], encode: TextEncodeFn, batch_size: int = 256, dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    """Run ``encode`` over batches of ``captions`` and concatenate; embeddings are cast to ``dtype`` on the CPU (the mask stays bool)."""
    if not captions:
        raise ValueError("no captions to encode")
    parts: dict[str, list[torch.Tensor]] = {TEXT_EMB: [], POOLED: [], ATTN_MASK: []}
    for i in range(0, len(captions), batch_size):
        out = encode(list(captions[i : i + batch_size]))
        parts[TEXT_EMB].append(out[TEXT_EMB].to("cpu", dtype))
        parts[POOLED].append(out[POOLED].to("cpu", dtype))
        parts[ATTN_MASK].append(out[ATTN_MASK].to("cpu", torch.bool))
    return {k: torch.cat(v).contiguous() for k, v in parts.items()}


def unique_captions(captions: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """``(unique, idx)`` with ``unique[idx[i]] == captions[i]``, in the order of first occurrence."""
    lookup: dict[str, int] = {}
    idx = np.empty(len(captions), dtype=np.int32)
    for i, c in enumerate(captions):
        j = lookup.get(c)
        if j is None:
            j = lookup[c] = len(lookup)
        idx[i] = j
    return list(lookup), idx


def resolve_snapshot(model_id: str, filename: str = "model_index.json", cache_dir: str | None = None) -> str:
    """The resolved Hugging Face Hub snapshot (commit hash) of ``model_id`` in the local cache (``cache_dir``, default ``HF_HUB_CACHE``). Raises
    ``FileNotFoundError`` when the model is not cached locally."""
    from huggingface_hub import try_to_load_from_cache

    cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
    resolved = try_to_load_from_cache(model_id, filename, cache_dir=cache_dir)
    if not isinstance(resolved, str):
        raise FileNotFoundError(f"{filename} of {model_id} not found. {missing_model_message(model_id, cache_dir)}")
    return Path(resolved).parent.name  # .../snapshots/<commit hash>/<filename>


# --------------------------------------------------------------------------- #
# per-shard text cache
# --------------------------------------------------------------------------- #
def shard_text_name(k: int) -> str:
    return f"shard_{int(k):05d}_text.safetensors"


def write_shard_text(path: str | Path, text: Mapping[str, torch.Tensor]) -> Path:
    """Write one shard's ``{text_emb, pooled, attn_mask}`` rows (aligned with its latent shard)."""
    missing = {TEXT_EMB, POOLED, ATTN_MASK} - set(text)
    if missing:
        raise ValueError(f"shard text is missing {sorted(missing)}")
    payload = save({k: text[k].contiguous() for k in (TEXT_EMB, POOLED, ATTN_MASK)})
    return atomic_write(Path(path), lambda fh: fh.write(payload), binary=True)


class ShardedLazyTextCache:
    """Memory-mapped, per-row access into the ``shard_NNNNN_text.safetensors`` files of a split. ``shard_ns`` is every shard's row count, used to
    map a global row to ``(shard, row in shard)``. A shard file is opened on first access to one of its rows, and at most ``max_open`` files stay
    open (least recently used eviction). The state is per instance, so every ``DataLoader`` worker opens its own files after the fork."""

    def __init__(self, root: str | Path, shard_ns: Sequence[int], max_open: int = 8) -> None:
        self.root = Path(root)
        self.max_open = max(1, int(max_open))
        self._offsets = [0]
        for n in shard_ns:
            self._offsets.append(self._offsets[-1] + int(n))
        self._total = self._offsets[-1]
        self._cache: OrderedDict[int, dict[str, object]] = OrderedDict()

    def __len__(self) -> int:
        return self._total

    def _locate(self, i: int) -> tuple[int, int]:
        if not (0 <= i < self._total):
            raise IndexError(f"index {i} out of range for {self._total} samples")
        lo, hi = 0, len(self._offsets) - 1  # binary search: the largest k with offsets[k] <= i
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= i:
                lo = mid
            else:
                hi = mid - 1
        return lo, i - self._offsets[lo]

    def _slices(self, k: int) -> dict[str, object]:
        entry = self._cache.get(k)
        if entry is not None:
            self._cache.move_to_end(k)
            return entry
        f = safe_open(str(self.root / shard_text_name(k)), framework="pt", device="cpu")
        entry = {name: f.get_slice(name) for name in (TEXT_EMB, POOLED, ATTN_MASK)}
        self._cache[k] = entry
        if len(self._cache) > self.max_open:
            self._cache.popitem(last=False)  # evict the least recently used shard's handle
        return entry

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        k, row = self._locate(i)
        return {name: s[row] for name, s in self._slices(k).items()}


# --------------------------------------------------------------------------- #
# empty-prompt cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmptyPromptManifest:
    model_snapshot: str
    max_sequence_length: int
    dtype: str
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def write_empty_prompt(root: str | Path, prompt_embeds: torch.Tensor, pooled: torch.Tensor, *, model_snapshot: str, max_sequence_length: int) -> Path:
    """Write the empty-prompt cache and its fingerprint manifest into the split directory ``root``."""
    payload = save({TEXT_EMB: prompt_embeds.contiguous(), POOLED: pooled.contiguous()})
    path = atomic_write(Path(root) / EMPTY_PROMPT_FILE, lambda fh: fh.write(payload), binary=True)
    manifest = EmptyPromptManifest(
        model_snapshot=str(model_snapshot),
        max_sequence_length=int(max_sequence_length),
        dtype=str(prompt_embeds.dtype),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    atomic_write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), Path(root) / EMPTY_PROMPT_MANIFEST_FILE)
    return path


def has_empty_prompt(root: str | Path) -> bool:
    return (Path(root) / EMPTY_PROMPT_FILE).is_file()


def load_empty_prompt(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """``(prompt_embeds [L, D], pooled [D'])`` of the empty prompt of the split directory ``root``."""
    t = load_file(str(Path(root) / EMPTY_PROMPT_FILE))
    return t[TEXT_EMB], t[POOLED]


def read_empty_prompt_manifest(root: str | Path) -> EmptyPromptManifest:
    """The recorded fingerprint of the empty-prompt cache of ``root``; ``ValueError`` when the manifest is missing."""
    root = Path(root)
    manifest_path = root / EMPTY_PROMPT_MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"{root}: {EMPTY_PROMPT_FILE} exists but {EMPTY_PROMPT_MANIFEST_FILE} is missing; refusing to reuse it")
    with open(manifest_path, encoding="utf-8") as fh:
        return EmptyPromptManifest(**json.load(fh))


def validate_empty_prompt(root: str | Path, model_snapshot: str, max_sequence_length: int) -> None:
    """Raise ``ValueError`` unless the empty-prompt cache of ``root`` matches its manifest's sha256 and was produced from ``model_snapshot``
    with ``max_sequence_length``."""
    root = Path(root)
    recorded = read_empty_prompt_manifest(root)
    payload = (root / EMPTY_PROMPT_FILE).read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    mismatches = {}
    if recorded.sha256 != actual_sha256:
        mismatches["sha256"] = (recorded.sha256, actual_sha256)
    if recorded.model_snapshot != model_snapshot:
        mismatches["model_snapshot"] = (recorded.model_snapshot, model_snapshot)
    if recorded.max_sequence_length != int(max_sequence_length):
        mismatches["max_sequence_length"] = (recorded.max_sequence_length, max_sequence_length)
    if mismatches:
        raise ValueError(f"{root}: {EMPTY_PROMPT_FILE} fingerprint mismatch, refusing to use it: {mismatches}")
