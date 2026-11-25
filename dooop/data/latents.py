"""Sharded storage of pre-extracted VAE latents and the training dataset.

Layout of a prepared split directory::

    index.json                  shard list, totals, latent shape, VAE scaling/shift factor
    captions.json               de-duplicated caption list; sample i has caption captions[caption_idx[i]]
    meta.jsonl                  one JSON object per sample (image id, file name, annotation id, caption)
    shard_00000.safetensors     latent_mean [n, C, h, w] fp16, latent_std [n, C, h, w] fp16, caption_idx [n] int32
    shard_00000_text.safetensors   the prompt encodings of the shard's rows (dooop.data.text)
    shard_00000.safetensors.done   commit marker, written after both shard files
    empty_prompt.safetensors (+ empty_prompt_manifest.json)

Shard ``k`` holds samples ``[k * shard_size, min((k + 1) * shard_size, total))``. Every file is written atomically. The dataset samples the VAE
posterior ``mean + std * eps`` on every access (``eps`` from the global CPU RNG) and returns ``(sample - shift_factor) * scaling_factor``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors.torch import safe_open, save
from torch.utils.data import Dataset

from dooop.data.text import POOLED, TEXT_EMB, ShardedLazyTextCache
from dooop.io_utils import atomic_write, atomic_write_text

INDEX_FILE = "index.json"
CAPTIONS_FILE = "captions.json"
META_FILE = "meta.jsonl"
COMMIT_MARKER_SUFFIX = ".done"
LATENT_MEAN, LATENT_STD, CAPTION_IDX = "latent_mean", "latent_std", "caption_idx"


def shard_name(k: int) -> str:
    return f"shard_{int(k):05d}.safetensors"


def shard_marker_name(k: int) -> str:
    return shard_name(k) + COMMIT_MARKER_SUFFIX


def num_shards(total: int, shard_size: int) -> int:
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    return (total + shard_size - 1) // shard_size


def shard_range(k: int, total: int, shard_size: int) -> range:
    return range(k * shard_size, min((k + 1) * shard_size, total))


@dataclass
class LatentIndex:
    """``index.json`` of a split. The training dataset supports the layout :mod:`scripts.prepare_coco` writes: ``text_sharded`` true, one caption
    per image, VAE posteriors (``deterministic`` and ``latents_normalized`` false), stored text encodings (``text_on_the_fly`` false)."""

    total: int
    shard_size: int
    latent_shape: list[int]
    scaling_factor: float
    shift_factor: float = 0.0
    shards: list[dict[str, Any]] = field(default_factory=list)  # [{"file": str, "n": int}]
    vae: str = ""
    text_encoder: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False
    latents_normalized: bool = False
    text_sharded: bool = False
    captions_per_image: int = 1
    text_on_the_fly: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LatentIndex":
        return cls(**dict(d))


# --------------------------------------------------------------------------- #
# writers and readers
# --------------------------------------------------------------------------- #
def write_shard(path: str | Path, latent_mean: torch.Tensor, latent_std: torch.Tensor, caption_idx: torch.Tensor) -> Path:
    n = latent_mean.shape[0]
    if latent_std.shape != latent_mean.shape or caption_idx.shape != (n,):
        raise ValueError(f"shape mismatch: mean {tuple(latent_mean.shape)} std {tuple(latent_std.shape)} idx {tuple(caption_idx.shape)}")
    payload = save(
        {
            LATENT_MEAN: latent_mean.detach().to("cpu", torch.float16).contiguous(),
            LATENT_STD: latent_std.detach().to("cpu", torch.float16).contiguous(),
            CAPTION_IDX: caption_idx.detach().to("cpu", torch.int32).contiguous(),
        }
    )
    return atomic_write(path, lambda fh: fh.write(payload), binary=True)


def write_shard_marker(root: str | Path, k: int) -> Path:
    return atomic_write_text(json.dumps({"shard": int(k)}) + "\n", Path(root) / shard_marker_name(k))


def has_shard_marker(root: str | Path, k: int) -> bool:
    return (Path(root) / shard_marker_name(k)).is_file()


def write_captions(root: str | Path, captions: Sequence[str]) -> Path:
    return atomic_write_text(json.dumps(list(captions), ensure_ascii=False, indent=0), Path(root) / CAPTIONS_FILE)


def read_captions(root: str | Path) -> list[str]:
    with open(Path(root) / CAPTIONS_FILE, encoding="utf-8") as fh:
        return list(json.load(fh))


def write_meta(root: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    text = "".join(json.dumps(dict(r), ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    return atomic_write_text(text, Path(root) / META_FILE)


def write_index(root: str | Path, index: LatentIndex) -> Path:
    return atomic_write_text(json.dumps(index.to_dict(), indent=2), Path(root) / INDEX_FILE)


def read_index(root: str | Path) -> LatentIndex:
    with open(Path(root) / INDEX_FILE, encoding="utf-8") as fh:
        return LatentIndex.from_dict(json.load(fh))


class LazyLatentShardCache:
    """Memory-mapped, per-row access to ``latent_mean`` / ``latent_std`` across the shard files; at most ``max_open`` files stay open (least
    recently used eviction). The state is per instance, so every ``DataLoader`` worker opens its own files after the fork."""

    def __init__(self, root: str | Path, shard_files: Sequence[str], shard_ns: Sequence[int], max_open: int = 8) -> None:
        self.root = Path(root)
        self.shard_files = list(shard_files)
        self.max_open = max(1, int(max_open))
        self._offsets = [0]
        for n in shard_ns:
            self._offsets.append(self._offsets[-1] + int(n))
        self._total = self._offsets[-1]
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

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

    def _slices(self, k: int) -> dict[str, Any]:
        entry = self._cache.get(k)
        if entry is not None:
            self._cache.move_to_end(k)
            return entry
        f = safe_open(str(self.root / self.shard_files[k]), framework="pt", device="cpu")
        entry = {LATENT_MEAN: f.get_slice(LATENT_MEAN), LATENT_STD: f.get_slice(LATENT_STD)}
        self._cache[k] = entry
        if len(self._cache) > self.max_open:
            self._cache.popitem(last=False)  # evict the least recently used shard's handle
        return entry

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, row = self._locate(i)
        s = self._slices(k)
        return s[LATENT_MEAN][row], s[LATENT_STD][row]


# --------------------------------------------------------------------------- #
# the training dataset
# --------------------------------------------------------------------------- #
class LatentTextDataset(Dataset):
    """The training view of a prepared split directory: item ``{"latent": float32 [C, h, w], "prompt_embeds": [L, D], "pooled": [D'],
    "index": int}``. The latent is a fresh posterior sample on every access (``mean + std * eps``, ``eps`` from the global CPU RNG), then
    ``(z - shift_factor) * scaling_factor``. Latents and text encodings are memory-mapped per shard (at most ``*_max_open`` files open)."""

    def __init__(self, split_dir: str | Path, text_shard_max_open: int = 8, latent_shard_max_open: int = 8) -> None:
        self.root = Path(split_dir)
        self.index = read_index(self.root)
        unsupported = {
            name: getattr(self.index, name)
            for name, supported in (("deterministic", False), ("latents_normalized", False), ("text_sharded", True), ("captions_per_image", 1),
                                    ("text_on_the_fly", False))
            if getattr(self.index, name) != supported
        }
        if unsupported:
            raise ValueError(f"{self.root}: unsupported split layout {unsupported} (prepare the split with sd3/scripts/prepare_coco.py or zimage/scripts/prepare_coco.py)")
        self.scaling_factor = float(self.index.scaling_factor)
        self.shift_factor = float(self.index.shift_factor)
        shard_files = [e["file"] for e in self.index.shards]
        shard_ns = [int(e["n"]) for e in self.index.shards]
        self._latents = LazyLatentShardCache(self.root, shard_files, shard_ns, max_open=latent_shard_max_open)
        self.text = ShardedLazyTextCache(self.root, shard_ns, max_open=text_shard_max_open)
        if len(self._latents) != self.index.total or len(self.text) != self.index.total:
            raise ValueError(f"{self.root}: index.json says {self.index.total} samples, the shards hold {len(self._latents)} latents and {len(self.text)} text rows")

    def __len__(self) -> int:
        return int(self.index.total)

    def latent(self, i: int) -> torch.Tensor:
        mean_t, std_t = self._latents[i]
        mean = mean_t.float()
        z = mean + std_t.float() * torch.randn(mean.shape)
        return (z - self.shift_factor) * self.scaling_factor

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        latent = self.latent(i)  # drawn first: the posterior noise comes from the global CPU RNG
        row = self.text[i]
        return {"latent": latent, "prompt_embeds": row[TEXT_EMB], "pooled": row[POOLED], "index": int(i)}
