"""Seeding: the global RNGs of a process and the stable seeds of the local random streams.

Every local stream is seeded by a sha256 of its arguments (never by Python's salted ``hash()``), so a resumed or restarted process draws exactly
what an uninterrupted one draws. Each derivation starts with a fixed tag string; the tags are those of the implementation the released model was
trained with, and changing one changes every draw of that stream.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

import numpy as np
import torch

ROLLOUT_STREAM_TAG = "looped_mmdit.proxy.opd"  # the online rollout noise (rollout_seed)
LOADER_STREAM_TAG = "looped_mmdit.data_iter"  # the data loader generator (loader_seed)
IMAGE_NOISE_TAG = "ldit-ngeneval"  # the sampling noise of one (prompt, sample) pair (image_noise_seed)


def seed_everything(seed: int, rank: int = 0) -> int:
    """Seed the Python, NumPy and PyTorch global RNGs with ``seed + rank`` and return that effective seed (every rank gets its own stream)."""
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def loader_seed(seed: int, epoch: int, rank: int) -> int:
    """The seed of the ``DataLoader`` generator of ``(seed, epoch, rank)``; it seeds the loader's worker processes."""
    digest = hashlib.sha256(f"{LOADER_STREAM_TAG}:{seed}:{epoch}:{rank}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def rollout_seed(seed: int, rollout_index: int) -> int:
    """The 64-bit seed of the online rollout ``rollout_index`` of a run with seed ``seed``."""
    digest = hashlib.sha256(f"{ROLLOUT_STREAM_TAG}:{seed}:{rollout_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def per_image_seed(seed: int, index: int) -> int:
    """A 64-bit per-row seed of ``(seed, index)``: sha256 of their decimal encoding joined by ``":"``, the first 8 bytes big-endian."""
    digest = hashlib.sha256(":".join(str(p) for p in (seed, index)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def per_image_noise(seed: int, indices: torch.Tensor, latent_shape: tuple[int, ...]) -> torch.Tensor:
    """``[len(indices), *latent_shape]`` float32 CPU noise, one generator per row seeded with :func:`per_image_seed`, so a row's noise does not
    depend on the other rows of the call."""
    rows = [torch.randn((1, *latent_shape), generator=torch.Generator(device="cpu").manual_seed(per_image_seed(seed, int(idx)))) for idx in indices.tolist()]
    return torch.cat(rows, dim=0)


def image_noise_seed(seed: int, prompt_index: int, sample_index: int) -> int:
    """The seed of the sampling noise of one ``(prompt, sample)`` pair: a stable hash of the three integers (63 bits), independent of the batch,
    the loop count and the process."""
    digest = hashlib.sha256(f"{IMAGE_NOISE_TAG}/{int(seed)}/{int(prompt_index)}/{int(sample_index)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def image_noise(seed: int, prompt_index: int, sample_index: int, shape: Sequence[int]) -> torch.Tensor:
    """The float32 CPU initial noise of one ``(prompt, sample)`` pair (:func:`image_noise_seed`)."""
    g = torch.Generator(device="cpu").manual_seed(int(image_noise_seed(seed, prompt_index, sample_index)))
    return torch.randn(tuple(shape), generator=g, dtype=torch.float32)
