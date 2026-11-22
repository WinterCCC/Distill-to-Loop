"""Resumable epoch / batch iteration over the training dataset.

:func:`resumable_batches` tracks ``(epoch, batch_in_epoch)`` and, on resume, starts at the right batch by slicing the epoch's index list and
handing the remainder to a fresh ``DataLoader`` (the skipped batches are never loaded). The index list of an epoch is a deterministic function of
``(seed, epoch, rank, world_size)`` (``DistributedSampler`` used as an index generator, so no process group is needed), and every loader gets its
own generator seeded from ``(seed, epoch, rank)`` (:func:`distill_to_loop.seeding.loader_seed`), so building a loader never touches the global
RNG, however often a resume rebuilds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from distill_to_loop.seeding import loader_seed


@dataclass(frozen=True)
class EpochProgress:
    """A resume point: ``batch_in_epoch`` batches of ``epoch`` have been consumed. Saved verbatim and passed back as ``start_epoch`` /
    ``start_batch``."""

    epoch: int
    batch_in_epoch: int


def epoch_indices(dataset: Dataset, world_size: int, rank: int, seed: int, epoch: int) -> list[int]:
    """This rank's shuffled sample indices of ``epoch`` (``drop_last=False``: a short epoch is padded so every rank gets the same count)."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
    sampler.set_epoch(epoch)
    return list(sampler)


def resumable_batches(
    dataset: Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    start_epoch: int = 0,
    start_batch: int = 0,
) -> Iterator[tuple[dict[str, Any], EpochProgress]]:
    """Yield ``(batch, progress)`` forever from ``(start_epoch, start_batch)``. ``progress`` is the state after the yielded batch, so saving it
    and resuming from its two fields continues with the next batch. Each epoch drops its final partial batch."""
    epoch = start_epoch
    batch_in_epoch = start_batch
    while True:
        indices = epoch_indices(dataset, world_size, rank, seed, epoch)
        n_batches = len(indices) // batch_size
        if n_batches == 0:
            raise ValueError(f"the dataset gives this rank {len(indices)} samples per epoch, fewer than one batch of {batch_size}")
        indices = indices[: n_batches * batch_size]
        if batch_in_epoch > n_batches:
            raise ValueError(
                f"start_batch={batch_in_epoch} but epoch {epoch} only has {n_batches} batches (world_size={world_size}, batch_size={batch_size})"
            )
        if batch_in_epoch == n_batches:  # the resume point is an epoch boundary: no loader for an empty remainder
            epoch += 1
            batch_in_epoch = 0
            continue
        remaining = indices[batch_in_epoch * batch_size :]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=remaining,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            generator=torch.Generator().manual_seed(loader_seed(seed, epoch, rank)),
        )
        for batch in loader:
            batch_in_epoch += 1
            yield batch, EpochProgress(epoch=epoch, batch_in_epoch=batch_in_epoch)
        epoch += 1
        batch_in_epoch = 0
