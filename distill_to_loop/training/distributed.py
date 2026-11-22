"""Process-group set-up under ``torchrun`` (or a single process) and the few collectives the trainer uses.

Every helper is a no-op outside a multi-process run, so callers need not branch on the world size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def init_distributed(device: str | None = None, timeout_minutes: int = 30) -> DistInfo:
    """Read ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` (set by ``torchrun``; a single process otherwise), select this rank's device (``cuda:<local
    rank>`` when CUDA is available, else the CPU; ``device="cpu"`` forces the CPU) and create the process group when the world size is above 1
    (``nccl`` on CUDA, ``gloo`` on the CPU)."""
    rank, world_size, local_rank = _env_int("RANK", 0), _env_int("WORLD_SIZE", 1), _env_int("LOCAL_RANK", 0)
    use_cuda = torch.cuda.is_available() and device != "cpu"
    if use_cuda:
        dev = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=timeout_minutes)
        )
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank, device=dev)


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def cleanup() -> None:
    if is_initialized():
        dist.destroy_process_group()


def all_reduce_all_ok(ok: bool, device: torch.device | None = None) -> bool:
    """``True`` only if every rank passed ``ok=True``. Every rank must call this together; it turns a per-rank failure into a decision all ranks
    reach together instead of leaving the others waiting at a barrier. ``device`` is this rank's CUDA device under ``nccl`` (ignored under
    ``gloo``)."""
    if not is_initialized():
        return bool(ok)
    reduce_device = device if (device is not None and dist.get_backend() != "gloo") else torch.device("cpu")
    t = torch.tensor([1.0 if ok else 0.0], dtype=torch.float64, device=reduce_device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item() >= 1.0)


def broadcast_tensors_(tensors: Sequence[torch.Tensor], src: int = 0) -> None:
    """Overwrite ``tensors`` in place with rank ``src``'s values, in one collective (flattened into one buffer of their common dtype)."""
    if not is_initialized() or not tensors:
        return
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) != 1:
        raise ValueError(f"broadcast_tensors_ needs tensors of one dtype, got {sorted(str(d) for d in dtypes)}")
    with torch.no_grad():
        flat = torch.cat([t.detach().reshape(-1) for t in tensors])
        dist.broadcast(flat, src=src)
        offset = 0
        for t in tensors:
            t.copy_(flat[offset : offset + t.numel()].view_as(t))
            offset += t.numel()


def _all_reduce(value: float, op: "dist.ReduceOp", device: torch.device) -> float:
    if not is_initialized():
        return float(value)
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    return float(t.item())


def all_reduce_mean(value: float, world_size: int, device: torch.device) -> float:
    """The mean of one detached scalar per rank."""
    if world_size <= 1:
        return float(value)
    return _all_reduce(value, dist.ReduceOp.SUM, device) / world_size


def all_reduce_max(value: float, world_size: int, device: torch.device) -> float:
    """The maximum of one detached scalar per rank."""
    if world_size <= 1:
        return float(value)
    return _all_reduce(value, dist.ReduceOp.MAX, device)


# --------------------------------------------------------------------------- #
# DDP around the multi-round forward
# --------------------------------------------------------------------------- #
class ForwardRoundsWrapper(nn.Module):
    """``forward(*args, **kwargs)`` is ``model.forward_rounds(*args, **kwargs)``, so that ``DistributedDataParallel`` wraps the training entry
    point and its gradient hooks run (calling ``ddp.module.forward_rounds`` directly would bypass them)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward_rounds(*args, **kwargs)


def wrap_for_ddp(model: nn.Module, device_ids: list[int] | None = None) -> DistributedDataParallel:
    """``DistributedDataParallel`` over :class:`ForwardRoundsWrapper`: calling the returned module runs ``forward_rounds`` under DDP's hooks.
    Every parameter receives a gradient in a training step (also the text projections of the last post block, whose output rows are discarded:
    their gradient is zero-valued but present), so ``find_unused_parameters`` stays off."""
    return DistributedDataParallel(ForwardRoundsWrapper(model), device_ids=device_ids, find_unused_parameters=False)
