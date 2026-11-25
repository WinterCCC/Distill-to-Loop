"""Exponential moving average (EMA) of the student's weights, kept in fp32 next to the training weights.

The EMA lives in a checkpoint's ``extra["ema"] = {"decay": float, "state_dict": {name: fp32 tensor}}``, keyed like ``student.state_dict()``.
:func:`update_ema_state` uses ``Tensor.lerp_``: ``lerp(p, p, w) == p`` exactly, so re-deriving a step's update after a resume reproduces the EMA of
an uninterrupted run bit for bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from dooop.training.checkpoint import EMA_CHECKPOINT_KEY


def init_ema_state(student: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A detached fp32 copy of every entry of ``student.state_dict()`` (parameters and buffers): the EMA of a fresh run."""
    return {k: v.detach().to(dtype=torch.float32).clone() for k, v in student.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state: dict[str, torch.Tensor], student: torch.nn.Module, decay: float) -> None:
    """In-place EMA update after an optimizer step: a parameter becomes ``decay * ema + (1 - decay) * param`` (``lerp_``), a buffer is copied."""
    param_names = {name for name, _ in student.named_parameters()}
    for name, value in student.state_dict().items():
        source = value.detach().to(dtype=torch.float32)
        if name in param_names:
            ema_state[name].lerp_(source, 1.0 - decay)
        else:
            ema_state[name].copy_(source)


def read_ema_state_dict(extra: Mapping[str, Any], path: str | Path) -> dict[str, torch.Tensor]:
    """A fresh clone of the EMA state dict recorded in a checkpoint's ``extra``; ``RuntimeError`` naming ``path`` when there is none."""
    ema = extra.get(EMA_CHECKPOINT_KEY)
    if ema is None:
        raise RuntimeError(f"{path}: the checkpoint records no EMA weights (extra[{EMA_CHECKPOINT_KEY!r}])")
    return {k: v.clone() for k, v in ema["state_dict"].items()}
