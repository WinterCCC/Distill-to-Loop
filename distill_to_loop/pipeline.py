"""``StableDiffusion3Pipeline`` with the looped student's sampling defaults (returned by :func:`distill_to_loop.inference.load_pipeline`)."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
from diffusers import StableDiffusion3Pipeline

from distill_to_loop.inference import TEXT_MAX_SEQUENCE_LENGTH  # the T5 prompt length of training (StableDiffusion3Pipeline's default is 256)
from distill_to_loop.precision import student_autocast


@contextlib.contextmanager
def _forward_under_autocast(module: torch.nn.Module, device: torch.device) -> Iterator[None]:
    """While open, every forward of ``module`` runs under :func:`~distill_to_loop.precision.student_autocast` of ``device``: an instance-level
    ``forward`` wraps the current one (the class's, or an instance-level one such as an offload hook) and is removed again on exit."""
    previous = vars(module).get("forward")
    inner = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        with student_autocast(device):
            return inner(*args, **kwargs)

    module.forward = forward
    try:
        yield
    finally:
        if previous is None:
            del module.forward
        else:
            module.forward = previous


class LoopedSD3Pipeline(StableDiffusion3Pipeline):
    """A ``StableDiffusion3Pipeline`` whose call defaults to the student's training settings: ``max_sequence_length=77`` (pass another value to
    override it), and on CUDA bf16 autocast (:func:`~distill_to_loop.precision.student_autocast`) around every transformer forward, where the
    fp32 student meets the bf16 text encodings, as in training. The text encoders and the VAE run without autocast, in their loaded dtype.
    On any other device the call is ``StableDiffusion3Pipeline``'s with ``max_sequence_length=77``."""

    def __call__(self, *args: Any, max_sequence_length: int = TEXT_MAX_SEQUENCE_LENGTH, **kwargs: Any) -> Any:
        with _forward_under_autocast(self.transformer, self._execution_device):
            return super().__call__(*args, max_sequence_length=max_sequence_length, **kwargs)
