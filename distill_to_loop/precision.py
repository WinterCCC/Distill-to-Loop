"""The precision policy shared by training and sampling.

On CUDA the student keeps fp32 weights and runs under bf16 autocast, the teacher, the VAE and the text encoders run in bf16, and the stored text
encodings stay bf16. Other devices have no bf16 autocast here, so there the text encodings are cast to fp32 before the student reads them and
the sampling VAE runs in fp32.
"""

from __future__ import annotations

import torch


def student_autocast(device: torch.device | str) -> torch.autocast:
    """bf16 autocast around a student forward on CUDA; disabled on any other device."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda")


def text_dtype_for(device: torch.device | str) -> torch.dtype | None:
    """The dtype the text encodings are cast to before the student reads them: ``None`` (unchanged) on CUDA, fp32 elsewhere."""
    return None if torch.device(device).type == "cuda" else torch.float32


def vae_dtype_for(device: torch.device | str) -> torch.dtype:
    """The dtype of the sampling VAE: bf16 on CUDA, fp32 elsewhere."""
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
