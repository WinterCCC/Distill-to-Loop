"""The Z-Image teacher (``Tongyi-MAI/Z-Image``) in the local Hugging Face cache, at one pinned revision.

Every loader of this package reads the teacher from the directory of that revision's snapshot (:func:`teacher_dir`): the transformer, the VAE,
the Qwen3 text encoder with its tokenizer, and the scheduler config. Nothing is downloaded; a missing snapshot raises ``FileNotFoundError`` with
the download command.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

TEACHER_MODEL_ID = "Tongyi-MAI/Z-Image"
TEACHER_REVISION = "04cc4abb7c5069926f75c9bfde9ef43d49423021"  # the revision the student is distilled from
# A training checkpoint's extra[CHECKPOINT_IDENTITY_KEY]: {"teacher_revision": ..., "latent_shape": [C, H, W]} of the run.
CHECKPOINT_IDENTITY_KEY = "zimage"


def missing_teacher_message(revision: str, cache_dir: str | None = None) -> str:
    """How to get the teacher's snapshot ``revision`` into the cache the loaders read."""
    where = cache_dir or os.environ.get("HF_HUB_CACHE") or "the default Hugging Face cache"
    return (
        f"{TEACHER_MODEL_ID} at revision {revision} is not in the Hugging Face cache ({where}), and this code does not download it. Download it "
        f"with `hf download {TEACHER_MODEL_ID} --revision {revision}` (`huggingface-cli download ...` with huggingface_hub < 1.0; set HF_HUB_CACHE "
        "to download into another cache), and run again with that cache (HF_HUB_CACHE, --hf-cache or paths.hf_cache)"
    )


def teacher_dir(revision: str = TEACHER_REVISION, cache_dir: str | None = None) -> Path:
    """The local snapshot directory of the teacher at ``revision`` in the Hugging Face cache ``cache_dir`` (default: ``HF_HUB_CACHE`` or the
    default cache). ``FileNotFoundError`` (:func:`missing_teacher_message`) when the snapshot is not there."""
    from huggingface_hub import snapshot_download

    cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
    try:
        return Path(snapshot_download(TEACHER_MODEL_ID, revision=revision, cache_dir=cache_dir, local_files_only=True))
    except Exception as exc:  # huggingface_hub raises its own error types for a missing snapshot
        raise FileNotFoundError(missing_teacher_message(revision, cache_dir)) from exc


def scheduler_config(model_dir: str | Path) -> dict[str, Any]:
    """The teacher's ``FlowMatchEulerDiscreteScheduler`` config (``scheduler/scheduler_config.json`` of ``model_dir``)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    return dict(FlowMatchEulerDiscreteScheduler.load_config(str(model_dir), subfolder="scheduler", local_files_only=True))
