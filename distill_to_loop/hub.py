"""The teacher model in the local Hugging Face cache.

Every loader of this package reads the cache only (``local_files_only=True``) and never downloads. When a load fails because the model is not in
the cache, :func:`explain_missing_model` replaces the loader's error with one that says how to download the model.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

MODEL_INDEX = "model_index.json"


def missing_model_message(model_id: str, cache_dir: str | None = None) -> str:
    """How to get ``model_id`` into the cache that the loaders read."""
    where = cache_dir or os.environ.get("HF_HUB_CACHE") or "the default Hugging Face cache"
    return (
        f"{model_id} is not in the Hugging Face cache ({where}), and this code does not download it. Accept the model's license on "
        f"https://huggingface.co/{model_id}, log in (`hf auth login`), download the model with `hf download {model_id}` (`huggingface-cli download "
        f"{model_id}` with huggingface_hub < 1.0; set HF_HUB_CACHE to download into another cache), and run again with that cache (HF_HUB_CACHE, "
        "--hf-cache or paths.hf_cache)"
    )


def is_cached(model_id: str, cache_dir: str | None = None) -> bool:
    """Whether ``model_id`` (a Hub id) has a snapshot in the cache; ``True`` for a local directory and for anything that is not a Hub id."""
    from huggingface_hub import try_to_load_from_cache

    if Path(model_id).exists():
        return True
    try:
        return isinstance(try_to_load_from_cache(model_id, MODEL_INDEX, cache_dir=cache_dir or os.environ.get("HF_HUB_CACHE")), str)
    except ValueError:  # not a valid Hub id: the loader's own error is the better one
        return True


@contextmanager
def explain_missing_model(model_id: str, cache_dir: str | None = None) -> Iterator[None]:
    """Run a load of ``model_id``; when it raises and the model is not in the cache, raise ``FileNotFoundError`` with
    :func:`missing_model_message` instead. Any other error passes through unchanged."""
    try:
        yield
    except Exception as exc:
        if is_cached(model_id, cache_dir):
            raise
        raise FileNotFoundError(missing_model_message(model_id, cache_dir)) from exc
