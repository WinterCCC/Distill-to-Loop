"""A teacher missing from the Hugging Face cache: every loader says how to download it; other errors pass through."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

import dooop.inference as inference
import dooop.training.trainer as trainer
from dooop.config import load_config
from dooop.data.text import Sd3PromptEncoder
from dooop.hub import explain_missing_model
from dooop.inference import load_vae_and_scheduler_factory, teacher_transformer_config

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
RELEASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sd3m_dooop.yaml"


def test_a_missing_teacher_explains_the_download(tmp_path: Path):
    empty = str(tmp_path / "empty_hub_cache")
    cfg = load_config(RELEASE_CONFIG, ["data.train_dir=/data/train", f"paths.hf_cache={empty}"])
    loaders = {
        "teacher transformer": lambda: trainer.load_teacher(cfg, torch.device("cpu")),
        "scheduler": lambda: trainer.load_scheduler(cfg),
        "snapshot": lambda: trainer.teacher_snapshot(cfg),
        "export config": lambda: teacher_transformer_config({"teacher_model_id": MODEL_ID}, empty),
        "VAE": lambda: load_vae_and_scheduler_factory(MODEL_ID, "cpu", cache_dir=empty),
        "text encoders": lambda: Sd3PromptEncoder(MODEL_ID, cache_dir=empty),
    }
    for name, load in loaders.items():
        with pytest.raises(FileNotFoundError) as info:
            load()
        message = str(info.value)
        for part in (f"{MODEL_ID} is not in the Hugging Face cache ({empty})", "license", "hf auth login", f"hf download {MODEL_ID}",
                     f"huggingface-cli download {MODEL_ID}"):
            assert part in message, (name, part, message)


def test_the_pipeline_loaders_raise_no_torch_dtype_deprecation(tmp_path: Path, monkeypatch):
    """The text encoders and ``load_pipeline`` pass diffusers' ``dtype=`` (``torch_dtype=`` warns that it is deprecated). diffusers reads the
    argument before it looks in the cache, so an empty cache is enough to see the warning."""
    empty = str(tmp_path / "empty_hub_cache")
    monkeypatch.setattr(inference, "load_student", lambda *args, **kwargs: None)  # the student is not what is checked here
    loaders = (lambda: Sd3PromptEncoder(MODEL_ID, cache_dir=empty), lambda: inference.load_pipeline("unused", device="cpu", model_id=MODEL_ID, cache_dir=empty))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for load in loaders:
            with pytest.raises(FileNotFoundError, match="is not in the Hugging Face cache"):
                load()
    assert not [str(w.message) for w in caught if "torch_dtype" in str(w.message)]


def test_other_errors_pass_through(tmp_path: Path):
    with pytest.raises(KeyError, match="unrelated"):
        with explain_missing_model(str(tmp_path)):  # a local directory counts as present
            raise KeyError("unrelated")
    with pytest.raises(ValueError, match="unrelated"):
        with explain_missing_model("not a valid hub id!", str(tmp_path)):
            raise ValueError("unrelated")
