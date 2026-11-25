"""The teacher snapshot in the Hugging Face cache; the package and the tests come from this checkout and run on the CPU."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import distill_to_loop
import dooop_zimage
import zimage_tiny as tiny
from dooop_zimage.teacher import TEACHER_REVISION, scheduler_config, teacher_dir

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_pinned_snapshot_is_read_from_the_cache(tmp_path, monkeypatch):
    cache = tiny.write_tiny_hub_cache(tmp_path / "cache")
    path = teacher_dir(cache_dir=str(cache))
    assert path.name == TEACHER_REVISION and path.is_dir()
    assert {k: scheduler_config(path)[k] for k in tiny.SCHEDULER_CONFIG} == tiny.SCHEDULER_CONFIG
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert teacher_dir() == path


def test_a_missing_snapshot_names_the_download_command(tmp_path):
    cache = tiny.write_tiny_hub_cache(tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="hf download Tongyi-MAI/Z-Image --revision 1111"):
        teacher_dir("1" * 40, cache_dir=str(cache))
    with pytest.raises(FileNotFoundError, match="not in the Hugging Face cache"):
        teacher_dir(cache_dir=str(tmp_path / "empty"))


def test_packages_are_imported_from_this_checkout_and_tests_run_on_the_cpu():
    assert Path(dooop_zimage.__file__).resolve().is_relative_to(REPO_ROOT / "zimage")
    assert Path(distill_to_loop.__file__).resolve().is_relative_to(REPO_ROOT)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_available()
