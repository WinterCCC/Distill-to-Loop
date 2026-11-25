"""The tests exercise this checkout's package, not another installed copy, and run on the CPU."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import dooop

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_is_imported_from_this_checkout():
    assert Path(dooop.__file__).resolve().is_relative_to(REPO_ROOT)


def test_tests_run_on_the_cpu():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""  # set by conftest.py, whatever the caller exported
    assert not torch.cuda.is_available()
