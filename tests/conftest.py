"""Shared test fixtures: CPU only, tiny random components, no weights downloaded."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The tests run on the CPU: hide every GPU before torch can initialize CUDA (a checkpoint save would otherwise read the CUDA RNG state).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _restore_global_torch_state():
    """Keep the global CPU RNG state of one test from leaking into the next."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)
