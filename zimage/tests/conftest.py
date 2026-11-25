"""Shared test fixtures of the Z-Image tests: CPU only, tiny random components, no weights downloaded."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The tests run on the CPU: hide every GPU before torch can initialize CUDA (a checkpoint save would otherwise read the CUDA RNG state).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402

ZIMAGE_ROOT = Path(__file__).resolve().parents[1]
for _path in (ZIMAGE_ROOT.parent, ZIMAGE_ROOT, ZIMAGE_ROOT / "tests"):  # distill_to_loop, dooop_zimage, zimage_tiny
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture(autouse=True)
def _restore_global_torch_state():
    """Keep the global CPU RNG state of one test from leaking into the next."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)
