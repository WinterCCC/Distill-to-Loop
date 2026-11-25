"""The stable seed derivations are pinned: their values are those of the implementation the released model was trained and evaluated with, so
a resumed run, the online rollout noise and the sampling noise stay reproducible across versions of this code."""

from __future__ import annotations

import pytest
import torch

from dooop.seeding import image_noise, image_noise_seed, loader_seed, per_image_seed, rollout_seed
from dooop.training.rollout import rollout_noise


def test_loader_seeds_are_pinned():
    assert [loader_seed(0, 0, 0), loader_seed(0, 3, 1), loader_seed(7, 81, 1)] == [5492433658012198700, 6305529153139948670, 1560221636610292957]


def test_rollout_seeds_are_pinned():
    assert [rollout_seed(0, 0), rollout_seed(0, 199999), rollout_seed(3, 5)] == [12596080704937847349, 13742150487482822562, 18318286073548603226]
    assert [per_image_seed(5, 0), per_image_seed(5, 7)] == [12959297840682230952, 9485080730621892585]
    assert float(rollout_noise(0, 12, 2, (2, 3, 3)).double().sum()) == pytest.approx(-2.035566883161664, abs=1e-12)


def test_image_noise_seeds_are_pinned():
    assert [image_noise_seed(0, 0, 0), image_noise_seed(0, 552, 3), image_noise_seed(42, 1, 2)] == [2533947581246294800, 8266331062587246316, 7934752685702110445]
    assert float(image_noise(0, 3, 1, (2, 3, 3)).double().sum()) == pytest.approx(6.9356072917580605, abs=1e-12)


def test_rollout_noise_rows_do_not_depend_on_the_batch():
    both = rollout_noise(1, 4, 3, (4, 2, 2))
    assert torch.equal(both[:2], rollout_noise(1, 4, 2, (4, 2, 2)))
