"""The Z-Image specifics of a training step: the data rows' sigma set and the rollout grid."""

from __future__ import annotations

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, get_default_z_image_sigmas, retrieve_timesteps

import zimage_tiny as tiny
from dooop_zimage.step import rollout_grid, sample_data_sigmas, sigma_table, training_sigma_set


def test_training_sigma_set_is_the_rounded_shift_6_grid():
    sigmas = training_sigma_set(tiny.SCHEDULER_CONFIG)
    reference = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0).sigmas
    assert len(sigmas) == 1000 and sigmas[0] == 1.0 and sigmas[-1] == 0.005970149
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    assert all(abs(a - float(b)) <= 5e-8 * max(1.0, abs(a)) for a, b in zip(sigmas, reference))
    assert all(len(f"{s:.15g}".lstrip("0.").rstrip("0")) <= 7 for s in sigmas)  # at most 7 significant digits


def test_data_sigmas_come_from_the_set():
    table = sigma_table(training_sigma_set(tiny.SCHEDULER_CONFIG), 1000, "cpu")
    torch.manual_seed(0)
    sigmas, timesteps = sample_data_sigmas(table, 64, ndim=4, dtype=torch.float32)
    torch.manual_seed(0)
    again, _ = sample_data_sigmas(table, 64, ndim=4, dtype=torch.float32)
    assert sigmas.shape == (64, 1, 1, 1) and torch.equal(sigmas, again)  # the global stream draws the indices
    assert torch.isin(sigmas.flatten(), table.sigmas).all()
    assert torch.equal(timesteps, sigmas.flatten() * 1000)


def test_rollout_grid_is_the_pipelines_grid():
    steps = 8
    grid = rollout_grid(tiny.SCHEDULER_CONFIG, steps, "cpu")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(tiny.SCHEDULER_CONFIG)
    mu = calculate_shift(1024)  # ZImagePipeline passes it; the static-shift config does not read it
    timesteps, _ = retrieve_timesteps(scheduler, steps, "cpu", sigmas=get_default_z_image_sigmas(steps), mu=mu)
    assert torch.equal(grid.timesteps, timesteps.float()) and torch.equal(grid.sigmas, scheduler.sigmas.float())
    assert grid.sigmas.shape == (steps + 1,) and grid.sigmas[0] == 1.0 and grid.sigmas[-1] == 0.0


def test_dynamic_shifting_is_refused():
    with pytest.raises(ValueError, match="static-shift"):
        rollout_grid({**tiny.SCHEDULER_CONFIG, "use_dynamic_shifting": True}, 8, "cpu")
