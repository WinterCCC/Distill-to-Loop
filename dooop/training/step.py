"""One training step's loss: the data rows, the online rollout rows and the trajectory loss on their union.

The order of the operations (and so of the random draws) is:

1. the ``n_data`` data rows are noised (``torch.randn_like`` and the logit-normal sigma draw, both on the global stream of the device);
2. with online rollout rows, the rollout of :mod:`dooop.training.rollout` (local generators only) on the prompts of the first
   ``n_trajectories`` data rows; its rows follow the data rows in the batch;
3. :func:`~dooop.training.objective.compute_trajectory_step_loss` on the whole batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from dooop.config import DistillConfig, online_rollout_split
from dooop.training.objective import StepLossResult, add_flow_noise, compute_trajectory_step_loss, sample_sigmas
from dooop.training.rollout import RolloutGrid, generate_online_rollout, rollout_grid, rollout_noise


@dataclass(frozen=True)
class RolloutStats:
    """What the log reports of one step's rollout: its wall time in seconds and the spread of the final states."""

    seconds: float
    end_std: float


@dataclass
class TrajectoryStepContext:
    """Everything a step reads that does not change from step to step. ``forward_fn`` is the training call (the DDP wrapper, or the student's
    ``forward_rounds``); ``student`` is the unwrapped student the rollout runs; ``grid`` is ``None`` without online rollout rows."""

    cfg: DistillConfig
    forward_fn: Callable[..., list[torch.Tensor]]
    student: torch.nn.Module
    teacher: torch.nn.Module
    scheduler: FlowMatchEulerDiscreteScheduler
    empty_prompt_embeds: torch.Tensor
    empty_pooled: torch.Tensor
    grid: RolloutGrid | None
    n_data: int
    n_rollout: int
    n_trajectories: int


def build_trajectory_step_context(
    cfg: DistillConfig,
    forward_fn: Callable[..., list[torch.Tensor]],
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    device: torch.device,
) -> TrajectoryStepContext:
    """The per-GPU row split of ``cfg`` and, with online rollout rows, the rollout grid on a copy of ``scheduler``'s config."""
    rollout = cfg.online_rollout
    n_data, n_rollout, n_trajectories = online_rollout_split(cfg.train.per_gpu_batch_size, rollout.ratio, rollout.steps)
    grid = rollout_grid(scheduler.config, rollout.steps, device) if n_rollout > 0 else None
    return TrajectoryStepContext(
        cfg=cfg, forward_fn=forward_fn, student=student, teacher=teacher, scheduler=scheduler, empty_prompt_embeds=empty_prompt_embeds,
        empty_pooled=empty_pooled, grid=grid, n_data=n_data, n_rollout=n_rollout, n_trajectories=n_trajectories,
    )


def trajectory_training_loss(
    ctx: TrajectoryStepContext, latents: torch.Tensor, prompt_embeds: torch.Tensor, pooled: torch.Tensor, rollout_index: int,
) -> tuple[StepLossResult, RolloutStats | None]:
    """The loss of one step (module docstring) on the ``n_data`` data rows (on the training device), with the rollout ``rollout_index``
    (:func:`dooop.training.rollout.online_rollout_index`). Returns the step result over every row and the rollout's statistics
    (``None`` without online rollout rows)."""
    cfg = ctx.cfg
    if latents.shape[0] != ctx.n_data:
        raise ValueError(f"the step takes {ctx.n_data} data rows, got {latents.shape[0]}")
    device = latents.device
    noise = torch.randn_like(latents)
    sigma_sample = sample_sigmas(
        ctx.scheduler, latents.shape[0], ndim=latents.dim(), device=device, dtype=latents.dtype,
        logit_mean=cfg.loss.logit_mean, logit_std=cfg.loss.logit_std,
    )
    x_t = add_flow_noise(latents, noise, sigma_sample.sigmas)
    timesteps = sigma_sample.timesteps
    stats: RolloutStats | None = None
    if ctx.grid is not None:
        n = ctx.n_trajectories
        t0 = time.time()
        rollout = generate_online_rollout(
            ctx.student, ctx.teacher, prompt_embeds[:n], pooled[:n], ctx.empty_prompt_embeds, ctx.empty_pooled, ctx.grid,
            rollout_noise(cfg.seed, rollout_index, n, tuple(latents.shape[1:])).to(device), cfg.model.n_rounds,
            cfg.online_rollout.mix_w, cfg.online_rollout.teacher_cfg_scale,
        )
        stats = RolloutStats(seconds=time.time() - t0, end_std=rollout.end_std)
        x_t = torch.cat([x_t, rollout.x_t.to(dtype=x_t.dtype)], dim=0)
        timesteps = torch.cat([timesteps, rollout.timesteps.to(dtype=timesteps.dtype)], dim=0)
        prompt_embeds = torch.cat([prompt_embeds, rollout.prompt_embeds], dim=0)
        pooled = torch.cat([pooled, rollout.pooled], dim=0)
    result = compute_trajectory_step_loss(
        ctx.forward_fn, ctx.teacher, x_t, timesteps, prompt_embeds, pooled, ctx.empty_prompt_embeds, ctx.empty_pooled,
        n_rounds=cfg.model.n_rounds, cfg_dropout_p=cfg.loss.cfg_dropout_p, prelude_weight=cfg.loss.prelude_weight,
    )
    return result, stats
