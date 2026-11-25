"""One training step's loss: the data rows, the online rollout rows and the trajectory loss on their union.

The Z-Image specifics on top of :mod:`dooop.training` (whose trajectory loss and online rollout this module reuses unchanged):

* Sigmas of the data rows. Each data row draws its sigma uniformly from the 1000 training sigmas of the teacher's scheduler
  (``FlowMatchEulerDiscreteScheduler``, shift 6, no dynamic shifting), rounded to 7 significant digits (:func:`training_sigma_set`); the
  timestep is ``sigma * 1000``, the value the pipeline's scheduler gives the model.
* Grid of the online rollout. The trajectories step on the pipeline's own grid: ``set_timesteps(sigmas=linspace(1, 1 / S, S))`` on the
  teacher's scheduler config (:func:`rollout_grid`).

The order of the operations (and so of the random draws) is:

1. the ``n_data`` data rows are noised (``torch.randn_like``, then one ``torch.randint`` for the sigma indices, both on the global stream of
   the device);
2. with online rollout rows, :func:`dooop.training.rollout.generate_online_rollout` (local generators only) on the prompts of the
   first ``n_trajectories`` data rows; its rows follow the data rows in the batch;
3. :func:`dooop.training.objective.compute_trajectory_step_loss` on the whole batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from dooop.training.objective import StepLossResult, add_flow_noise, compute_trajectory_step_loss
from dooop.training.rollout import RolloutGrid, generate_online_rollout, rollout_noise
from dooop.training.step import RolloutStats
from dooop_zimage.config import ZImageDistillConfig, rollout_rows

SIGMA_SET_DIGITS = 7  # significant digits of the training sigmas, as recorded in the released run's config


# --------------------------------------------------------------------------- #
# the sigmas of the data rows and the rollout grid
# --------------------------------------------------------------------------- #
def training_sigma_set(scheduler_config: Mapping[str, Any]) -> list[float]:
    """The ``num_train_timesteps`` (1000) training sigmas of a ``FlowMatchEulerDiscreteScheduler`` built from the teacher's ``scheduler_config``
    (shift 6: 1.0 down to 0.005970149), each rounded to :data:`SIGMA_SET_DIGITS` significant digits."""
    sigmas = FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler_config)).sigmas
    return [float(f"{float(s):.{SIGMA_SET_DIGITS}g}") for s in sigmas]


@dataclass(frozen=True)
class SigmaTable:
    """The sigmas the data rows draw from (``[N]`` float32) and their timesteps ``sigma * num_train_timesteps`` (``[N]`` float32), on the
    training device."""

    sigmas: torch.Tensor
    timesteps: torch.Tensor


def sigma_table(sigma_set: Sequence[float], num_train_timesteps: int, device: torch.device | str) -> SigmaTable:
    sigmas = torch.tensor([float(s) for s in sigma_set], dtype=torch.float32, device=device)
    return SigmaTable(sigmas=sigmas, timesteps=sigmas * num_train_timesteps)


def sample_data_sigmas(table: SigmaTable, batch_size: int, ndim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sigmas, timesteps)`` of ``batch_size`` rows, each drawn uniformly from ``table`` (one ``torch.randint`` on the global stream of the
    table's device); ``sigmas`` in ``dtype``, right-padded with singleton dims to broadcast against an ``ndim``-dimensional latent."""
    indices = torch.randint(len(table.sigmas), (batch_size,), device=table.sigmas.device)
    sigmas = table.sigmas.to(dtype=dtype)[indices].flatten()
    while sigmas.dim() < ndim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas, table.timesteps[indices]


def rollout_grid(scheduler_config: Mapping[str, Any], steps: int, device: torch.device | str) -> RolloutGrid:
    """The grid ``ZImagePipeline`` samples on: a fresh ``FlowMatchEulerDiscreteScheduler`` from ``scheduler_config`` after
    ``set_timesteps(sigmas=linspace(1, 1 / steps, steps))``: ``steps + 1`` sigmas ending in 0 and ``steps`` timesteps (``sigma * 1000``), float32
    on ``device``. A dynamic-shifting config is refused (the pipeline would pass a resolution-dependent shift)."""
    if steps < 1:
        raise ValueError(f"the rollout needs at least one step, got {steps}")
    if scheduler_config.get("use_dynamic_shifting"):
        raise ValueError("the rollout grid is the static-shift one (scheduler_config.use_dynamic_shifting is true)")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler_config))
    scheduler.set_timesteps(sigmas=torch.linspace(1.0, 1.0 / steps, steps).tolist(), device=device)
    return RolloutGrid(timesteps=scheduler.timesteps.to(device=device, dtype=torch.float32), sigmas=scheduler.sigmas.to(device=device, dtype=torch.float32))


# --------------------------------------------------------------------------- #
# the step
# --------------------------------------------------------------------------- #
@dataclass
class StepContext:
    """Everything a step reads that does not change from step to step. ``forward_fn`` is the training call (the DDP wrapper, or the student's
    ``forward_rounds``); ``student`` is the unwrapped student the rollout runs; ``grid`` is ``None`` without online rollout rows."""

    cfg: ZImageDistillConfig
    forward_fn: Callable[..., list[torch.Tensor]]
    student: torch.nn.Module
    teacher: torch.nn.Module
    sigma_table: SigmaTable
    empty_prompt_embeds: torch.Tensor
    empty_pooled: torch.Tensor
    grid: RolloutGrid | None
    n_data: int
    n_rollout: int
    n_trajectories: int


def build_step_context(
    cfg: ZImageDistillConfig,
    forward_fn: Callable[..., list[torch.Tensor]],
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    scheduler_config: Mapping[str, Any],
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    device: torch.device,
) -> StepContext:
    """The per-GPU row split of ``cfg``, the data rows' sigma table and, with online rollout rows, the rollout grid, from the teacher's
    ``scheduler_config``."""
    n_data, n_rollout, n_trajectories = rollout_rows(cfg)
    table = sigma_table(training_sigma_set(scheduler_config), int(scheduler_config["num_train_timesteps"]), device)
    grid = rollout_grid(scheduler_config, cfg.online_rollout.steps, device) if n_rollout > 0 else None
    return StepContext(
        cfg=cfg, forward_fn=forward_fn, student=student, teacher=teacher, sigma_table=table, empty_prompt_embeds=empty_prompt_embeds,
        empty_pooled=empty_pooled, grid=grid, n_data=n_data, n_rollout=n_rollout, n_trajectories=n_trajectories,
    )


def training_loss(
    ctx: StepContext, latents: torch.Tensor, prompt_embeds: torch.Tensor, pooled: torch.Tensor, rollout_index: int,
) -> tuple[StepLossResult, RolloutStats | None]:
    """The loss of one step (module docstring) on the ``n_data`` data rows (on the training device; ``pooled`` the captions' validity masks),
    with the rollout ``rollout_index`` (:func:`dooop.training.rollout.online_rollout_index`). Returns the step result over every row and
    the rollout's statistics (``None`` without online rollout rows)."""
    cfg = ctx.cfg
    if latents.shape[0] != ctx.n_data:
        raise ValueError(f"the step takes {ctx.n_data} data rows, got {latents.shape[0]}")
    device = latents.device
    noise = torch.randn_like(latents)
    sigmas, timesteps = sample_data_sigmas(ctx.sigma_table, latents.shape[0], ndim=latents.dim(), dtype=latents.dtype)
    x_t = add_flow_noise(latents, noise, sigmas)
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
