"""The online rollout rows: states the current student visits, supervised by the teacher like the data rows.

At every training step each rank draws ``n`` fresh trajectories of the current training weights and adds every visited state to the step's
batch. A trajectory starts from Gaussian noise and takes ``S`` Euler steps on the teacher scheduler's inference grid (``set_timesteps(S)`` on a
copy of the scheduler config, its shift included: ``x_{i+1} = x_i + (sigma_{i+1} - sigma_i) v_i``). Each step's velocity mixes teacher and
student (S^2-Guidance, arXiv 2508.12880)::

    v_i = CFG_w(v_T) + (wc - 1) (v_T,c - v_S,c)

``CFG_w(v_T) = v_T,u + w (v_T,c - v_T,u)`` is the teacher's CFG velocity with the cached empty prompt (one doubled-batch teacher forward),
``v_T,c`` its conditional half and ``v_S,c`` the student's conditional velocity (``n_rounds`` rounds on the uniform depth grid, the inference
forward). ``wc == 1`` runs no student. The recorded rows are the ``S`` states each velocity was queried at, with the scheduler timestep of that
step and the trajectory's prompt; the final state (``sigma = 0``) is not supervised and gives ``end_std``.

The noise comes from local generators only (:func:`rollout_noise`), so the trainer's global random streams are untouched. The prompts are those
of the step's first ``n`` data rows (before caption dropout), so a resumed run redraws the rollouts an uninterrupted run draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from distill_to_loop.seeding import per_image_noise, rollout_seed
from distill_to_loop.training.objective import combine_guided_velocity, run_student, teacher_velocity_cond_uncond


@dataclass(frozen=True)
class RolloutGrid:
    """The rollout's inference grid: ``timesteps`` ``[S]`` (the model's time input of step ``i``) and ``sigmas`` ``[S + 1]`` (``sigmas[S] = 0``),
    float32 on the training device."""

    timesteps: torch.Tensor
    sigmas: torch.Tensor


def rollout_grid(scheduler_config: Mapping[str, Any], steps: int, device: torch.device) -> RolloutGrid:
    """The grid of a fresh ``FlowMatchEulerDiscreteScheduler`` built from ``scheduler_config`` with ``set_timesteps(steps)``; the training
    scheduler itself keeps its 1000-step arrays, which the data rows' sigma sampling indexes."""
    if steps < 1:
        raise ValueError(f"the rollout needs at least one Euler step, got {steps}")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler_config))
    scheduler.set_timesteps(steps, device=device)
    return RolloutGrid(
        timesteps=scheduler.timesteps.to(device=device, dtype=torch.float32),
        sigmas=scheduler.sigmas.to(device=device, dtype=torch.float32),
    )


def online_rollout_index(step: int, rank: int, world_size: int) -> int:
    """The index of the rollout of optimizer step ``step`` (0-based, the count of updates made before it) on ``rank``: ``step * world_size +
    rank``. Every ``(step, rank)`` has its own index, and the index depends on the step alone, so a resume redraws the same rollouts."""
    return step * world_size + rank


def rollout_noise(seed: int, rollout_index: int, n_trajectories: int, latent_shape: tuple[int, int, int]) -> torch.Tensor:
    """``[n_trajectories, *latent_shape]`` float32 CPU noise of the trajectories of rollout ``rollout_index``, independent of the global streams."""
    return per_image_noise(rollout_seed(seed, rollout_index), torch.arange(n_trajectories), latent_shape)


def s2_mix_velocity(v_guided: torch.Tensor, v_cond: torch.Tensor, v_weak: torch.Tensor, w: float) -> torch.Tensor:
    """``v_guided + (w - 1) * (v_cond - v_weak)``: the teacher's CFG velocity, its conditional half, the student's conditional velocity, ``wc``."""
    return v_guided + (w - 1.0) * (v_cond - v_weak)


def s2_rollout_velocity(
    student_call: Callable[..., list[torch.Tensor]],
    teacher: torch.nn.Module,
    x: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    n_rounds: int,
    mix_w: float,
    teacher_cfg_scale: float,
) -> torch.Tensor:
    """One Euler step's velocity ``CFG_w(v_T) + (wc - 1) (v_T,c - v_S,c)`` (fp32) at the states ``x``; ``student_call`` is the student's
    ``forward_rounds``, run for its last readout of ``n_rounds`` rounds (conditional branch only)."""
    v_cond, v_uncond = teacher_velocity_cond_uncond(teacher, x, timestep, prompt_embeds, pooled, empty_prompt_embeds, empty_pooled)
    v_guided = combine_guided_velocity(v_cond, v_uncond, torch.full((x.shape[0],), float(teacher_cfg_scale), device=x.device))
    if mix_w == 1.0:
        return v_guided
    v_student = run_student(student_call, x, timestep, prompt_embeds, pooled, n_rounds=n_rounds, readout="last")[0].float()
    return s2_mix_velocity(v_guided, v_cond, v_student, mix_w)


@dataclass(frozen=True)
class OnlineRollout:
    """The rows of one rollout, trajectory-major (trajectory ``j``'s ``S`` states are rows ``j S .. j S + S - 1``): ``x_t`` ``[n S, C, H, W]``
    float32, ``timesteps`` ``[n S]``, ``prompt_embeds`` / ``pooled`` the trajectory's prompt on each of its rows, and ``end_std`` the standard
    deviation of the final states over every trajectory and element."""

    x_t: torch.Tensor
    timesteps: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled: torch.Tensor
    end_std: float


@torch.no_grad()
def generate_online_rollout(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    grid: RolloutGrid,
    noise: torch.Tensor,
    n_rounds: int,
    mix_w: float,
    teacher_cfg_scale: float,
) -> OnlineRollout:
    """The rollout of the module docstring from ``noise`` (``[n, C, H, W]``) and the ``n`` prompts. ``student`` is the unwrapped training
    student (its current weights, eval mode for the call and restored after; no gradient)."""
    n = noise.shape[0]
    if prompt_embeds.shape[0] != n or pooled.shape[0] != n:
        raise ValueError(f"{n} trajectories need {n} prompts, got prompt_embeds {tuple(prompt_embeds.shape)} and pooled {tuple(pooled.shape)}")
    steps = grid.timesteps.shape[0]
    was_training = student.training
    student.eval()
    states: list[torch.Tensor] = []
    try:
        x = noise.to(device=grid.sigmas.device, dtype=torch.float32)
        for i in range(steps):
            states.append(x)
            t = grid.timesteps[i].expand(n)
            v = s2_rollout_velocity(
                student.forward_rounds, teacher, x, t, prompt_embeds, pooled, empty_prompt_embeds, empty_pooled, n_rounds, mix_w, teacher_cfg_scale,
            )
            x = x + (grid.sigmas[i + 1] - grid.sigmas[i]) * v
    finally:
        student.train(was_training)
    return OnlineRollout(
        x_t=torch.stack(states, dim=1).flatten(0, 1),
        timesteps=grid.timesteps.repeat(n),
        prompt_embeds=prompt_embeds.repeat_interleave(steps, dim=0),
        pooled=pooled.repeat_interleave(steps, dim=0),
        end_std=float(x.float().std()),
    )
