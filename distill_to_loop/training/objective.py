"""The training objective: noising, caption dropout, depth coordinates and the per-round trajectory loss.

Every round ``k`` of the student completes one stretch of a straight path in velocity space towards the teacher velocity ``v_T`` instead of
regressing ``v_T`` again. With ``u_0`` the readout of the pre blocks' output (no shared block) and ``u_k`` the readout after round ``k``::

    coordinates s_0 = 0 < s_1 < ... < s_{K-1} < s_K = 1     K - 1 sorted FP64 U(0, 1) draws spread by eps = 1e-6, per sample
    targets     t_k = sg(u_{k-1}) + a_k (v_T - sg(u_{k-1})),   a_k = (s_k - s_{k-1}) / (1 - s_{k-1})
    loss        L = (1 / K) sum_k mean ||u_k - t_k||^2  +  lambda mean ||u_0 - v_T||^2

Round ``k`` reads ``s_k`` through the depth embedding and moves the streams by ``alpha_k = s_k - s_{k-1}``. Every squared error is averaged over
the elements of the whole batch. Precision: the student keeps fp32 weights and runs under bf16 autocast on CUDA (no autocast on the CPU); the
teacher runs in its native bf16 without autocast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from distill_to_loop.models.looped_sd3 import step_alphas
from distill_to_loop.precision import student_autocast

DEPTH_COORDINATE_EPS = 1e-6  # the minimum spacing of the depth coordinates


# --------------------------------------------------------------------------- #
# noising and caption dropout
# --------------------------------------------------------------------------- #
@dataclass
class SigmaSample:
    """``sigmas`` broadcastable against the latent; ``timesteps`` ``[B]``; ``indices`` into the scheduler's training arrays."""

    sigmas: torch.Tensor
    timesteps: torch.Tensor
    indices: torch.Tensor


def sample_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    batch_size: int,
    ndim: int,
    device: torch.device | str,
    dtype: torch.dtype,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    generator: torch.Generator | None = None,
) -> SigmaSample:
    """One (sigma, timestep) pair per row, the convention of diffusers' SD3 training examples: ``u ~ logit-normal(logit_mean, logit_std)``,
    ``idx = floor(u * num_train_timesteps)``, ``timestep = scheduler.timesteps[idx]``, ``sigma = scheduler.sigmas[idx]`` (the arrays already
    hold the scheduler's shift). ``sigmas`` is right-padded with singleton dims to broadcast against an ``ndim``-dimensional latent."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        device=device,
        generator=generator,
    )
    num_train_timesteps = scheduler.config.num_train_timesteps
    indices = (u * num_train_timesteps).long().clamp_(0, num_train_timesteps - 1)
    timesteps = scheduler.timesteps.to(device=device)[indices]
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[indices].flatten()
    while sigmas.dim() < ndim:
        sigmas = sigmas.unsqueeze(-1)
    return SigmaSample(sigmas=sigmas, timesteps=timesteps, indices=indices)


def add_flow_noise(x0: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """``x_t = (1 - sigma) * x0 + sigma * noise``."""
    return (1.0 - sigmas) * x0 + sigmas * noise


def apply_cfg_dropout(
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    p: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(prompt_embeds, pooled, dropped)`` with every row replaced by the empty-prompt encoding independently with probability ``p``. The mask is
    drawn with ``torch.rand`` on the CPU (``generator``, default the global CPU stream), so it does not depend on the device."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    batch = prompt_embeds.shape[0]
    if p <= 0.0:
        return prompt_embeds, pooled, torch.zeros(batch, dtype=torch.bool)
    dropped = torch.rand(batch, generator=generator) < p
    dropped = dropped.to(prompt_embeds.device)
    out_embeds = torch.where(
        dropped.view(-1, *([1] * (prompt_embeds.dim() - 1))),
        empty_prompt_embeds.to(prompt_embeds.dtype).unsqueeze(0).expand_as(prompt_embeds),
        prompt_embeds,
    )
    out_pooled = torch.where(
        dropped.view(-1, *([1] * (pooled.dim() - 1))),
        empty_pooled.to(pooled.dtype).unsqueeze(0).expand_as(pooled),
        pooled,
    )
    return out_embeds, out_pooled, dropped


# --------------------------------------------------------------------------- #
# model calls
# --------------------------------------------------------------------------- #
@torch.no_grad()
def teacher_velocity(teacher: torch.nn.Module, x_t: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
    """The teacher's velocity (a plain forward in the teacher's dtype, no autocast), returned in fp32."""
    v = teacher(
        hidden_states=x_t.to(teacher.dtype),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds.to(teacher.dtype),
        pooled_projections=pooled.to(teacher.dtype),
        return_dict=False,
    )[0]
    return v.float()


def teacher_velocity_cond_uncond(
    teacher: torch.nn.Module,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(v_cond, v_uncond)`` (fp32) from one doubled-batch teacher forward on ``[uncond; cond]``; the empty-prompt encoding is broadcast."""
    b = x_t.shape[0]
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([timestep, timestep], dim=0)
    empty_embeds = empty_prompt_embeds.to(prompt_embeds.dtype).unsqueeze(0).expand(b, *empty_prompt_embeds.shape)
    empty_pooled_b = empty_pooled.to(pooled.dtype).unsqueeze(0).expand(b, *empty_pooled.shape)
    embeds_in = torch.cat([empty_embeds, prompt_embeds], dim=0)
    pooled_in = torch.cat([empty_pooled_b, pooled], dim=0)
    v = teacher_velocity(teacher, x_in, t_in, embeds_in, pooled_in)
    v_uncond, v_cond = v[:b], v[b:]
    return v_cond, v_uncond


def combine_guided_velocity(v_cond: torch.Tensor, v_uncond: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``v_uncond + g * (v_cond - v_uncond)`` in fp32, ``g`` a per-row ``[B]`` tensor."""
    g = g.to(device=v_cond.device, dtype=torch.float32).view(-1, *([1] * (v_cond.dim() - 1)))
    return v_uncond.float() + g * (v_cond.float() - v_uncond.float())


def run_student(
    call: Callable[..., list[torch.Tensor]],
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    n_rounds: int,
    readout: Literal["all", "last"] = "all",
    forward_kwargs: Mapping[str, Any] | None = None,
) -> list[torch.Tensor]:
    """``call(x_t, timestep, prompt_embeds, pooled, n_rounds=..., readout=..., **forward_kwargs)`` under bf16 autocast on CUDA (no autocast on
    any other device). ``call`` is the student's ``forward_rounds`` or its DDP wrapper."""
    kwargs: dict[str, Any] = {"n_rounds": n_rounds, "readout": readout}
    if forward_kwargs:
        kwargs.update(forward_kwargs)
    with student_autocast(x_t.device):
        return call(x_t, timestep, prompt_embeds, pooled, **kwargs)


# --------------------------------------------------------------------------- #
# depth coordinates and the trajectory loss
# --------------------------------------------------------------------------- #
def sample_depth_coordinates(batch_size: int, n_rounds: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    """``[B, K]`` float32 depth coordinates ``s_1 .. s_K``: per sample ``K - 1`` FP64 ``U(0, 1)`` draws (one ``torch.rand`` on ``generator``,
    default the global stream of ``device``), sorted and spread to ``s_i = (1 - K eps) r_(i) + i eps`` in FP64, cast to float32, then
    ``s_K = 1``. Consecutive coordinates are at least ``eps`` apart and none sits on an endpoint."""
    draws = torch.rand(batch_size, n_rounds - 1, dtype=torch.float64, device=device, generator=generator)
    eps = DEPTH_COORDINATE_EPS
    rank = torch.arange(1, n_rounds, dtype=torch.float64, device=draws.device)
    interior = (1.0 - n_rounds * eps) * draws.sort(dim=1).values + rank * eps
    return torch.cat([interior.float(), torch.ones(draws.shape[0], 1, dtype=torch.float32, device=draws.device)], dim=1)


def _per_sample(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``values`` (``[B]``) viewed as ``[B, 1, ..., 1]`` to broadcast against ``like``."""
    return values.view(-1, *[1] * (like.dim() - 1))


def previous_anchor_coefficients(depth_s: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` float32 ``a_k = (s_k - s_{k-1}) / (1 - s_{k-1})`` with ``s_0 = 0`` (``a_1 = s_1``, ``a_K = 1``). Raises ``ValueError`` when
    some ``s_{k-1} >= 1``."""
    coords = depth_s.float()
    previous = torch.cat([torch.zeros_like(coords[:, :1]), coords[:, :-1]], dim=1)
    if not bool((previous < 1.0).all()):
        raise ValueError("every depth coordinate s_1 .. s_{K-1} must be < 1 (a_k divides by 1 - s_{k-1})")
    return (coords - previous) / (1.0 - previous)


def trajectory_targets(u0: torch.Tensor, u_rounds: list[torch.Tensor], v_teacher: torch.Tensor, depth_s: torch.Tensor) -> list[torch.Tensor]:
    """The per-round targets ``t_k = sg(u_{k-1}) + a_k (v_T - sg(u_{k-1}))`` (fp32, round 1 first; ``u_{k-1}`` is ``u_0`` for round 1)."""
    coefficients = previous_anchor_coefficients(depth_s)
    anchors = [u0, *u_rounds[:-1]]
    end = v_teacher.float()
    return [
        anchor.detach().float() + _per_sample(coefficients[:, k], end) * (end - anchor.detach().float()) for k, anchor in enumerate(anchors)
    ]


@dataclass
class TrajectoryLoss:
    """``total = trajectory_mse + prelude_weight * prelude_mse``; ``per_round_mse`` are the readouts' MSEs against ``v_T`` (values only)."""

    total: torch.Tensor
    trajectory_mse: torch.Tensor
    prelude_mse: torch.Tensor
    per_round_mse: list[torch.Tensor]


def trajectory_loss(u0: torch.Tensor, u_rounds: list[torch.Tensor], v_teacher: torch.Tensor, depth_s: torch.Tensor, prelude_weight: float) -> TrajectoryLoss:
    """The loss of the module docstring; ``u0`` the prelude readout, ``u_rounds`` the ``K`` round readouts, ``depth_s`` their ``[B, K]``
    coordinates."""
    if depth_s.shape[1] != len(u_rounds):
        raise ValueError(f"depth_s has {depth_s.shape[1]} coordinates per sample, but there are {len(u_rounds)} round readouts")
    targets = trajectory_targets(u0, u_rounds, v_teacher, depth_s)
    trajectory_mse = torch.stack([F.mse_loss(u.float(), target) for u, target in zip(u_rounds, targets)]).mean()
    prelude_mse = F.mse_loss(u0.float(), v_teacher.float())
    with torch.no_grad():
        per_round_mse = [F.mse_loss(u.float(), v_teacher.float()) for u in u_rounds]
    return TrajectoryLoss(total=trajectory_mse + prelude_weight * prelude_mse, trajectory_mse=trajectory_mse, prelude_mse=prelude_mse, per_round_mse=per_round_mse)


# Metric keys the trainer reduces across ranks with the maximum instead of the mean.
MAX_REDUCED_METRICS = frozenset({"alpha_max"})


@dataclass
class StepLossResult:
    """One step's loss ``total``, its detached ``metrics``, the caption-dropout mask ``dropped`` and the row count ``n``."""

    total: torch.Tensor
    metrics: dict[str, torch.Tensor]
    dropped: torch.Tensor
    n: int


def compute_trajectory_step_loss(
    forward_fn: Callable[..., list[torch.Tensor]],
    teacher: torch.nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    n_rounds: int,
    cfg_dropout_p: float,
    prelude_weight: float,
) -> StepLossResult:
    """The loss of one step on a batch (data rows and rollout rows alike), in this order: the caption dropout over the whole batch, the per-row
    depth coordinates (one FP64 ``torch.rand`` on the global stream of the batch's device), one teacher forward at the dropped conditioning (a
    dropped row's target is the teacher's empty-prompt velocity), one student forward with those coordinates and the prelude readout, then
    :func:`trajectory_loss`. Metrics: ``round_{k}_mse``, ``prelude_mse``, ``trajectory_mse``, ``alpha_max`` and ``alpha_mean`` of the depth
    steps."""
    bsz = x_t.shape[0]
    prompt_embeds_d, pooled_d, dropped = apply_cfg_dropout(prompt_embeds, pooled, empty_prompt_embeds, empty_pooled, cfg_dropout_p)
    depth_s = sample_depth_coordinates(bsz, n_rounds, x_t.device)
    v_teacher = teacher_velocity(teacher, x_t, timesteps, prompt_embeds_d, pooled_d)
    outputs = run_student(
        forward_fn, x_t, timesteps, prompt_embeds_d, pooled_d, n_rounds=n_rounds, readout="all",
        forward_kwargs={"depth_s": depth_s, "include_prelude": True},
    )
    out = trajectory_loss(outputs[0], outputs[1:], v_teacher, depth_s, prelude_weight)
    metrics = {f"round_{k + 1}_mse": m for k, m in enumerate(out.per_round_mse)}
    metrics["prelude_mse"] = out.prelude_mse.detach()
    metrics["trajectory_mse"] = out.trajectory_mse.detach()
    alpha = step_alphas(depth_s)
    metrics["alpha_max"], metrics["alpha_mean"] = alpha.max(), alpha.mean()
    return StepLossResult(total=out.total, metrics=metrics, dropped=dropped, n=bsz)
