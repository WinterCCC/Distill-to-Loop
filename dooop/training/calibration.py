"""Calibration of the QK-norm gains at the start of a run.

A fresh run sets every QK-norm gain to the pooled per-head RMS of the ``q`` / ``k`` (image stream) or ``add_q`` / ``add_k`` (text stream) its site
receives in a forward without QK-norm (pooled over the batch, the tokens and every call of the site: every round of a shared block and every
readout of a post block), broadcast over ``head_dim``. A token with a typical ``q`` / ``k`` norm then keeps the logits it had without QK-norm;
only a token with a larger norm is compressed. The forward is the training forward of the student (every round read out, bf16 autocast on
CUDA) without the depth-step update (``step_scale=False``), on a training-like batch built from the first :data:`QK_CALIBRATION_N` training
samples with a fixed seed. Every rank calibrates on the same batch, then rank 0's gains are broadcast. The global random streams are neither read
nor advanced.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler

from dooop.models.attention import QKNorm
from dooop.training.distributed import broadcast_tensors_
from dooop.training.objective import add_flow_noise, apply_cfg_dropout, run_student, sample_sigmas

QK_CALIBRATION_N = 64  # the first 64 samples of the training split
QK_CALIBRATION_SEED = 0  # the seed of the calibration batch's posterior noise, flow noise, sigmas and caption-dropout mask


def qk_norm_modules(model: nn.Module) -> dict[str, QKNorm]:
    """Every :class:`QKNorm` of ``model`` by module name (``loop.0.attn.qk_norm_img``, ...)."""
    return {name: module for name, module in model.named_modules() if isinstance(module, QKNorm)}


class _PooledSumSquares:
    """Per-head sums of squares of the ``q`` and ``k`` a site receives and the element count per head, accumulated in fp64."""

    def __init__(self, num_heads: int, device: torch.device) -> None:
        self.q_sumsq = torch.zeros(num_heads, dtype=torch.float64, device=device)
        self.k_sumsq = torch.zeros(num_heads, dtype=torch.float64, device=device)
        self.count = 0

    def add(self, q: torch.Tensor, k: torch.Tensor) -> None:
        self.q_sumsq += q.detach().float().square().sum(dim=(0, 2, 3)).double()
        self.k_sumsq += k.detach().float().square().sum(dim=(0, 2, 3)).double()
        self.count += q.numel() // q.shape[1]


class QKRmsRecorder:
    """Context manager for the calibration forward: makes every :class:`QKNorm` of ``model`` the identity and records, through a forward
    pre-hook on each, the pooled per-head RMS of the ``q`` and ``k`` it receives (:meth:`rms`). Hooks and bypass are removed on exit."""

    def __init__(self, model: nn.Module) -> None:
        self._modules = qk_norm_modules(model)
        if not self._modules:
            raise ValueError("the model has no QKNorm module to calibrate")
        self._sums: dict[str, _PooledSumSquares] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "QKRmsRecorder":
        if self._handles:
            raise RuntimeError("QKRmsRecorder is already active")
        self._sums = {}
        for name, module in self._modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(name, module)))
            module.bypass = True
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module in self._modules.values():
            module.bypass = False
        return False

    def _hook(self, name: str, module: QKNorm) -> Callable[[nn.Module, tuple], None]:
        def hook(_module: nn.Module, args: tuple) -> None:
            q, k = args
            if name not in self._sums:
                self._sums[name] = _PooledSumSquares(module.num_heads, q.device)
            self._sums[name].add(q, k)

        return hook

    def rms(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """``{module name: (q_rms, k_rms)}``, each ``[heads]`` fp64. Raises ``RuntimeError`` if a module was never called or a value is not
        finite."""
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in self._modules:
            sums = self._sums.get(name)
            if sums is None or sums.count == 0:
                raise RuntimeError(f"{name} was not called by the calibration forward, so it has no q/k statistics")
            q_rms, k_rms = (sums.q_sumsq / sums.count).sqrt(), (sums.k_sumsq / sums.count).sqrt()
            if not (torch.isfinite(q_rms).all() and torch.isfinite(k_rms).all()):
                raise RuntimeError(f"{name}: the calibration forward gave a non-finite q/k RMS")
            out[name] = (q_rms, k_rms)
        return out


def calibrate_qk_gains(model: nn.Module, forward: Callable[[], object]) -> list[str]:
    """Set every QK-norm gain of ``model`` to the pooled per-head RMS its site receives while ``forward()`` runs without QK-norm (no autograd),
    and return the state_dict keys written."""
    with QKRmsRecorder(model) as recorder, torch.no_grad():
        forward()
    stats = recorder.rms()
    written: list[str] = []
    with torch.no_grad():
        for name, module in qk_norm_modules(model).items():
            q_rms, k_rms = stats[name]
            module.q_gain.copy_(q_rms.unsqueeze(-1).expand_as(module.q_gain))
            module.k_gain.copy_(k_rms.unsqueeze(-1).expand_as(module.k_gain))
            written += [f"{name}.q_gain", f"{name}.k_gain"]
    return written


def build_qk_calibration_batch(
    dataset: torch.utils.data.Dataset,
    scheduler: FlowMatchEulerDiscreteScheduler,
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    cfg_dropout_p: float,
    logit_mean: float,
    logit_std: float,
    device: torch.device,
    n: int = QK_CALIBRATION_N,
    seed: int = QK_CALIBRATION_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x_t, timesteps, prompt_embeds, pooled)`` of the calibration batch: the first ``n`` samples of ``dataset`` turned into the inputs of a
    training step (posterior sample, flow noise, sigma draw, caption dropout). The items are read under ``torch.random.fork_rng(devices=[])``
    with the CPU default generator seeded with ``seed`` (the dataset draws its posterior noise from the global CPU stream, which the fork
    restores afterwards; no CUDA generator is touched), the rest is drawn from one CPU generator seeded with ``seed``."""
    n = min(n, len(dataset))
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        items = [dataset[i] for i in range(n)]
    latents = torch.stack([item["latent"] for item in items]).float()
    prompt_embeds = torch.stack([item["prompt_embeds"] for item in items])
    pooled = torch.stack([item["pooled"] for item in items])
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(latents.shape, generator=generator)
    sigma_sample = sample_sigmas(
        scheduler, n, ndim=latents.dim(), device="cpu", dtype=latents.dtype, logit_mean=logit_mean, logit_std=logit_std, generator=generator,
    )
    x_t = add_flow_noise(latents, noise, sigma_sample.sigmas)
    prompt_embeds, pooled, _ = apply_cfg_dropout(prompt_embeds, pooled, empty_prompt_embeds.cpu(), empty_pooled.cpu(), cfg_dropout_p, generator=generator)
    return x_t.to(device), sigma_sample.timesteps.to(device), prompt_embeds.to(device), pooled.to(device)


def calibrate_student_qk_norm(student: nn.Module, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], n_rounds: int) -> list[str]:
    """Set the QK-norm gains of ``student`` (unwrapped, every rank) from ``batch`` as the module docstring says, broadcast rank 0's gains, and
    return the state_dict keys written."""
    x_t, timesteps, prompt_embeds, pooled = batch
    written = calibrate_qk_gains(
        student,
        lambda: run_student(student.forward_rounds, x_t, timesteps, prompt_embeds, pooled, n_rounds=n_rounds, readout="all", forward_kwargs={"step_scale": False}),
    )
    broadcast_tensors_([p for module in qk_norm_modules(student).values() for p in (module.q_gain, module.k_gain)], src=0)
    return written


def calibration_summary(student: nn.Module, written: list[str], n_samples: int) -> str:
    """The one-line summary the trainer prints: site count, sample count, range and mean of the gains written."""
    state = student.state_dict()
    values = torch.cat([state[name].detach().float().flatten().cpu() for name in written])
    return (
        f"qk_norm: calibrated {len(written) // 2} QK-norm stream site(s) on the first {n_samples} train samples; gains in "
        f"[{values.min().item():.4g}, {values.max().item():.4g}], mean {values.mean().item():.4g}"
    )
