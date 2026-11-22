"""The distillation training loop. Launched by ``scripts/train.py`` under ``torchrun`` (one rank per GPU).

A run starts from the teacher: the student is built from the frozen bf16 teacher (:meth:`LoopedSD3Transformer.from_teacher`), its QK-norm gains
are calibrated (:mod:`distill_to_loop.training.calibration`) and the EMA is seeded from its weights. Every step draws ``n_data`` rows per rank
from the prepared split, adds the online rollout rows and takes the trajectory loss (:mod:`distill_to_loop.training.step`); then the gradient is
clipped, AdamW steps, the EMA follows and the learning rate advances (linear warm-up, then constant). Checkpoints hold the weights, the
optimizer and scheduler states, the EMA and the data position, plus one RNG file per rank, so a resume (``resume_from``) continues where the
uninterrupted run would be: exactly with ``data.num_workers=0``. With data loader workers the sample order is the same, but after a resume within
an epoch the latents the workers draw from the stored VAE posteriors differ from those of the uninterrupted run.

``global_step`` counts completed optimizer updates (0 before any); a ``max_steps=N`` run's last checkpoint is ``step_<N>.pt``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

from distill_to_loop.config import DistillConfig, global_batch_warning, lr_multiplier, online_rollout_split, to_dict, validate_config
from distill_to_loop.data.latents import LatentTextDataset
from distill_to_loop.data.text import load_empty_prompt, read_empty_prompt_manifest, resolve_snapshot, validate_empty_prompt
from distill_to_loop.hub import explain_missing_model
from distill_to_loop.models.attention import installed_versions
from distill_to_loop.models.looped_sd3 import LoopedSD3Transformer
from distill_to_loop.precision import text_dtype_for
from distill_to_loop.seeding import seed_everything
from distill_to_loop.training.calibration import build_qk_calibration_batch, calibrate_student_qk_norm, calibration_summary
from distill_to_loop.training.checkpoint import (
    EMA_CHECKPOINT_KEY,
    check_resume,
    check_resume_target,
    get_rng_state,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    set_rng_state,
)
from distill_to_loop.training.data_iter import EpochProgress, resumable_batches
from distill_to_loop.training.distributed import DistInfo, all_reduce_max, all_reduce_mean, wrap_for_ddp
from distill_to_loop.training.ema import init_ema_state, read_ema_state_dict, update_ema_state
from distill_to_loop.training.logger import create_logger
from distill_to_loop.training.objective import MAX_REDUCED_METRICS
from distill_to_loop.training.rollout import online_rollout_index
from distill_to_loop.training.step import build_trajectory_step_context, trajectory_training_loss

LOG_PREFIX = "[train]"


# --------------------------------------------------------------------------- #
# component construction
# --------------------------------------------------------------------------- #
def load_teacher(cfg: DistillConfig, device: torch.device) -> SD3Transformer2DModel:
    """The frozen bf16 teacher transformer: that of ``model.teacher_model_id``, or the one in ``model.teacher_transformer_path`` when set."""
    if cfg.model.teacher_transformer_path:
        teacher = SD3Transformer2DModel.from_pretrained(
            cfg.model.teacher_transformer_path, local_files_only=True, torch_dtype=torch.bfloat16, use_safetensors=True,
        ).to(device)
    else:
        with explain_missing_model(cfg.model.teacher_model_id, cfg.paths.hf_cache):
            teacher = SD3Transformer2DModel.from_pretrained(
                cfg.model.teacher_model_id, subfolder="transformer", cache_dir=cfg.paths.hf_cache, local_files_only=True, torch_dtype=torch.bfloat16,
            ).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def build_student(cfg: DistillConfig, teacher: SD3Transformer2DModel, device: torch.device) -> LoopedSD3Transformer:
    """The fp32 student initialized from ``teacher`` (:meth:`LoopedSD3Transformer.from_teacher`)."""
    student = LoopedSD3Transformer.from_teacher(teacher, n_pre=cfg.model.n_pre, n_loop=cfg.model.n_loop, n_post=cfg.model.n_post, n_rounds=cfg.model.n_rounds)
    return student.to(dtype=torch.float32, device=device)


def load_scheduler(cfg: DistillConfig) -> FlowMatchEulerDiscreteScheduler:
    """The teacher's training scheduler (its 1000-step sigma and timestep arrays)."""
    with explain_missing_model(cfg.model.teacher_model_id, cfg.paths.hf_cache):
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            cfg.model.teacher_model_id, subfolder="scheduler", cache_dir=cfg.paths.hf_cache, local_files_only=True
        )


def teacher_snapshot(cfg: DistillConfig) -> str:
    """The teacher's identity the empty-prompt cache records: the snapshot of ``model.teacher_model_id`` in the local Hugging Face cache."""
    return resolve_snapshot(cfg.model.teacher_model_id, cache_dir=cfg.paths.hf_cache)


def build_optimizer(student: torch.nn.Module, cfg: DistillConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(student.parameters(), lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2), weight_decay=cfg.optim.weight_decay)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up over ``warmup_steps`` updates, then a constant learning rate (:func:`distill_to_loop.config.lr_multiplier`)."""

    def lr_lambda(step: int) -> float:
        return lr_multiplier(step, warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_empty_prompt_checked(cfg: DistillConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The training split's empty-prompt encoding on ``device``, after checking that it was produced from the teacher snapshot this run loads."""
    train_dir = Path(cfg.data.train_dir)
    manifest = read_empty_prompt_manifest(train_dir)
    validate_empty_prompt(train_dir, teacher_snapshot(cfg), manifest.max_sequence_length)
    empty_prompt_embeds, empty_pooled = load_empty_prompt(train_dir)
    return empty_prompt_embeds.to(device), empty_pooled.to(device)


def batch_conditioning(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(latents, prompt_embeds, pooled)`` of a collated batch on ``device`` (the text cast by :func:`text_dtype_for`)."""
    dtype = text_dtype_for(device)
    return (
        batch["latent"].to(device, non_blocking=True),
        batch["prompt_embeds"].to(device, dtype=dtype, non_blocking=True),
        batch["pooled"].to(device, dtype=dtype, non_blocking=True),
    )


def _log(dist_info: DistInfo, message: str) -> None:
    if dist_info.is_main:
        print(f"{LOG_PREFIX} {message}", flush=True)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def run_training(cfg: DistillConfig, dist_info: DistInfo, resume_from: str | Path | None = None) -> None:
    """Train (module docstring). Without ``resume_from`` the run directory (``<output_dir>/<name>``) must hold no checkpoint; with it, the run
    continues from that checkpoint (``step_<N>.pt``, with its ``.rng`` directory next to it) and keeps writing into the run directory: the
    checkpoint's own directory, which must hold no checkpoint after step ``N``, or another one that holds no checkpoint and no W&B run id (resume
    an older checkpoint into a new run directory: another ``name`` or ``output_dir``)."""
    validate_config(cfg)
    device = dist_info.device
    seed_everything(cfg.seed, dist_info.rank)
    _log(dist_info, installed_versions())
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = latest_checkpoint(run_dir)
    if resume_from is None and existing is not None:
        raise RuntimeError(f"{run_dir} already holds {existing.name}: resume it with --resume-from {existing}, or pick another run name")
    if resume_from is not None:
        if not Path(resume_from).is_file():
            raise FileNotFoundError(f"--resume-from {resume_from}: no such checkpoint file")
        check_resume_target(run_dir, resume_from)

    teacher = load_teacher(cfg, device)
    student = build_student(cfg, teacher, device)
    student.loop_grad_checkpoint = cfg.train.loop_grad_checkpoint
    if cfg.train.block_grad_checkpoint:
        student.enable_gradient_checkpointing()  # diffusers' per-block checkpointing (sets the student's checkpointing function as well)
    scheduler = load_scheduler(cfg)

    ddp_model = wrap_for_ddp(student, device_ids=[device.index] if device.type == "cuda" else None) if dist_info.distributed else student
    forward_fn = ddp_model if dist_info.distributed else student.forward_rounds
    optimizer = build_optimizer(student, cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg.optim.warmup_steps)
    empty_prompt_embeds, empty_pooled = load_empty_prompt_checked(cfg, device)
    recorded = to_dict(cfg)
    logger = create_logger(
        cfg.wandb.enabled, dist_info.is_main, name=cfg.name, project=cfg.wandb.project, entity=cfg.wandb.entity, config=recorded, run_dir=run_dir,
        tags=list(cfg.wandb.tags), notes=cfg.wandb.notes,
    )

    # --- resume: weights, optimizer, scheduler, this rank's RNG, EMA and the data position ---
    start_step, start_epoch, start_batch = 0, 0, 0
    resume_rng_state: dict[str, Any] | None = None
    resumed_ema_state: dict[str, torch.Tensor] | None = None
    if resume_from is not None:
        meta = load_checkpoint(resume_from, student, optimizer, lr_scheduler, rank=dist_info.rank, seed=cfg.seed)
        # Re-applied right before the loop: building the dataset and the calibration batch below must not shift the restored streams.
        resume_rng_state = get_rng_state()
        start_step, extra = meta["step"], meta["extra"]
        if meta["rng_restored"] is False:
            print(f"{LOG_PREFIX} rank {dist_info.rank}: no RNG file next to {resume_from}; reseeded with seed {cfg.seed} + step {start_step} + rank", flush=True)
        check_resume(extra, meta["format_version"], start_step, recorded, dist_info.world_size, cfg.train.per_gpu_batch_size)
        resumed_ema_state = {k: v.to(device=device) for k, v in read_ema_state_dict(extra, resume_from).items()}
        if start_step > 0:
            start_epoch, start_batch = int(extra["data_progress"]["epoch"]), int(extra["data_progress"]["batch_in_epoch"])
        _log(dist_info, f"resumed {resume_from} at step {start_step} (epoch {start_epoch}, batch {start_batch})")

    online = cfg.online_rollout
    n_real, n_online, n_trajectories = online_rollout_split(cfg.train.per_gpu_batch_size, online.ratio, online.steps)
    _log(
        dist_info,
        f"per_gpu_batch_size={cfg.train.per_gpu_batch_size} = {n_real} data rows + {n_online} online rollout rows ({n_trajectories} trajectories x "
        f"{online.steps} Euler steps, wc={online.mix_w}, teacher CFG {online.teacher_cfg_scale}); global data rows per step "
        f"{n_real * dist_info.world_size}, lr {cfg.optim.lr:g}",
    )
    layout_warning = global_batch_warning(dist_info.world_size, cfg.train.per_gpu_batch_size, online.ratio, online.steps)
    if layout_warning is not None:
        _log(dist_info, layout_warning)
    train_dataset = LatentTextDataset(cfg.data.train_dir)

    # A fresh run calibrates the QK-norm gains, after the empty prompt is loaded and before the EMA is seeded (a resume restores them).
    if resume_from is None:
        calibration_batch = build_qk_calibration_batch(
            train_dataset, scheduler, empty_prompt_embeds, empty_pooled, cfg.loss.cfg_dropout_p, cfg.loss.logit_mean, cfg.loss.logit_std, device,
        )
        x_t, timesteps, prompt_embeds, pooled = calibration_batch
        dtype = text_dtype_for(device)
        calibration_batch = (x_t, timesteps, prompt_embeds.to(dtype=dtype), pooled.to(dtype=dtype))
        calibrated = calibrate_student_qk_norm(student, calibration_batch, cfg.model.n_rounds)
        _log(dist_info, calibration_summary(student, calibrated, calibration_batch[0].shape[0]))
        del calibration_batch
    ema_state = resumed_ema_state if resumed_ema_state is not None else init_ema_state(student)

    batches = resumable_batches(
        train_dataset, batch_size=n_real, world_size=dist_info.world_size, rank=dist_info.rank, seed=cfg.seed, num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda", start_epoch=start_epoch, start_batch=start_batch,
    )
    step_ctx = build_trajectory_step_context(cfg, forward_fn, student, teacher, scheduler, empty_prompt_embeds, empty_pooled, device)
    if resume_rng_state is not None:
        set_rng_state(resume_rng_state)

    rollout_seconds_sum, rollout_end_std_sum, rollout_iterations = 0.0, 0.0, 0  # the online rollouts since the last log line
    t_window, samples_window = time.time(), 0
    global_step = start_step
    progress = EpochProgress(epoch=start_epoch, batch_in_epoch=start_batch)
    while global_step < cfg.train.max_steps:
        batch, progress = next(batches)
        latents, prompt_embeds, pooled = batch_conditioning(batch, device)
        loss_out, rollout_stats = trajectory_training_loss(
            step_ctx, latents, prompt_embeds, pooled, rollout_index=online_rollout_index(global_step, dist_info.rank, dist_info.world_size),
        )
        if rollout_stats is not None:
            rollout_seconds_sum += rollout_stats.seconds
            rollout_end_std_sum += rollout_stats.end_std
            rollout_iterations += 1
        loss_out.total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.optim.grad_clip)
        used_lr = optimizer.param_groups[0]["lr"]  # the learning rate of this update, read before the scheduler advances it
        optimizer.step()
        update_ema_state(ema_state, student, cfg.ema.decay)
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        samples_window += loss_out.n * dist_info.world_size
        is_last_step = global_step == cfg.train.max_steps

        if global_step % cfg.train.log_every == 0 or is_last_step:
            elapsed = all_reduce_max(time.time() - t_window, dist_info.world_size, device)  # the slowest rank
            loss_mean = all_reduce_mean(loss_out.total.item(), dist_info.world_size, device)
            dropout_mean = all_reduce_mean(loss_out.dropped.float().mean().item(), dist_info.world_size, device)
            # Sorted keys: every rank must issue the same collectives in the same order.
            metric_means = {
                k: (all_reduce_max if k in MAX_REDUCED_METRICS else all_reduce_mean)(loss_out.metrics[k].item(), dist_info.world_size, device)
                for k in sorted(loss_out.metrics)
            }
            rollout_metrics: dict[str, float] = {}
            if rollout_iterations > 0:
                rollout_metrics["rollout/seconds"] = all_reduce_mean(rollout_seconds_sum / rollout_iterations, dist_info.world_size, device)
                rollout_metrics["rollout/end_std"] = all_reduce_mean(rollout_end_std_sum / rollout_iterations, dist_info.world_size, device)
                rollout_seconds_sum, rollout_end_std_sum, rollout_iterations = 0.0, 0.0, 0
            if dist_info.is_main:
                metrics = {
                    "train/loss": loss_mean,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": used_lr,
                    "train/cfg_dropout_rate": dropout_mean,
                    "train/samples_per_sec": samples_window / elapsed if elapsed > 0 else float("nan"),
                }
                if device.type == "cuda":
                    metrics["train/peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
                metrics.update({f"train/{k}": m for k, m in metric_means.items()})
                metrics.update(rollout_metrics)
                logger.log(metrics, step=global_step)
                mse_str = " ".join(f"{k}={m:.4f}" for k, m in metric_means.items())
                rollout_str = (
                    f" rollout_s={rollout_metrics['rollout/seconds']:.3f} end_std={rollout_metrics['rollout/end_std']:.4f}" if rollout_metrics else ""
                )
                print(f"[step {global_step}] loss={loss_mean:.4f} {mse_str} lr={used_lr:.2e} grad_norm={float(grad_norm):.3f} "
                      f"samples/s={metrics['train/samples_per_sec']:.1f}{rollout_str}", flush=True)
            t_window, samples_window = time.time(), 0

        if global_step % cfg.train.checkpoint_every == 0 or is_last_step:
            extra = {
                "config": recorded,
                "world_size": dist_info.world_size,
                "per_gpu_batch_size": cfg.train.per_gpu_batch_size,
                "data_progress": {"epoch": progress.epoch, "batch_in_epoch": progress.batch_in_epoch},
                EMA_CHECKPOINT_KEY: {"decay": cfg.ema.decay, "state_dict": ema_state},
            }
            path = save_checkpoint(
                run_dir, global_step, model=student, optimizer=optimizer, scheduler=lr_scheduler, extra=extra, keep_last=cfg.train.keep_last,
                rank=dist_info.rank,
            )
            _log(dist_info, f"step {global_step}: checkpoint {path}")
    logger.finish()
