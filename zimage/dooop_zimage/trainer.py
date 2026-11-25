"""The distillation training loop of the Z-Image student. Launched by ``zimage/scripts/train.py`` under ``torchrun`` (one rank per GPU).

A run starts from the teacher: the student is built from the frozen bf16 teacher (:meth:`LoopedZImageTransformer.from_teacher`) and the EMA is
seeded from its weights. Every step draws ``n_data`` rows per rank from the prepared split, adds the online rollout rows and takes the
trajectory loss (:mod:`dooop_zimage.step`); then the gradient is clipped, AdamW steps, the EMA follows and the learning rate advances (linear
warm-up, then constant). Checkpoints (:mod:`distill_to_loop.training.checkpoint`) hold the weights, the optimizer and scheduler states, the EMA,
the data position and the run's Z-Image identity (the teacher revision and the latent shape of the split, ``extra["zimage"]``), plus one RNG file
per rank, so a resume (``resume_from``) continues where the uninterrupted run would be: exactly with ``data.num_workers=0``. With data loader
workers the sample order is the same, but after a resume within an epoch the latents the workers draw from the stored VAE posteriors differ from
those of the uninterrupted run.

``global_step`` counts completed optimizer updates (0 before any); a ``max_steps=N`` run's last checkpoint is ``step_<N>.pt``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import diffusers
import torch

from distill_to_loop.config import lr_multiplier, to_dict
from distill_to_loop.data.latents import LatentTextDataset
from distill_to_loop.data.text import load_empty_prompt, read_empty_prompt_manifest, validate_empty_prompt
from distill_to_loop.precision import text_dtype_for
from distill_to_loop.seeding import seed_everything
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
from dooop_zimage.config import ZImageDistillConfig, global_batch_warning, rollout_rows, validate_config
from dooop_zimage.model import LoopedZImageTransformer, ZImageTeacherAdapter
from dooop_zimage.step import build_step_context, training_loss
from dooop_zimage.teacher import CHECKPOINT_IDENTITY_KEY, scheduler_config, teacher_dir

LOG_PREFIX = "[train]"


# --------------------------------------------------------------------------- #
# component construction
# --------------------------------------------------------------------------- #
def load_teacher(cfg: ZImageDistillConfig, device: torch.device) -> ZImageTeacherAdapter:
    """The frozen teacher: the bf16 transformer of the teacher snapshot ``model.teacher_revision``, ``eval()``, no gradient, behind
    :class:`ZImageTeacherAdapter`."""
    from diffusers import ZImageTransformer2DModel

    transformer = ZImageTransformer2DModel.from_pretrained(
        str(teacher_dir(cfg.model.teacher_revision, cfg.paths.hf_cache)), subfolder="transformer", local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    transformer.requires_grad_(False)
    transformer.eval()
    return ZImageTeacherAdapter(transformer).eval()


def load_scheduler_config(cfg: ZImageDistillConfig) -> dict[str, Any]:
    """The teacher's scheduler config (the data rows' sigmas and the rollout grid)."""
    return scheduler_config(teacher_dir(cfg.model.teacher_revision, cfg.paths.hf_cache))


def build_student(cfg: ZImageDistillConfig, teacher: ZImageTeacherAdapter, device: torch.device) -> LoopedZImageTransformer:
    """The fp32 student initialized from ``teacher`` (:meth:`LoopedZImageTransformer.from_teacher`)."""
    student = LoopedZImageTransformer.from_teacher(
        teacher.transformer, n_pre=cfg.model.n_pre, n_loop=cfg.model.n_loop, n_post=cfg.model.n_post, n_rounds=cfg.model.n_rounds,
    )
    return student.to(dtype=torch.float32, device=device)


def build_optimizer(student: torch.nn.Module, cfg: ZImageDistillConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(student.parameters(), lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2), weight_decay=cfg.optim.weight_decay)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up over ``warmup_steps`` updates, then a constant learning rate (:func:`distill_to_loop.config.lr_multiplier`)."""

    def lr_lambda(step: int) -> float:
        return lr_multiplier(step, warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_empty_prompt_checked(cfg: ZImageDistillConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The split's empty-prompt encoding (embeddings and validity mask) on ``device``, after checking that it was produced from the teacher
    revision this run loads."""
    train_dir = Path(cfg.data.train_dir)
    manifest = read_empty_prompt_manifest(train_dir)
    validate_empty_prompt(train_dir, cfg.model.teacher_revision, manifest.max_sequence_length)
    embeds, valid = load_empty_prompt(train_dir)
    return embeds.to(device), valid.to(device)


def check_identity(recorded: Any, identity: dict[str, Any], resume_from: str | Path) -> None:
    """Raise ``RuntimeError`` unless a checkpoint's recorded Z-Image identity equals this run's."""
    if recorded != identity:
        raise RuntimeError(f"cannot resume {resume_from}: its Z-Image identity {recorded} differs from this run's {identity} (teacher revision, latent shape)")


def batch_conditioning(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(latents, prompt_embeds, valid)`` of a collated batch on ``device`` (the text cast by :func:`text_dtype_for`)."""
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
def run_training(cfg: ZImageDistillConfig, dist_info: DistInfo, resume_from: str | Path | None = None) -> None:
    """Train (module docstring). Without ``resume_from`` the run directory (``<output_dir>/<name>``) must hold no checkpoint; with it, the run
    continues from that checkpoint (``step_<N>.pt``, with its ``.rng`` directory next to it) and keeps writing into the run directory: the
    checkpoint's own directory, which must hold no checkpoint after step ``N``, or another one that holds no checkpoint and no W&B run id."""
    validate_config(cfg)
    device = dist_info.device
    seed_everything(cfg.seed, dist_info.rank)
    _log(dist_info, f"torch {torch.__version__}, diffusers {diffusers.__version__}")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = latest_checkpoint(run_dir)
    if resume_from is None and existing is not None:
        raise RuntimeError(f"{run_dir} already holds {existing.name}: resume it with --resume-from {existing}, or pick another run name")
    if resume_from is not None:
        if not Path(resume_from).is_file():
            raise FileNotFoundError(f"--resume-from {resume_from}: no such checkpoint file")
        check_resume_target(run_dir, resume_from)

    sched_config = load_scheduler_config(cfg)
    teacher = load_teacher(cfg, device)
    student = build_student(cfg, teacher, device)
    student.loop_grad_checkpoint = cfg.train.loop_grad_checkpoint
    if cfg.train.block_grad_checkpoint:
        student.enable_gradient_checkpointing()  # diffusers' per-block checkpointing (sets the student's checkpointing function as well)

    ddp_model = wrap_for_ddp(student, device_ids=[device.index] if device.type == "cuda" else None) if dist_info.distributed else student
    forward_fn = ddp_model if dist_info.distributed else student.forward_rounds
    optimizer = build_optimizer(student, cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg.optim.warmup_steps)
    empty_prompt_embeds, empty_valid = load_empty_prompt_checked(cfg, device)
    train_dataset = LatentTextDataset(cfg.data.train_dir)
    identity = {"teacher_revision": cfg.model.teacher_revision, "latent_shape": [int(v) for v in train_dataset.index.latent_shape]}
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
        # Re-applied right before the loop: what runs between here and the loop must not shift the restored streams.
        resume_rng_state = get_rng_state()
        start_step, extra = meta["step"], meta["extra"]
        if meta["rng_restored"] is False:
            print(f"{LOG_PREFIX} rank {dist_info.rank}: no RNG file next to {resume_from}; reseeded with seed {cfg.seed} + step {start_step} + rank", flush=True)
        check_resume(extra, meta["format_version"], start_step, recorded, dist_info.world_size, cfg.train.per_gpu_batch_size)
        check_identity(extra.get(CHECKPOINT_IDENTITY_KEY), identity, resume_from)
        resumed_ema_state = {k: v.to(device=device) for k, v in read_ema_state_dict(extra, resume_from).items()}
        if start_step > 0:
            start_epoch, start_batch = int(extra["data_progress"]["epoch"]), int(extra["data_progress"]["batch_in_epoch"])
        _log(dist_info, f"resumed {resume_from} at step {start_step} (epoch {start_epoch}, batch {start_batch})")

    online = cfg.online_rollout
    n_real, n_online, n_trajectories = rollout_rows(cfg)
    _log(
        dist_info,
        f"per_gpu_batch_size={cfg.train.per_gpu_batch_size} = {n_real} data rows + {n_online} online rollout rows ({n_trajectories} {'trajectory' if n_trajectories == 1 else 'trajectories'} x "
        f"{online.steps} Euler steps, wc={online.mix_w}, teacher CFG w={online.teacher_cfg_scale}); global data rows per step "
        f"{n_real * dist_info.world_size}, lr {cfg.optim.lr:g}",
    )
    layout_warning = global_batch_warning(dist_info.world_size, cfg.train.per_gpu_batch_size, online.ratio, online.steps)
    if layout_warning is not None:
        _log(dist_info, layout_warning)
    ema_state = resumed_ema_state if resumed_ema_state is not None else init_ema_state(student)

    batches = resumable_batches(
        train_dataset, batch_size=n_real, world_size=dist_info.world_size, rank=dist_info.rank, seed=cfg.seed, num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda", start_epoch=start_epoch, start_batch=start_batch,
    )
    step_ctx = build_step_context(cfg, forward_fn, student, teacher, sched_config, empty_prompt_embeds, empty_valid, device)
    if resume_rng_state is not None:
        set_rng_state(resume_rng_state)

    rollout_seconds_sum, rollout_end_std_sum, rollout_iterations = 0.0, 0.0, 0  # the online rollouts since the last log line
    t_window, samples_window = time.time(), 0
    global_step = start_step
    progress = EpochProgress(epoch=start_epoch, batch_in_epoch=start_batch)
    while global_step < cfg.train.max_steps:
        batch, progress = next(batches)
        latents, prompt_embeds, valid = batch_conditioning(batch, device)
        loss_out, rollout_stats = training_loss(
            step_ctx, latents, prompt_embeds, valid, rollout_index=online_rollout_index(global_step, dist_info.rank, dist_info.world_size),
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
                CHECKPOINT_IDENTITY_KEY: identity,
            }
            path = save_checkpoint(
                run_dir, global_step, model=student, optimizer=optimizer, scheduler=lr_scheduler, extra=extra, keep_last=cfg.train.keep_last,
                rank=dist_info.rank,
            )
            _log(dist_info, f"step {global_step}: checkpoint {path}")
    logger.finish()
