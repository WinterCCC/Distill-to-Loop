"""The Z-Image training configuration: typed dataclasses on the YAML loading and dotted overrides of :mod:`distill_to_loop.config`.

``load_config(path, overrides)`` reads a YAML mapping, applies ``key.subkey=value`` overrides and builds :class:`ZImageDistillConfig`, with the
type rules of :func:`distill_to_loop.config.from_dict` (unknown keys raise).

The defaults are the released recipe (``zimage/configs/zimage_dooop.yaml``): 3 pre + 6 shared x 4 + 3 post blocks initialized from
Z-Image, AdamW 1.414e-4 after a 500-step warm-up, 96 rows per step on 2 GPUs (48 per GPU: 32 data rows and 16 online rollout rows), 20,000
steps, EMA 0.9995. Each data row draws its sigma uniformly from the 1000 training sigmas of the teacher's scheduler
(:func:`dooop_zimage.step.training_sigma_set`); the online rollout steps on the pipeline's own grid.

CFG conventions. ``online_rollout.teacher_cfg_scale`` w combines ``neg + w (pos - neg)``; ``ZImagePipeline``'s ``guidance_scale`` g combines
``pos + g (pos - neg)``, so ``w = g + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from distill_to_loop.config import (
    PLACEHOLDER,
    DataConfig,
    EmaConfig,
    OnlineRolloutConfig,
    OptimConfig,
    PathsConfig,
    TrainConfig,
    WandbConfig,
    apply_overrides,
    from_dict,
    load_yaml,
    online_rollout_split,
    recipe_layouts,
)
from distill_to_loop.config import validate_config as validate_shared_fields
from dooop_zimage.teacher import TEACHER_REVISION

RECIPE_GLOBAL_BATCH = 96  # rows per optimizer step of the released run (2 GPUs x 48); the learning rate and the other settings are tuned for it
RECIPE_PER_GPU_BATCH = 48  # rows per GPU of the released run; the warning offers no layout with more rows per GPU


@dataclass
class ZImageModelConfig:
    teacher_revision: str = TEACHER_REVISION  # the snapshot of Tongyi-MAI/Z-Image the teacher, VAE and text encoder are read from
    n_pre: int = 3  # student blocks copied from the teacher's first layers
    n_loop: int = 6  # shared blocks, re-entered every round
    n_post: int = 3  # student blocks copied from the teacher's last layers
    n_rounds: int = 4  # training rounds K


@dataclass
class ZImageLossConfig:
    cfg_dropout_p: float = 0.1  # caption dropout of every row (data and rollout) to the empty prompt
    prelude_weight: float = 0.1  # weight of the prelude readout's regression on the teacher velocity


@dataclass
class ZImageDistillConfig:
    name: str = "zimage_dooop"
    seed: int = 0
    output_dir: str = "checkpoints"
    paths: PathsConfig = field(default_factory=PathsConfig)
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(project="dooop-zimage"))
    model: ZImageModelConfig = field(default_factory=ZImageModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(lr=1.414e-4))
    loss: ZImageLossConfig = field(default_factory=ZImageLossConfig)
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(per_gpu_batch_size=48, max_steps=20_000, keep_last=3, loop_grad_checkpoint=False, block_grad_checkpoint=True)
    )
    online_rollout: OnlineRolloutConfig = field(default_factory=lambda: OnlineRolloutConfig(teacher_cfg_scale=5.0))
    ema: EmaConfig = field(default_factory=EmaConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> ZImageDistillConfig:
    """:class:`ZImageDistillConfig` from an optional YAML file plus dotted overrides."""
    data = load_yaml(path) if path is not None else {}
    return from_dict(ZImageDistillConfig, apply_overrides(data, overrides))


def validate_config(cfg: ZImageDistillConfig) -> None:
    """Raise ``ValueError`` on an unset path or an inconsistent value, before any model is built (the checks of
    :func:`distill_to_loop.config.validate_config` on the fields the two configs share, plus the Z-Image ones)."""
    if cfg.data.train_dir == PLACEHOLDER:
        raise ValueError("data.train_dir is not set: pass data.train_dir=<a split directory written by zimage/scripts/prepare_coco.py>")
    if not cfg.model.teacher_revision:
        raise ValueError("model.teacher_revision is empty: the teacher snapshot must be pinned")
    validate_shared_fields(cfg)


def rollout_rows(cfg: ZImageDistillConfig) -> tuple[int, int, int]:
    """``(n_data, n_rollout, n_trajectories)`` of one per-GPU batch (:func:`distill_to_loop.config.online_rollout_split`)."""
    return online_rollout_split(cfg.train.per_gpu_batch_size, cfg.online_rollout.ratio, cfg.online_rollout.steps)


def global_batch_warning(world_size: int, per_gpu_batch_size: int, ratio: float, steps: int) -> str | None:
    """A warning when ``world_size x per_gpu_batch_size`` is not the recipe's :data:`RECIPE_GLOBAL_BATCH` (there is no gradient accumulation),
    else ``None``. The layouts it offers keep :data:`RECIPE_GLOBAL_BATCH` with at most :data:`RECIPE_PER_GPU_BATCH` rows per GPU."""
    total = world_size * per_gpu_batch_size
    if total == RECIPE_GLOBAL_BATCH:
        return None
    layouts = ", ".join(
        f"{gpus}x{rows}" for gpus, rows in recipe_layouts(ratio, steps, global_batch=RECIPE_GLOBAL_BATCH) if rows <= RECIPE_PER_GPU_BATCH
    )
    return (
        f"WARNING: {world_size} rank(s) x train.per_gpu_batch_size={per_gpu_batch_size} = {total} rows per step, but the released recipe (learning "
        f"rate, warm-up, EMA decay) was tuned for {RECIPE_GLOBAL_BATCH} rows per step; layouts (ranks x per-GPU rows) that keep "
        f"{RECIPE_GLOBAL_BATCH}: {layouts}"
    )
