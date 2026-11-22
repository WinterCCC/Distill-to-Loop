"""The training configuration: typed dataclasses, YAML loading and dotted command-line overrides.

``load_config(path, overrides)`` reads a YAML mapping, applies ``key.subkey=value`` overrides and builds :class:`DistillConfig`. Unknown keys
raise. Scalars are checked against the annotated type (``bool`` is not an ``int``; an ``int`` may be promoted to ``float``); override values
arrive as strings and are converted by the target type, so ``name=yes`` stays the string ``"yes"`` while ``optim.lr=1e-4`` becomes ``0.0001``.
Nested sections may also be given as a YAML mapping on the command line (``wandb={enabled: true}``); list fields take YAML flow lists
(``wandb.tags=[a, b]``).

The defaults are the released recipe (``configs/sd3m_d2l.yaml``): 2 pre + 5 shared x 4 + 2 post blocks initialized from SD3-medium, AdamW
1.732e-4 after a 500-step warm-up, 144 rows per step on 2 GPUs (72 per GPU: 48 data rows and 24 online rollout rows), 100,000 steps, EMA 0.9995.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Type, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")

PLACEHOLDER = "PLACEHOLDER"  # a value the user must set (a path), refused at start-up

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}
_NULL = {"null", "none", "~", ""}
# YAML 1.1 (PyYAML) reads ``1e-4`` as a string; only ``1.0e-4`` is a float.
_SCI_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


# --------------------------------------------------------------------------- #
# the schema
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    hf_cache: str | None = None  # Hugging Face hub cache directory of the teacher (None: the HF_HUB_CACHE environment variable or the default)


@dataclass
class WandbConfig:
    enabled: bool = False  # Weights & Biases logging is optional and off by default
    project: str = "distill-to-loop"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ModelConfig:
    teacher_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    # Optional local diffusers transformer directory whose weights replace the teacher transformer of teacher_model_id (scheduler, VAE and
    # text encoders stay those of teacher_model_id); empty: the model's own transformer.
    teacher_transformer_path: str = ""
    n_pre: int = 2  # student blocks copied from the teacher's first layers
    n_loop: int = 5  # shared blocks, re-entered every round
    n_post: int = 2  # student blocks copied from the teacher's last layers
    n_rounds: int = 4  # training rounds K


@dataclass
class DataConfig:
    train_dir: str = PLACEHOLDER  # a split directory written by scripts/prepare_coco.py
    # DataLoader workers. A resume reproduces the uninterrupted run exactly only with 0: with workers the sample order is the same, but after a
    # resume within an epoch the latents drawn from the stored VAE posteriors differ from those of the uninterrupted run.
    num_workers: int = 8


@dataclass
class OptimConfig:
    lr: float = 1.732e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 500  # linear warm-up, then a constant learning rate
    grad_clip: float = 1.0


@dataclass
class LossConfig:
    logit_mean: float = 0.0  # logit-normal timestep sampling of the data rows
    logit_std: float = 1.0
    cfg_dropout_p: float = 0.1  # caption dropout of every row (data and rollout)
    prelude_weight: float = 0.1  # weight of the prelude readout's regression on the teacher velocity


@dataclass
class TrainConfig:
    per_gpu_batch_size: int = 72  # rows of one step on one rank, data rows plus online rollout rows
    max_steps: int = 100_000
    checkpoint_every: int = 1000
    keep_last: int = 10
    log_every: int = 10
    dist_timeout_minutes: int = 120
    loop_grad_checkpoint: bool = True  # round-level activation checkpointing of the shared blocks (memory only, same numbers)
    # Per-block activation checkpointing of every transformer block (memory only, same numbers; may change on resume). Off in the released run.
    block_grad_checkpoint: bool = False


@dataclass
class OnlineRolloutConfig:
    ratio: float = 0.333333  # share of the per-GPU rows that are online rollout states (0: none)
    steps: int = 8  # Euler steps of each rollout trajectory (one row per step)
    mix_w: float = 2.0  # the mixing weight wc of the rollout velocity CFG_w(v_T) + (wc - 1)(v_T,c - v_S,c)
    teacher_cfg_scale: float = 4.5  # the teacher CFG scale w of the rollout velocity


@dataclass
class EmaConfig:
    decay: float = 0.9995


@dataclass
class DistillConfig:
    name: str = "sd3m_d2l"
    seed: int = 0
    output_dir: str = "checkpoints"
    paths: PathsConfig = field(default_factory=PathsConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    online_rollout: OnlineRolloutConfig = field(default_factory=OnlineRolloutConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


# --------------------------------------------------------------------------- #
# derived values and checks
# --------------------------------------------------------------------------- #
def online_rollout_split(per_gpu_batch_size: int, ratio: float, steps: int) -> tuple[int, int, int]:
    """``(n_data, n_rollout, n_trajectories)`` of one per-GPU batch: ``n_rollout = round(per_gpu_batch_size * ratio)`` rollout rows from
    ``n_rollout / steps`` trajectories, the other ``n_data`` rows from the dataset; ``ratio == 0`` gives ``(per_gpu_batch_size, 0, 0)``. Raises
    ``ValueError`` unless ``n_rollout`` is a positive multiple of ``steps`` that leaves at least one data row and the trajectories fit in the data
    rows whose prompts they take."""
    if ratio == 0.0:
        return per_gpu_batch_size, 0, 0
    n_rollout = round(per_gpu_batch_size * ratio)
    n_data = per_gpu_batch_size - n_rollout
    if steps < 1 or n_rollout < steps or n_rollout % steps != 0 or n_data < 1:
        raise ValueError(
            f"online_rollout.ratio={ratio} of train.per_gpu_batch_size={per_gpu_batch_size} gives {n_rollout} rollout rows: need a positive "
            f"multiple of online_rollout.steps={steps} that leaves at least one data row"
        )
    n_trajectories = n_rollout // steps
    if n_trajectories > n_data:
        raise ValueError(f"{n_trajectories} rollout trajectories take the prompts of the first data rows, but there are only {n_data} data rows")
    return n_data, n_rollout, n_trajectories


RECIPE_GLOBAL_BATCH = 144  # rows per optimizer step of the released run (2 GPUs x 72); the learning rate and the other settings are tuned for it


def recipe_layouts(ratio: float, steps: int, global_batch: int = RECIPE_GLOBAL_BATCH) -> list[tuple[int, int]]:
    """Every ``(GPUs, per_gpu_batch_size)`` with ``GPUs x per_gpu_batch_size == global_batch`` whose per-GPU rows split into whole online rollout
    trajectories (:func:`online_rollout_split`); for the released recipe ``[(1, 144), (2, 72), (3, 48), (6, 24)]``."""
    layouts = []
    for gpus in range(1, global_batch + 1):
        if global_batch % gpus:
            continue
        try:
            online_rollout_split(global_batch // gpus, ratio, steps)
        except ValueError:
            continue
        layouts.append((gpus, global_batch // gpus))
    return layouts


def global_batch_warning(world_size: int, per_gpu_batch_size: int, ratio: float, steps: int) -> str | None:
    """A warning when ``world_size x per_gpu_batch_size`` is not the recipe's :data:`RECIPE_GLOBAL_BATCH` (there is no gradient accumulation),
    else ``None``."""
    total = world_size * per_gpu_batch_size
    if total == RECIPE_GLOBAL_BATCH:
        return None
    layouts = ", ".join(f"{gpus}x{rows}" for gpus, rows in recipe_layouts(ratio, steps))
    return (
        f"WARNING: {world_size} rank(s) x train.per_gpu_batch_size={per_gpu_batch_size} = {total} rows per step, but the released recipe (learning "
        f"rate, warm-up, EMA decay) was tuned for {RECIPE_GLOBAL_BATCH} rows per step; layouts (ranks x per-GPU rows) that keep "
        f"{RECIPE_GLOBAL_BATCH}: {layouts}"
    )


def validate_config(cfg: DistillConfig) -> None:
    """Raise ``ValueError`` on an unset path or an inconsistent value, before any model is built."""
    if cfg.data.train_dir == PLACEHOLDER:
        raise ValueError("data.train_dir is not set: pass data.train_dir=<a split directory written by scripts/prepare_coco.py>")
    for key in ("checkpoint_every", "log_every", "keep_last"):
        if getattr(cfg.train, key) < 1:
            raise ValueError(f"train.{key} must be >= 1, got {getattr(cfg.train, key)}")
    model, rollout = cfg.model, cfg.online_rollout
    if model.n_pre < 0 or model.n_loop < 1 or model.n_post < 1 or model.n_rounds < 1:
        raise ValueError(f"need n_pre >= 0, n_loop >= 1, n_post >= 1 and n_rounds >= 1, got {model}")
    weight = cfg.loss.prelude_weight
    if not (math.isfinite(weight) and weight >= 0.0):
        raise ValueError(f"loss.prelude_weight must be a finite number >= 0, got {weight!r}")
    if not (0.0 <= cfg.loss.cfg_dropout_p <= 1.0):
        raise ValueError(f"loss.cfg_dropout_p must be in [0, 1], got {cfg.loss.cfg_dropout_p}")
    if not (0.0 <= rollout.ratio < 1.0):
        raise ValueError(f"online_rollout.ratio must be in [0, 1), got {rollout.ratio}")
    if not (math.isfinite(rollout.mix_w) and rollout.mix_w >= 0.0):
        raise ValueError(f"online_rollout.mix_w must be a finite number >= 0, got {rollout.mix_w}")
    if not (math.isfinite(rollout.teacher_cfg_scale) and rollout.teacher_cfg_scale > 0.0):
        raise ValueError(f"online_rollout.teacher_cfg_scale must be a finite number > 0, got {rollout.teacher_cfg_scale}")
    online_rollout_split(cfg.train.per_gpu_batch_size, rollout.ratio, rollout.steps)
    if not (0.0 <= cfg.ema.decay < 1.0):
        raise ValueError(f"ema.decay must be in [0, 1), got {cfg.ema.decay}")
    if cfg.optim.warmup_steps < 0:
        raise ValueError(f"optim.warmup_steps must be >= 0, got {cfg.optim.warmup_steps}")


def lr_multiplier(step: int, warmup_steps: int) -> float:
    """``lr(step) / optim.lr``: linear warm-up over ``warmup_steps`` optimizer updates, then 1. ``step`` is the number of updates already made
    (``LambdaLR``'s convention), so the value is the multiplier of the next update."""
    if warmup_steps > 0 and step < warmup_steps:
        return min(1.0, (step + 1) / warmup_steps)
    return 1.0


# --------------------------------------------------------------------------- #
# dataclass <-> dict
# --------------------------------------------------------------------------- #
class RawOverride(str):
    """Marker for values coming from the command line: converted by the target type."""


def _split_optional(tp: Any) -> tuple[Any, bool]:
    """``Optional[X]`` -> ``(X, True)``; anything else -> ``(tp, False)``."""
    if get_origin(tp) is not None and type(None) in get_args(tp):
        inner = [a for a in get_args(tp) if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return tp, False


def _from_raw(raw: str, target: Any, name: str) -> Any:
    """Convert a raw override string to ``target``."""
    if target is str:
        return str(raw)
    if target is bool:
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise TypeError(f"{name}: cannot convert {raw!r} to bool")
    if target is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise TypeError(f"{name}: cannot convert {raw!r} to int") from exc
    if target is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise TypeError(f"{name}: cannot convert {raw!r} to float") from exc
    return yaml.safe_load(raw)  # lists, dicts, ...


def _coerce_scalar(value: Any, target: Any, name: str) -> Any:
    """Validate or convert ``value`` for a field annotated with ``target``."""
    target, optional = _split_optional(target)
    if isinstance(value, RawOverride):
        if optional and value.strip().lower() in _NULL:
            return None
        value = _from_raw(str(value), target, name)
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name}: expected {getattr(target, '__name__', target)}, got None")
    if target is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name}: expected bool, got {type(value).__name__} {value!r}")
    elif target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name}: expected int, got {type(value).__name__} {value!r}")
    elif target is float:
        if isinstance(value, str) and _SCI_FLOAT_RE.match(value.strip()):
            value = float(value)  # YAML 1.1 artifact from a config file
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}: expected float, got {type(value).__name__} {value!r}")
        value = float(value)
    elif target is str:
        if not isinstance(value, str):
            raise TypeError(f"{name}: expected str, got {type(value).__name__} {value!r}")
    return value


def from_dict(cls: Type[T], data: Mapping[str, Any] | None, _prefix: str = "") -> T:
    """Instantiate dataclass ``cls`` from a mapping, recursing into nested dataclasses; ``TypeError`` on unknown keys or mismatched types."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    data = dict(data or {})
    hints = get_type_hints(cls)
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise TypeError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        name = f"{_prefix}{f.name}"
        target, _ = _split_optional(hints[f.name])
        if is_dataclass(target):
            if isinstance(value, RawOverride):
                value = yaml.safe_load(str(value))
            if not isinstance(value, Mapping):
                raise TypeError(f"{name}: expected a mapping for {target.__name__}, got {type(value).__name__} {value!r}")
            value = from_dict(target, value, _prefix=name + ".")
        elif get_origin(target) is list or target is list:
            if isinstance(value, RawOverride):
                value = yaml.safe_load(str(value))
            if not isinstance(value, list):
                raise TypeError(f"{name}: expected a list, got {type(value).__name__} {value!r}")
        else:
            value = _coerce_scalar(value, hints[f.name], name)
        kwargs[f.name] = value
    return cls(**kwargs)


def to_dict(cfg: Any) -> dict[str, Any]:
    """A (nested) dataclass instance as a plain dict (what a checkpoint records)."""
    if not is_dataclass(cfg):
        raise TypeError(f"{cfg!r} is not a dataclass instance")
    return dataclasses.asdict(cfg)


# --------------------------------------------------------------------------- #
# YAML and dotted overrides
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file holding a mapping (or nothing)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{path}: the top-level YAML value must be a mapping, got {type(data).__name__}")
    return dict(data)


def parse_override(item: str) -> tuple[list[str], RawOverride]:
    """``"a.b.c=value"`` -> ``(["a", "b", "c"], RawOverride("value"))``; the value is converted later, by the target type."""
    if "=" not in item:
        raise ValueError(f"Override must look like key.path=value, got {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override has an empty key: {item!r}")
    return key.split("."), RawOverride(raw)


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """A deep copy of ``data`` with the dotted overrides applied (intermediate mappings created as needed)."""
    out = copy.deepcopy(dict(data))
    for item in overrides:
        keys, value = parse_override(item)
        node = out
        for k in keys[:-1]:
            child = node.get(k)
            if child is None:
                child = node[k] = {}
            elif not isinstance(child, dict):
                raise TypeError(f"Cannot override {'.'.join(keys)}: {k!r} is not a mapping")
            node = child
        node[keys[-1]] = value
    return out


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> DistillConfig:
    """:class:`DistillConfig` from an optional YAML file plus dotted overrides."""
    data = load_yaml(path) if path is not None else {}
    return from_dict(DistillConfig, apply_overrides(data, overrides))
