"""The config schema, the released recipe and the start-up checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from dooop.config import (
    RECIPE_GLOBAL_BATCH,
    DistillConfig,
    PLACEHOLDER,
    global_batch_warning,
    load_config,
    online_rollout_split,
    recipe_layouts,
    to_dict,
    validate_config,
)

SD3_ROOT = Path(__file__).resolve().parents[1]
RELEASE_CONFIG = SD3_ROOT / "configs" / "sd3m_dooop.yaml"


def test_yaml_equals_the_schema_defaults():
    assert to_dict(load_config(RELEASE_CONFIG)) == to_dict(DistillConfig())


def test_overrides_are_typed():
    cfg = load_config(RELEASE_CONFIG, ["optim.lr=1e-4", "train.max_steps=7", "wandb.enabled=true", "wandb.tags=[a,b]", "paths.hf_cache=null", "name=yes"])
    assert cfg.optim.lr == 1e-4 and cfg.train.max_steps == 7 and cfg.wandb.enabled is True and cfg.wandb.tags == ["a", "b"]
    assert cfg.paths.hf_cache is None and cfg.name == "yes"
    with pytest.raises(TypeError):
        load_config(RELEASE_CONFIG, ["train.max_steps=seven"])
    with pytest.raises(TypeError):
        load_config(RELEASE_CONFIG, ["model.untie_loop=true"])


def test_validation():
    cfg = load_config(RELEASE_CONFIG)
    assert cfg.data.train_dir == PLACEHOLDER
    with pytest.raises(ValueError, match="train_dir"):
        validate_config(cfg)
    cfg.data.train_dir = "/data/train"
    validate_config(cfg)
    assert online_rollout_split(72, 0.333333, 8) == (48, 24, 3)
    assert online_rollout_split(24, 0.333333, 8) == (16, 8, 1)
    assert online_rollout_split(24, 0.0, 8) == (24, 0, 0)
    bad_values = (
        ["train.per_gpu_batch_size=30"], ["ema.decay=1.0"], ["online_rollout.ratio=1.0"], ["loss.cfg_dropout_p=2"],
        ["train.checkpoint_every=0"], ["train.log_every=0"], ["train.keep_last=0"],
    )
    for bad in bad_values:
        cfg = load_config(RELEASE_CONFIG, ["data.train_dir=/data/train", *bad])
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_the_rollout_rows_must_be_whole_trajectories():
    """The rule the config comment states: the rollout rows of a per-GPU batch, round(per_gpu_batch_size * ratio), must be a positive multiple of
    the 8 rollout steps (24, 48 or 72 rows per GPU; 25 also works, the per-GPU batch need not be a multiple of 24)."""
    for rows, split in ((24, (16, 8, 1)), (25, (17, 8, 1)), (48, (32, 16, 2)), (72, (48, 24, 3))):
        assert online_rollout_split(rows, 0.333333, 8) == split
    for rows in (12, 30, 36):
        with pytest.raises(ValueError, match="multiple of online_rollout.steps=8"):
            online_rollout_split(rows, 0.333333, 8)


def test_layouts_of_the_recipe_batch():
    """The layouts that keep 144 rows per step pass the checks without a warning; another global batch warns; a per-GPU batch whose rollout rows
    are not whole trajectories is an error."""
    cfg = load_config(RELEASE_CONFIG, ["data.train_dir=/data/train"])
    ratio, steps = cfg.online_rollout.ratio, cfg.online_rollout.steps
    assert RECIPE_GLOBAL_BATCH == 2 * cfg.train.per_gpu_batch_size
    assert recipe_layouts(ratio, steps) == [(1, 144), (2, 72), (3, 48), (6, 24)]
    for gpus, rows in recipe_layouts(ratio, steps):
        cfg.train.per_gpu_batch_size = rows
        validate_config(cfg)
        assert global_batch_warning(gpus, rows, ratio, steps) is None
    warning = global_batch_warning(4, 24, ratio, steps)
    assert "4 rank(s) x train.per_gpu_batch_size=24 = 96 rows per step" in warning and "1x144, 2x72, 3x48, 6x24" in warning
    cfg.train.per_gpu_batch_size = 36  # 4 GPUs x 36 = 144, but 12 rollout rows are not whole trajectories of 8 steps
    with pytest.raises(ValueError, match="multiple of online_rollout.steps=8"):
        validate_config(cfg)
