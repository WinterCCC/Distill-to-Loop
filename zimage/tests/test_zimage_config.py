"""The Z-Image config: the YAML file is the schema's defaults, overrides, and the start-up checks."""

from __future__ import annotations

import pytest

import zimage_tiny as tiny
from distill_to_loop.config import PLACEHOLDER, to_dict
from dooop_zimage.config import global_batch_warning, load_config, rollout_rows, validate_config


def test_the_yaml_file_holds_the_schema_defaults():
    assert to_dict(load_config(tiny.CONFIG)) == to_dict(load_config(None))


def test_released_recipe_values():
    cfg = load_config(tiny.CONFIG)
    assert (cfg.model.n_pre, cfg.model.n_loop, cfg.model.n_post, cfg.model.n_rounds) == (3, 6, 3, 4)
    assert cfg.optim.lr == 1.414e-4 and cfg.ema.decay == 0.9995 and cfg.train.max_steps == 20000
    assert (cfg.online_rollout.ratio, cfg.online_rollout.steps, cfg.online_rollout.mix_w, cfg.online_rollout.teacher_cfg_scale) == (0.333333, 8, 2.0, 5.0)
    assert cfg.train.block_grad_checkpoint and not cfg.train.loop_grad_checkpoint
    assert rollout_rows(cfg) == (32, 16, 2)
    assert cfg.data.train_dir == PLACEHOLDER


def test_overrides_are_typed():
    cfg = load_config(tiny.CONFIG, ["optim.lr=2e-4", "train.max_steps=7", "name=yes", "wandb.tags=[a, b]"])
    assert cfg.optim.lr == 2e-4 and cfg.train.max_steps == 7 and cfg.name == "yes" and cfg.wandb.tags == ["a", "b"]
    with pytest.raises(TypeError, match="Unknown config keys"):
        load_config(tiny.CONFIG, ["model.loop_init=stride"])


def test_start_up_checks(tmp_path):
    with pytest.raises(ValueError, match="zimage/scripts/prepare_coco.py"):
        validate_config(load_config(tiny.CONFIG))
    ok = [f"data.train_dir={tmp_path}"]
    validate_config(load_config(tiny.CONFIG, ok))
    with pytest.raises(ValueError, match="multiple of online_rollout.steps=8"):
        validate_config(load_config(tiny.CONFIG, [*ok, "train.per_gpu_batch_size=36"]))
    with pytest.raises(ValueError, match="teacher_revision"):
        validate_config(load_config(tiny.CONFIG, [*ok, "model.teacher_revision="]))
    with pytest.raises(ValueError, match="checkpoint_every"):
        validate_config(load_config(tiny.CONFIG, [*ok, "train.checkpoint_every=0"]))


def test_global_batch_warning():
    assert global_batch_warning(2, 48, 0.333333, 8) is None
    assert global_batch_warning(4, 24, 0.333333, 8) is None
    warning = global_batch_warning(1, 24, 0.333333, 8)
    assert "24 rows per step" in warning and warning.endswith("keep 96: 2x48, 4x24") and "1x96" not in warning  # no more rows per GPU than released
