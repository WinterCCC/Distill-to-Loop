"""The Z-Image trainer on tiny components (CPU): checkpoints with the run's identity, a bit-exact resume, memory-only checkpointing options, and
the refused starts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import zimage_tiny as tiny
from dooop.data.text import EMPTY_PROMPT_FILE, load_empty_prompt, write_empty_prompt
from dooop_zimage.teacher import CHECKPOINT_IDENTITY_KEY, TEACHER_REVISION
from tests.tiny import assert_training_states_equal, read_payload

N_SAMPLES = 40


@pytest.fixture
def split(tmp_path: Path) -> Path:
    return tiny.write_tiny_split(tmp_path / "data" / "train", n=N_SAMPLES, text_dtype=torch.bfloat16)


def test_a_run_trains_and_checkpoints(split, tmp_path, monkeypatch, capsys):
    run_dir = tiny.run_training(tmp_path / "ckpt", split, 3, monkeypatch)
    out = capsys.readouterr().out
    assert "16 data rows + 8 online rollout rows (1 trajectory x 8 Euler steps, wc=2.0, teacher CFG w=5.0)" in out
    assert "keep 96: 2x48, 4x24\n" in out and "1x96" not in out and "[step 3] loss=" in out and "rollout_s=" in out
    payload = read_payload(run_dir, 3)
    assert payload["step"] == 3 and payload["extra"]["world_size"] == 1 and payload["extra"]["per_gpu_batch_size"] == tiny.PER_GPU
    assert payload["extra"][CHECKPOINT_IDENTITY_KEY] == {"teacher_revision": TEACHER_REVISION, "latent_shape": [tiny.C, tiny.H, tiny.H]}
    assert payload["extra"]["config"]["model"]["n_loop"] == 6
    ema = payload["extra"]["ema"]["state_dict"]
    assert list(ema) == list(payload["model"]) and any(k.startswith("depth_embedder.") for k in ema)
    assert not all(torch.equal(ema[k], payload["model"][k]) for k in ema)  # the EMA trails the weights
    assert sorted(p.name for p in Path(run_dir).glob("step_*.pt")) == [f"step_{s:08d}.pt" for s in (1, 2, 3)]


def test_a_resume_reproduces_the_uninterrupted_run(split, tmp_path, monkeypatch):
    whole = tiny.run_training(tmp_path / "ckpt", split, 3, monkeypatch, name="whole")
    part = tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch, name="part")
    tiny.run_training(tmp_path / "ckpt", split, 3, monkeypatch, name="part", resume_from=Path(part) / "step_00000002.pt")
    assert_training_states_equal(read_payload(whole, 3), read_payload(part, 3))


def test_activation_checkpointing_options_are_memory_only(split, tmp_path, monkeypatch):
    default = tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch, name="default")  # per-block checkpointing on
    other = tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch, name="other",
                              overrides=["train.block_grad_checkpoint=false", "train.loop_grad_checkpoint=true"])
    for step in (1, 2):
        assert_training_states_equal(read_payload(default, step), read_payload(other, step))


def test_refused_starts(split, tmp_path, monkeypatch):
    run_dir = tiny.run_training(tmp_path / "ckpt", split, 1, monkeypatch, name="run")
    with pytest.raises(RuntimeError, match="already holds step_00000001.pt"):
        tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch, name="run")
    ckpt = Path(run_dir) / "step_00000001.pt"
    with pytest.raises(RuntimeError, match="resume config mismatch"):
        tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch, name="run", resume_from=ckpt, overrides=["optim.lr=2e-4"])
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["extra"][CHECKPOINT_IDENTITY_KEY]["latent_shape"] = [tiny.C, 4, 4]
    torch.save(payload, ckpt)
    with pytest.raises(RuntimeError, match="Z-Image identity"):
        tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch, name="run", resume_from=ckpt)


def test_an_empty_prompt_of_another_teacher_revision_is_refused(split, tmp_path, monkeypatch):
    embeds, valid = load_empty_prompt(split)
    (split / EMPTY_PROMPT_FILE).unlink()
    write_empty_prompt(split, embeds, valid, model_snapshot="0" * 40, max_sequence_length=tiny.TEXT_ROWS)
    with pytest.raises(ValueError, match="model_snapshot"):
        tiny.run_training(tmp_path / "ckpt", split, 1, monkeypatch)


def test_an_unset_split_is_refused_before_any_model_is_loaded(tmp_path, monkeypatch):
    import dooop_zimage.trainer as trainer

    def fail(*args, **kwargs):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(trainer, "load_teacher", fail)
    cfg = tiny.tiny_config(tmp_path / "ckpt", tmp_path, 1, overrides=["data.train_dir=PLACEHOLDER"])
    with pytest.raises(ValueError, match="data.train_dir is not set"):
        trainer.run_training(cfg, tiny.CPU)
