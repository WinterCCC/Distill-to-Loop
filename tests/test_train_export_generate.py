"""End to end on tiny components (CPU): train with bf16 text encodings, refuse an unrequested restart, resume, export, load and sample."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKL

import distill_to_loop.training.trainer as trainer
from distill_to_loop.config import load_config
from distill_to_loop.inference import WEIGHTS_NAME, check_recorded_model, export_student, generate_images, load_student, read_student_metadata
from distill_to_loop.training.distributed import DistInfo
from tests import tiny

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU = DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))


@pytest.fixture
def run(tmp_path: Path, monkeypatch):
    """A 2-step run on a bf16 split with a tiny teacher saved to disk (read through model.teacher_transformer_path, as the smoke test does)."""
    teacher_dir = tmp_path / "teacher"
    tiny.tiny_teacher().save_pretrained(teacher_dir)
    monkeypatch.setattr(trainer, "load_scheduler", lambda cfg: tiny.tiny_scheduler())
    monkeypatch.setattr(trainer, "teacher_snapshot", lambda cfg: tiny.SNAPSHOT)
    split = tiny.write_tiny_split(tmp_path / "data" / "train", n=40, text_dtype=torch.bfloat16)

    def config(max_steps: int):
        return load_config(REPO_ROOT / "configs" / "sd3m_d2l.yaml", [
            "name=smoke", f"output_dir={tmp_path / 'ckpt'}", f"data.train_dir={split}", "data.num_workers=0", f"model.teacher_transformer_path={teacher_dir}",
            "train.per_gpu_batch_size=24", f"train.max_steps={max_steps}", "train.checkpoint_every=1", "train.log_every=1", "train.keep_last=2",
        ])

    trainer.run_training(config(2), CPU)
    return config


def test_resume_and_restart_rules(run):
    cfg = run(3)
    with pytest.raises(RuntimeError, match="resume"):
        trainer.run_training(cfg, CPU)
    trainer.run_training(cfg, CPU, resume_from=cfg.run_dir / "step_00000002.pt")
    assert sorted(p.name for p in cfg.run_dir.glob("step_*.pt")) == ["step_00000002.pt", "step_00000003.pt"]  # keep_last=2
    changed = run(4)
    changed.optim.lr = 1e-3
    with pytest.raises(RuntimeError, match="config mismatch"):
        trainer.run_training(changed, CPU, resume_from=cfg.run_dir / "step_00000003.pt")


def test_export_load_and_sample(run, tmp_path: Path, caplog):
    cfg = run(2)
    checkpoint = cfg.run_dir / "step_00000002.pt"
    out = tmp_path / "export"
    export_student(checkpoint, out)
    assert read_student_metadata(out)["training_step"] == "2"
    caplog.clear()
    student = load_student(out)  # strict
    assert all(p.dtype == torch.float32 for p in student.parameters())
    # no "modules ... should be kept in float32: []" from diffusers' ModelMixin.to (the student keeps no module in another dtype)
    assert not [r.getMessage() for r in caplog.records if "kept in float32" in r.getMessage()]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema = payload["extra"]["ema"]["state_dict"]
    assert all(torch.equal(student.state_dict()[k], v) for k, v in ema.items())
    assert (out / WEIGHTS_NAME).is_file() and (out / "config.json").is_file()

    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(8, 8),
        layers_per_block=1, latent_channels=16, norm_num_groups=4, sample_size=16, scaling_factor=1.5305, shift_factor=0.0609,
    ).eval()
    g = torch.Generator().manual_seed(0)
    encodings = {
        "prompt_embeds": torch.randn(3, tiny.SEQ, tiny.D, generator=g).to(torch.bfloat16), "pooled": torch.randn(3, tiny.D_POOLED, generator=g).to(torch.bfloat16),
        "negative_prompt_embeds": torch.randn(tiny.SEQ, tiny.D, generator=g).to(torch.bfloat16), "negative_pooled": torch.randn(tiny.D_POOLED, generator=g).to(torch.bfloat16),
    }
    images = generate_images(student, vae, tiny.tiny_scheduler, encodings, loop_rounds=3, steps=2, cfg_scale=4.0, resolution=16, seed=0,
                             num_samples=2, micro_batch_size=4, device="cpu")
    assert [(p, s) for p, s, _ in images] == [(p, s) for p in range(3) for s in range(2)]
    assert images[0][2].size == (16, 16)
    again = generate_images(student, vae, tiny.tiny_scheduler, encodings, loop_rounds=3, steps=2, cfg_scale=4.0, resolution=16, seed=0,
                            num_samples=2, micro_batch_size=4, device="cpu")
    assert all(a[2].tobytes() == b[2].tobytes() for a, b in zip(images, again))


def test_generate_checks_the_image_size_before_loading(tmp_path: Path, capsys):
    """``--size 520`` stops at argument parsing (exit status 2): no model is read and no output directory is made."""
    module_spec = importlib.util.spec_from_file_location("d2l_generate_script", REPO_ROOT / "scripts" / "generate.py")
    generate = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(generate)
    out = tmp_path / "samples"
    with pytest.raises(SystemExit) as info:
        generate.main(["--student", str(tmp_path / "no_student"), "--prompts", str(tmp_path / "no_prompts.txt"), "--out", str(out), "--size", "520"])
    assert info.value.code == 2
    assert "--size must be a positive multiple of 16" in capsys.readouterr().err
    assert not out.exists()


def test_export_refuses_another_architecture():
    check_recorded_model({"teacher_model_id": "x", "n_pre": 2, "qk_norm": "loop_post", "depth_embed": True})
    with pytest.raises(ValueError, match="qk_norm"):
        check_recorded_model({"qk_norm": "all"})
    with pytest.raises(ValueError, match="untie_loop"):
        check_recorded_model({"untie_loop": True})
