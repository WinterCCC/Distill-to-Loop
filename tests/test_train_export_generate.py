"""End to end on tiny components (CPU): train with bf16 text encodings, refuse an unrequested restart, resume, export, load and sample."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file

import distill_to_loop.training.trainer as trainer
from distill_to_loop.config import load_config
from distill_to_loop.inference import (
    STANDARD_METADATA_KEYS, WEIGHTS_NAME, check_recorded_model, check_sigmas, export_student, generate_images, load_student, read_student_metadata,
)
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


def _tiny_vae() -> AutoencoderKL:
    torch.manual_seed(0)
    return AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(8, 8),
        layers_per_block=1, latent_channels=16, norm_num_groups=4, sample_size=16, scaling_factor=1.5305, shift_factor=0.0609,
    ).eval()


def _tiny_encodings(n: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "prompt_embeds": torch.randn(n, tiny.SEQ, tiny.D, generator=g).to(torch.bfloat16), "pooled": torch.randn(n, tiny.D_POOLED, generator=g).to(torch.bfloat16),
        "negative_prompt_embeds": torch.randn(tiny.SEQ, tiny.D, generator=g).to(torch.bfloat16), "negative_pooled": torch.randn(tiny.D_POOLED, generator=g).to(torch.bfloat16),
    }


def _load_generate_script():
    module_spec = importlib.util.spec_from_file_location("d2l_generate_script", REPO_ROOT / "scripts" / "generate.py")
    generate = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(generate)
    return generate


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

    vae, encodings = _tiny_vae(), _tiny_encodings(3)
    images = generate_images(student, vae, tiny.tiny_scheduler, encodings, loop_rounds=3, steps=2, cfg_scale=4.0, resolution=16, seed=0,
                             num_samples=2, micro_batch_size=4, device="cpu")
    assert [(p, s) for p, s, _ in images] == [(p, s) for p in range(3) for s in range(2)]
    assert images[0][2].size == (16, 16)
    again = generate_images(student, vae, tiny.tiny_scheduler, encodings, loop_rounds=3, steps=2, cfg_scale=4.0, resolution=16, seed=0,
                            num_samples=2, micro_batch_size=4, device="cpu")
    assert all(a[2].tobytes() == b[2].tobytes() for a, b in zip(images, again))


def test_generate_checks_the_image_size_before_loading(tmp_path: Path, capsys):
    """``--size 520`` stops at argument parsing (exit status 2): no model is read and no output directory is made."""
    generate = _load_generate_script()
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


def test_export_extra_metadata(run, tmp_path: Path):
    """``extra_metadata`` adds entries next to the standard ones and leaves the weights alone; without it the metadata holds the standard
    entries only; a standard entry cannot be replaced."""
    checkpoint = run(2).run_dir / "step_00000002.pt"
    plain, extended = tmp_path / "plain", tmp_path / "extended"
    export_student(checkpoint, plain)
    export_student(checkpoint, extended, extra_metadata={"averaging": "uniform average of steps 1 and 2", "distilled_from": "org/model@abc"})
    assert sorted(read_student_metadata(plain)) == sorted(STANDARD_METADATA_KEYS)
    assert read_student_metadata(extended) == {**read_student_metadata(plain), "averaging": "uniform average of steps 1 and 2", "distilled_from": "org/model@abc"}
    a, b = load_file(str(plain / WEIGHTS_NAME)), load_file(str(extended / WEIGHTS_NAME))
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert (plain / "config.json").read_bytes() == (extended / "config.json").read_bytes()
    with pytest.raises(ValueError, match="standard entries"):
        export_student(checkpoint, tmp_path / "x", extra_metadata={"training_step": "7"})
    with pytest.raises(ValueError, match="strings"):
        export_student(checkpoint, tmp_path / "y", extra_metadata={"note": 7})


def test_export_script_metadata_flag(run, tmp_path: Path, capsys):
    module_spec = importlib.util.spec_from_file_location("d2l_export_script", REPO_ROOT / "scripts" / "export_student.py")
    export_script = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(export_script)
    checkpoint = run(2).run_dir / "step_00000002.pt"
    out = tmp_path / "export"
    assert export_script.main(["--checkpoint", str(checkpoint), "--out", str(out), "--metadata", "note=a=b", "--metadata", "other=x"]) == 0
    assert read_student_metadata(out)["note"] == "a=b" and read_student_metadata(out)["other"] == "x"
    for bad, message in ((["--metadata", "novalue"], "KEY=VALUE"), (["--metadata", "weights=x"], "standard entry"),
                         (["--metadata", "k=1", "--metadata", "k=2"], "twice")):
        with pytest.raises(SystemExit) as info:
            export_script.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "never"), *bad])
        assert info.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


def test_generate_images_with_sigmas(run, tmp_path: Path):
    """``sigmas`` reaches the scheduler (shift 3: 1.0, 0.75, 0.5, 0.25 -> 1.0, 0.9, 0.75, 0.5), sets the step count, and is checked."""
    out = tmp_path / "export"
    export_student(run(2).run_dir / "step_00000002.pt", out)
    student, vae, encodings = load_student(out), _tiny_vae(), _tiny_encodings(2)
    schedulers = []

    def scheduler_factory():
        schedulers.append(tiny.tiny_scheduler())
        return schedulers[-1]

    kwargs = dict(loop_rounds=3, cfg_scale=1.0, resolution=16, seed=0, num_samples=1, micro_batch_size=2, device="cpu", output_type="latent")
    with_sigmas = generate_images(student, vae, scheduler_factory, encodings, steps=28, sigmas=[1.0, 0.75, 0.5, 0.25], **kwargs)
    assert len(schedulers) == 1
    assert torch.allclose(schedulers[0].sigmas, torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]))
    assert torch.allclose(schedulers[0].timesteps, torch.tensor([1000.0, 900.0, 750.0, 500.0]))
    grid = generate_images(student, vae, scheduler_factory, encodings, steps=4, **kwargs)
    assert not torch.allclose(schedulers[1].sigmas, schedulers[0].sigmas)
    assert not torch.equal(with_sigmas[0][2], grid[0][2])
    again = generate_images(student, vae, scheduler_factory, encodings, steps=4, sigmas=(1.0, 0.75, 0.5, 0.25), **kwargs)
    assert all(torch.equal(a[2], b[2]) for a, b in zip(with_sigmas, again))
    for bad in ([0.5, 0.75], [1.5, 0.5], [0.5, 0.0], [], [float("nan")]):
        with pytest.raises(ValueError, match="sigmas"):
            generate_images(student, vae, scheduler_factory, encodings, sigmas=bad, **kwargs)
    assert check_sigmas([1, 0.5]) == [1.0, 0.5]


def test_generate_script_sigmas(run, tmp_path: Path, monkeypatch, capsys):
    """``scripts/generate.py --sigmas``: the step count is their number, the record names them (and only then); a conflicting ``--steps`` or
    an invalid list stops at argument parsing."""
    generate = _load_generate_script()
    student = tmp_path / "export"
    export_student(run(2).run_dir / "step_00000002.pt", student)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a\nb\n", encoding="utf-8")
    monkeypatch.setattr(generate, "encode_prompts", lambda texts, *args, **kwargs: _tiny_encodings(len(texts)))
    monkeypatch.setattr(generate, "load_vae_and_scheduler_factory", lambda *args, **kwargs: (_tiny_vae(), tiny.tiny_scheduler))
    common = ["--student", str(student), "--prompts", str(prompts), "--size", "16", "--K", "2", "--micro-batch", "2", "--device", "cpu"]

    assert generate.main([*common, "--out", str(tmp_path / "sigmas"), "--sigmas", "1.0", "0.75", "0.5", "0.25", "--cfg", "1.0"]) == 0
    record = json.loads((tmp_path / "sigmas" / "generation.json").read_text(encoding="utf-8"))
    assert record["sigmas"] == [1.0, 0.75, 0.5, 0.25] and record["steps"] == 4 and record["cfg"] == 1.0
    assert generate.main([*common, "--out", str(tmp_path / "grid"), "--steps", "2"]) == 0
    record = json.loads((tmp_path / "grid" / "generation.json").read_text(encoding="utf-8"))
    assert "sigmas" not in record and record["steps"] == 2
    capsys.readouterr()

    for bad, message in ((["--sigmas", "1.0", "0.75", "--steps", "3"], "conflicts"), (["--sigmas", "0.5", "0.75"], "strictly decreasing"),
                         (["--sigmas", "1.0", "0.0"], "(0, 1]")):
        with pytest.raises(SystemExit) as info:
            generate.main([*common, "--out", str(tmp_path / "never"), *bad])
        assert info.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "never").exists()
