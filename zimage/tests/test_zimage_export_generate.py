"""Export, loading and sampling on tiny components (CPU), through a stand-in teacher snapshot in a Hugging Face cache."""

from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

import dooop_zimage.inference as inference
import zimage_tiny as tiny
from dooop_zimage.inference import (
    STANDARD_METADATA_KEYS, WEIGHTS_NAME, check_recorded_model, export_student, generate_images, load_student, load_vae_and_scheduler_factory,
    read_student_metadata,
)
from dooop_zimage.teacher import TEACHER_MODEL_ID, TEACHER_REVISION

ZIMAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trained(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """``(checkpoint of a 2-step tiny run, Hugging Face cache with the stand-in teacher snapshot)``."""
    split = tiny.write_tiny_split(tmp_path / "data" / "train", n=40)
    run_dir = tiny.run_training(tmp_path / "ckpt", split, 2, monkeypatch)
    cache = tiny.write_tiny_hub_cache(tmp_path / "hf_cache")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return Path(run_dir) / "step_00000002.pt", cache


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"zimage_script_{name}", ZIMAGE_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_is_the_checkpoints_ema(trained, tmp_path):
    checkpoint, cache = trained
    path = export_student(checkpoint, tmp_path / "exported", hf_cache=str(cache))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema = payload["extra"]["ema"]["state_dict"]
    stored = load_file(str(path))
    assert sorted(stored) == sorted(ema)
    assert all(torch.equal(stored[k], ema[k]) for k in ema)
    assert read_student_metadata(tmp_path / "exported") == {
        "teacher_model_id": TEACHER_MODEL_ID, "teacher_revision": TEACHER_REVISION, "training_step": "2", "ema_decay": "0.9995", "weights": "ema",
    }
    assert set(read_student_metadata(tmp_path / "exported")) == set(STANDARD_METADATA_KEYS)
    config = json.loads((tmp_path / "exported" / "config.json").read_text())
    assert config["_class_name"] == "LoopedZImageTransformer" and (config["n_pre"], config["n_loop"], config["n_post"], config["n_rounds"]) == (3, 6, 3, 4)
    student = load_student(tmp_path / "exported")
    assert all(torch.equal(v, ema[k]) for k, v in student.state_dict().items()) and not student.training and student.loop_rounds == 4
    again = export_student(checkpoint, tmp_path / "again", hf_cache=str(cache))
    for name in ("config.json", WEIGHTS_NAME):  # deterministic bytes
        assert (tmp_path / "exported" / name).read_bytes() == (again.parent / name).read_bytes()


def test_export_refusals(trained, tmp_path):
    checkpoint, cache = trained
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for change, message in ((lambda p: p["extra"].pop("zimage"), "Z-Image identity"), (lambda p: p["extra"].pop("ema"), "no EMA")):
        broken = torch.load(checkpoint, map_location="cpu", weights_only=False)
        change(broken)
        torch.save(broken, tmp_path / "broken.pt")
        with pytest.raises(ValueError, match=message):
            export_student(tmp_path / "broken.pt", tmp_path / "out", hf_cache=str(cache))
    check_recorded_model(payload["extra"]["config"]["model"])  # a release record names none of the fixed fields
    with pytest.raises(ValueError, match="loop_init"):
        check_recorded_model({"loop_init": "chunk", "family": "zimage"})
    payload["extra"]["zimage"]["teacher_revision"] = "f" * 40
    torch.save(payload, tmp_path / "other_revision.pt")
    with pytest.raises(FileNotFoundError, match=f"hf download {TEACHER_MODEL_ID} --revision {'f' * 40}"):
        export_student(tmp_path / "other_revision.pt", tmp_path / "out", hf_cache=str(cache))


@pytest.fixture
def exported(trained, tmp_path) -> Path:
    checkpoint, cache = trained
    export_student(checkpoint, tmp_path / "exported", hf_cache=str(cache))
    return tmp_path / "exported"


def test_generate_images(exported):
    student = load_student(exported)
    vae, scheduler_factory = load_vae_and_scheduler_factory("cpu")
    assert vae.dtype == torch.bfloat16
    encodings = tiny.tiny_encodings(3)
    kwargs = dict(loop_rounds=2, steps=3, guidance_scale=8.0, resolution=64, seed=0, num_samples=1, micro_batch_size=2, device="cpu")
    images = generate_images(student, vae, scheduler_factory, encodings, **kwargs)
    assert [(p, s) for p, s, _ in images] == [(0, 0), (1, 0), (2, 0)] and images[0][2].size == (64, 64)
    again = generate_images(student, vae, scheduler_factory, encodings, **kwargs)
    assert all(np.array_equal(np.asarray(a[2]), np.asarray(b[2])) for a, b in zip(images, again))
    latents = generate_images(student, vae, scheduler_factory, encodings, **{**kwargs, "loop_rounds": 4}, output_type="latent")
    assert latents[0][2].shape == (tiny.C, 8, 8) and student.loop_rounds == 4
    with pytest.raises(ValueError, match="multiple of 16"):
        generate_images(student, vae, scheduler_factory, encodings, **{**kwargs, "resolution": 72})


def test_the_whole_pipeline_call_runs_under_the_sampling_autocast(exported, monkeypatch):
    """On CUDA the pipeline call runs under bf16 autocast (the evaluation's scope: the student and the VAE decode). On the CPU the autocast is
    off; a stand-in records that every transformer forward and the VAE decode run inside it."""
    student = load_student(exported)
    vae, scheduler_factory = load_vae_and_scheduler_factory("cpu")
    inside = {"active": False}
    seen: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def stand_in(device):
        inside["active"] = True
        try:
            yield
        finally:
            inside["active"] = False

    real_forward, real_decode = student.forward, vae.decode
    monkeypatch.setattr(student, "forward", lambda *a, **k: (seen.append(("student", inside["active"])), real_forward(*a, **k))[1])
    monkeypatch.setattr(vae, "decode", lambda *a, **k: (seen.append(("vae", inside["active"])), real_decode(*a, **k))[1])
    monkeypatch.setattr(inference, "sampling_autocast", stand_in)
    generate_images(student, vae, scheduler_factory, tiny.tiny_encodings(1), loop_rounds=1, steps=2, resolution=64, micro_batch_size=1, device="cpu")
    assert ("student", True) in seen and ("vae", True) in seen and all(flag for _, flag in seen)


def test_generate_script(exported, tmp_path, monkeypatch):
    script = _load_script("generate")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a red bicycle\n\na bowl of fruit\n", encoding="utf-8")
    recorded = {}

    def fake_encode(texts, device, revision, cache_dir=None):
        recorded["texts"], recorded["revision"] = list(texts), revision
        return tiny.tiny_encodings(len(texts))

    monkeypatch.setattr(script, "encode_prompts", fake_encode)
    out = tmp_path / "samples"
    assert script.main(["--student", str(exported), "--prompts", str(prompts), "--out", str(out), "--K", "2", "--steps", "2", "--size", "64",
                        "--micro-batch", "2", "--device", "cpu"]) == 0
    assert recorded == {"texts": ["a red bicycle", "a bowl of fruit"], "revision": TEACHER_REVISION}
    assert sorted(p.name for p in out.iterdir()) == ["00000_00.png", "00001_00.png", "generation.json", "prompts.txt"]
    record = json.loads((out / "generation.json").read_text())
    assert record["K"] == 2 and record["guidance_scale"] == 8.0 and record["student"]["teacher_revision"] == TEACHER_REVISION
    with pytest.raises(SystemExit) as exit_info:
        script.main(["--student", str(exported), "--prompts", str(prompts), "--out", str(tmp_path / "never"), "--size", "72"])
    assert exit_info.value.code == 2 and not (tmp_path / "never").exists()


def test_the_generate_script_writes_each_image_as_soon_as_it_is_sampled(exported, tmp_path, monkeypatch):
    """Every pipeline call after the first finds the images of the earlier calls on disk; the written PNGs equal ``generate_images``."""
    script = _load_script("generate")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a red bicycle\na bowl of fruit\na lighthouse\n", encoding="utf-8")
    monkeypatch.setattr(script, "encode_prompts", lambda texts, device, revision, cache_dir=None: tiny.tiny_encodings(len(texts)))
    out = tmp_path / "samples"
    on_disk_at_call: list[list[str] | None] = []
    real_build = inference.build_pipeline

    def spy_build(*args, **kwargs):
        pipe = real_build(*args, **kwargs)

        def call(*call_args, **call_kwargs):
            on_disk_at_call.append(sorted(p.name for p in out.glob("*.png")) if out.is_dir() else None)
            return pipe(*call_args, **call_kwargs)

        return call

    monkeypatch.setattr(inference, "build_pipeline", spy_build)
    settings = dict(loop_rounds=2, steps=2, guidance_scale=8.0, resolution=64, seed=0, micro_batch_size=1, device="cpu")
    assert script.main(["--student", str(exported), "--prompts", str(prompts), "--out", str(out), "--K", "2", "--steps", "2", "--size", "64",
                        "--micro-batch", "1", "--device", "cpu"]) == 0
    assert on_disk_at_call == [[], ["00000_00.png"], ["00000_00.png", "00001_00.png"]]
    vae, scheduler_factory = load_vae_and_scheduler_factory("cpu")
    expected = generate_images(load_student(exported), vae, scheduler_factory, tiny.tiny_encodings(3), **settings)
    assert sorted(p.name for p in out.glob("*.png")) == [f"{p:05d}_{s:02d}.png" for p, s, _ in expected]
    for p, s, image in expected:
        assert np.array_equal(np.asarray(Image.open(out / f"{p:05d}_{s:02d}.png")), np.asarray(image))


def test_export_script(trained, tmp_path, capsys):
    checkpoint, _ = trained
    script = _load_script("export_student")
    assert script.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "exported_cli")]) == 0
    out = capsys.readouterr().out
    n_params = sum(p.numel() for p in load_student(tmp_path / "exported_cli").parameters())
    assert f"{n_params:,} parameters" in out and "'teacher_revision': '04cc4abb7c5069926f75c9bfde9ef43d49423021'" in out
