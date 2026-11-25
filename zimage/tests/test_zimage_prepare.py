"""``zimage/scripts/prepare_coco.py`` end to end on a tiny COCO-like source (CPU): the tiny VAE of a stand-in teacher snapshot and a stand-in
text pipeline; the split it writes is what the trainer reads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import zimage_tiny as tiny
from distill_to_loop.data.latents import LatentTextDataset, read_index
from distill_to_loop.data.text import read_empty_prompt_manifest
from dooop_zimage.teacher import TEACHER_REVISION
from dooop_zimage.text import ZImagePromptEncoder

ZIMAGE_ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ["a red bus", "two dogs on a sofa near windows", "a plate", "a b c d e f g h i j"]


def _coco(root: Path) -> tuple[Path, Path]:
    images = root / "train2017"
    images.mkdir(parents=True)
    entries, annotations = [], []
    for i, caption in enumerate(CAPTIONS):
        name = f"{i + 1:012d}.jpg"
        Image.new("RGB", (40 + 8 * i, 32), color=(30 * i, 100, 200)).save(images / name)
        entries.append({"id": i + 1, "file_name": name})
        annotations.append({"id": 100 + i, "image_id": i + 1, "caption": caption})
    path = root / "captions_train2017.json"
    path.write_text(json.dumps({"images": entries, "annotations": annotations}), encoding="utf-8")
    return images, path


@pytest.fixture
def prepare(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location("zimage_script_prepare_coco", ZIMAGE_ROOT / "scripts" / "prepare_coco.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script.ZImagePromptEncoder, "from_pretrained",
                        classmethod(lambda cls, model_dir, device, dtype, **kwargs: ZImagePromptEncoder(tiny.FakeTextPipe(), device=device, **kwargs)))
    cache = tiny.write_tiny_hub_cache(tmp_path / "hf_cache")
    images, annotations = _coco(tmp_path / "coco")

    def run(out: Path, *extra: str, other_split: bool = True) -> int:
        return script.main(["--images-root", str(images), "--annotations", str(annotations), "--out", str(out), "--image-size", "32",
                            "--shard-size", "2", "--batch", "2", "--workers", "0", "--stored-text-rows", "8", "--hf-cache", str(cache),
                            "--device", "cpu", *(["--allow-other-split"] if other_split else []), *extra])

    run.script = script
    return run


def test_the_split_is_what_the_trainer_reads(prepare, tmp_path):
    out = tmp_path / "split" / "train"
    assert prepare(out, "--limit", "3") == 0
    index = read_index(out)
    assert index.total == 3 and [s["n"] for s in index.shards] == [2, 1] and index.latent_shape == [tiny.C, 4, 4]
    assert (index.scaling_factor, index.shift_factor) == (0.3611, 0.1159)
    assert read_empty_prompt_manifest(out).model_snapshot == TEACHER_REVISION
    dataset = LatentTextDataset(out)
    item = dataset[1]
    assert item["latent"].shape == (tiny.C, 4, 4) and item["prompt_embeds"].shape == (8, tiny.TEXT_PIPE_WIDTH)
    assert item["prompt_embeds"].dtype == torch.bfloat16 and item["pooled"].tolist() == [1.0] * 7 + [0.0]  # 7 words of 8 rows
    manifest = json.loads((out / "extract_manifest.json").read_text())
    assert manifest["teacher_revision"] == TEACHER_REVISION and manifest["stored_text_rows"] == 8
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert prepare(out, "--limit", "3") == 0  # a finished split is left as it is
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    with pytest.raises(ValueError, match="other settings"):
        prepare(out, "--limit", "3", "--caption-seed", "1")


def test_a_caption_longer_than_the_stored_rows_stops_the_run(prepare, tmp_path):
    with pytest.raises(ValueError, match=r"captions \[1\] have more than 8 tokens"):  # the 4th caption, the 2nd row of shard 1
        prepare(tmp_path / "split" / "train")


def test_an_annotation_file_other_than_coco2017_train_is_refused(prepare, tmp_path, monkeypatch):
    out = tmp_path / "split" / "train"
    with pytest.raises(ValueError, match=r"lists 4 images with a caption, but COCO 2017 train .* has 118,287; pass --allow-other-split"):
        prepare(out, "--limit", "3", other_split=False)  # --limit does not count: the check reads the whole annotation file
    assert not out.exists()  # refused before anything is written
    monkeypatch.setattr(prepare.script, "COCO2017_TRAIN_IMAGES", len(CAPTIONS))  # the expected count passes without the flag
    assert prepare(out, "--limit", "3", other_split=False) == 0 and read_index(out).total == 3
