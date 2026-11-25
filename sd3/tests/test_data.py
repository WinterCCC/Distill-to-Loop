"""Data preparation pieces and the training dataset, on tiny synthetic inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dooop.data.coco import CocoTrainSource, pick_one_annotation_per_image
from dooop.data.latents import LatentTextDataset, read_index, write_index
from dooop.data.text import load_empty_prompt, unique_captions, validate_empty_prompt
from tests import tiny


def test_dataset_items(tmp_path: Path):
    split = tiny.write_tiny_split(tmp_path / "train", n=20, shard_size=8, text_dtype=torch.bfloat16)
    ds = LatentTextDataset(split)
    assert len(ds) == 20
    item = ds[11]
    assert item["latent"].shape == (tiny.C, tiny.H, tiny.H) and item["latent"].dtype == torch.float32
    assert item["prompt_embeds"].shape == (tiny.SEQ, tiny.D) and item["prompt_embeds"].dtype == torch.bfloat16
    assert item["pooled"].shape == (tiny.D_POOLED,) and item["index"] == 11
    torch.manual_seed(3)
    a = ds[11]["latent"]
    torch.manual_seed(3)
    assert torch.equal(a, ds[11]["latent"])  # the posterior sample comes from the global CPU RNG
    assert not torch.equal(a, ds[11]["latent"])


def test_dataset_refuses_other_layouts(tmp_path: Path):
    split = tiny.write_tiny_split(tmp_path / "train", n=4)
    index = read_index(split)
    index.captions_per_image = 2
    write_index(split, index)
    with pytest.raises(ValueError, match="captions_per_image"):
        LatentTextDataset(split)


def test_empty_prompt_fingerprint(tmp_path: Path):
    split = tiny.write_tiny_split(tmp_path / "train", n=4)
    validate_empty_prompt(split, tiny.SNAPSHOT, tiny.SEQ // 2)
    with pytest.raises(ValueError, match="model_snapshot"):
        validate_empty_prompt(split, "another-snapshot", tiny.SEQ // 2)
    embeds, pooled = load_empty_prompt(split)
    assert embeds.shape == (tiny.SEQ, tiny.D) and pooled.shape == (tiny.D_POOLED,)


def test_unique_captions():
    unique, idx = unique_captions(["a", "b", "a", "c"])
    assert unique == ["a", "b", "c"] and idx.tolist() == [0, 1, 0, 2]


def test_coco_source(tmp_path: Path):
    from PIL import Image

    images = tmp_path / "train2017"
    images.mkdir()
    for i, size in ((9, (64, 48)), (3, (40, 80))):
        Image.fromarray(np.full((size[1], size[0], 3), 10 * i, dtype=np.uint8)).save(images / f"{i:012d}.jpg")
    annotations = {
        "images": [{"id": 9, "file_name": f"{9:012d}.jpg"}, {"id": 3, "file_name": f"{3:012d}.jpg"}],
        "annotations": [{"image_id": 9, "id": 1, "caption": " a dog "}, {"image_id": 9, "id": 2, "caption": "a cat"}, {"image_id": 3, "id": 5, "caption": "a car"}],
    }
    path = tmp_path / "captions.json"
    path.write_text(json.dumps(annotations))
    source = CocoTrainSource(images, path, image_size=32, seed=0)
    assert [r["image_id"] for r in source.records] == [3, 9]
    assert source.captions()[0] == "a car" and source.captions()[1] in ("a dog", "a cat")
    assert pick_one_annotation_per_image(annotations["annotations"], 0) == pick_one_annotation_per_image(annotations["annotations"], 0)
    x = source.image(1)
    assert x.shape == (3, 32, 32) and x.min() >= -1.0 and x.max() <= 1.0
    annotations["images"].append({"id": 4, "file_name": "x.jpg"})
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError):
        CocoTrainSource(images, path)
