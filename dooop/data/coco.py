"""COCO 2017 train as the image source of the training split, and the VAE encoding of its images.

Every image with at least one caption is included (118,287 for the full ``captions_train2017.json``), sorted by ``image_id``. One caption per
image is picked once with a fixed seed. Preprocessing: short-side antialiased resize to ``image_size``, center crop, no flip, values in
``[-1, 1]``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ImageEncodeFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _group_annotations_by_image(annotations: Sequence[Mapping[str, Any]]) -> dict[int, list[dict]]:
    """``{image_id: [{"annotation_id": int, "caption": str}, ...]}`` in the order ``annotations`` lists them."""
    by_image: dict[int, list[dict]] = {}
    for a in annotations:
        by_image.setdefault(int(a["image_id"]), []).append({"annotation_id": int(a["id"]), "caption": str(a["caption"]).strip()})
    return by_image


def pick_one_annotation_per_image(annotations: Sequence[Mapping[str, Any]], seed: int) -> dict[int, dict]:
    """One random annotation per ``image_id``, deterministic given ``seed`` (``random.Random(seed).choice`` in sorted ``image_id`` order)."""
    by_image = _group_annotations_by_image(annotations)
    rng = random.Random(seed)
    return {image_id: rng.choice(by_image[image_id]) for image_id in sorted(by_image)}


def resize_and_crop_to_tensor(img, image_size: int) -> torch.Tensor:
    """Short-side resize (Lanczos) and center crop of a PIL image to a square; ``float32 [3, size, size]`` in ``[-1, 1]``."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    scale = image_size / min(w, h)
    new_w, new_h = max(image_size, round(w * scale)), max(image_size, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - image_size) // 2, (new_h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    arr = np.asarray(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).div_(127.5).sub_(1.0).contiguous()


def load_image(path: str | Path, image_size: int) -> torch.Tensor:
    """:func:`resize_and_crop_to_tensor` of the image file at ``path``."""
    from PIL import Image

    with Image.open(path) as im:
        return resize_and_crop_to_tensor(im, image_size)


class CocoTrainSource:
    """COCO 2017 train images with one caption per image. ``images_root`` holds the ``train2017/`` JPEGs; ``annotations_path`` is
    ``captions_train2017.json``. Rows are sorted by ``image_id``; a mismatch between the annotated and the listed images raises."""

    name = "coco2017_train"

    def __init__(self, images_root: str | Path, annotations_path: str | Path, image_size: int = 512, seed: int = 0) -> None:
        self.images_root = Path(images_root)
        self.annotations_path = Path(annotations_path)
        self.image_size = int(image_size)
        self.seed = int(seed)
        with open(self.annotations_path, encoding="utf-8") as fh:
            data = json.load(fh)
        picked = pick_one_annotation_per_image(data["annotations"], self.seed)
        images_by_id = {int(im["id"]): im for im in data["images"]}
        caption_ids, image_id_set = set(picked), set(images_by_id)
        if caption_ids != image_id_set:
            raise ValueError(
                f"{self.annotations_path}: the image ids of the annotations and of the images disagree ({len(caption_ids - image_id_set)} captioned ids "
                f"without an image entry, {len(image_id_set - caption_ids)} image entries without a caption)"
            )
        self.records = [
            {
                "image_id": image_id,
                "file_name": images_by_id[image_id]["file_name"],
                "caption": picked[image_id]["caption"],
                "annotation_id": picked[image_id]["annotation_id"],
            }
            for image_id in sorted(image_id_set)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def captions(self) -> list[str]:
        return [r["caption"] for r in self.records]

    def meta(self, index: int) -> dict[str, Any]:
        r = self.records[index]
        return {"index": index, "image_id": r["image_id"], "file_name": r["file_name"], "annotation_id": r["annotation_id"], "prompt": r["caption"]}

    def image(self, index: int) -> torch.Tensor:
        return load_image(self.images_root / self.records[index]["file_name"], self.image_size)

    def describe(self) -> dict[str, Any]:
        """The source's settings as recorded in ``index.json`` (no paths)."""
        return {"name": self.name, "n": len(self.records), "image_size": self.image_size, "seed": self.seed}


class _IndexDataset(Dataset):
    """Lets a ``DataLoader`` load the images of a source in worker processes."""

    def __init__(self, source: CocoTrainSource, indices: list[int]) -> None:
        self.source, self.indices = source, list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        idx = self.indices[i]
        return idx, self.source.image(idx)


class VaeImageEncoder:
    """The posterior mean and std of a ``diffusers.AutoencoderKL`` as an :data:`ImageEncodeFn`. ``scaling_factor`` and ``shift_factor`` are read
    from the VAE config and reported, never applied here."""

    def __init__(
        self,
        model_id: str,
        subfolder: str | None = "vae",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        from diffusers import AutoencoderKL

        kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": local_files_only, "cache_dir": cache_dir}
        if subfolder:
            kwargs["subfolder"] = subfolder
        self.vae = AutoencoderKL.from_pretrained(model_id, **kwargs).to(device).eval()
        self.device, self.dtype = torch.device(device), dtype
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.vae.encode(x.to(self.device, self.dtype)).latent_dist
        return dist.mean, dist.std


def encode_shard(
    source: CocoTrainSource,
    indices: range,
    encode_images: ImageEncodeFn,
    batch_size: int,
    workers: int,
    log: Callable[[str], None],
    sync_fn: Callable[[], None] = lambda: None,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Encode one contiguous range of ``source`` rows; returns ``(mean fp16, std fp16, n, seconds)``. ``sync_fn`` runs after each encode call,
    before the timer stops (e.g. ``torch.cuda.synchronize``)."""
    loader = DataLoader(_IndexDataset(source, list(indices)), batch_size=batch_size, num_workers=workers, shuffle=False)
    means, stds, n, seconds = [], [], 0, 0.0
    for idx, x in loader:
        t0 = time.time()
        mean, std = encode_images(x)
        sync_fn()
        seconds += time.time() - t0
        means.append(mean.to("cpu", torch.float16))
        stds.append(std.to("cpu", torch.float16))
        n += len(idx)
        log(f"  encoded {indices.start + n}/{indices.stop}")
    return torch.cat(means), torch.cat(stds), n, seconds
