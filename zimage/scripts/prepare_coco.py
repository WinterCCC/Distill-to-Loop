#!/usr/bin/env python
"""Prepare the Z-Image training split from COCO 2017 train: Z-Image VAE latents and Qwen3 prompt encodings.

For every image with a caption (118,287 in ``captions_train2017.json``, sorted by image id; one caption per image picked with ``--caption-seed``)
the image is resized (short side) and center-cropped to ``--image-size``, encoded by the Z-Image VAE in fp32 (the posterior mean and std are
stored in fp16; the dataset returns ``(z - 0.1159) * 0.3611``), and its caption is encoded by ``ZImagePipeline._encode_prompt`` (the Qwen3 text
encoder in bf16, the chat template, the rows of the real tokens) and stored in bf16 in ``--stored-text-rows`` rows with its validity mask
(``dooop_zimage.text``). A caption of more rows stops the run. The empty prompt's encoding (the caption-dropout target and the negative prompt)
is stored next to them. Output layout: ``distill_to_loop.data.latents``. An annotation file with another image count than COCO 2017 train is
refused unless ``--allow-other-split`` is given (``--limit`` applies after this check).

Multi-process: launch with ``torchrun``, one rank per GPU; rank ``r`` encodes the shards ``k`` with ``k % world_size == r``. A rerun skips the
shards already committed (``shard_NNNNN.safetensors.done``) and refuses an output directory written with other settings
(``extract_manifest.json``); a finished directory (``index.json`` present) is left as it is.

Example::

    torchrun --nproc_per_node=2 zimage/scripts/prepare_coco.py --images-root /data/coco/train2017 \\
        --annotations /data/coco/annotations/captions_train2017.json --out /data/zimage_coco2017_512/train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ZIMAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ZIMAGE_ROOT), str(ZIMAGE_ROOT.parent)]

import diffusers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from distill_to_loop.data.coco import CocoTrainSource, VaeImageEncoder, encode_shard  # noqa: E402
from distill_to_loop.data.latents import (  # noqa: E402
    INDEX_FILE,
    LatentIndex,
    has_shard_marker,
    num_shards,
    shard_name,
    shard_range,
    write_captions,
    write_index,
    write_meta,
    write_shard,
    write_shard_marker,
)
from distill_to_loop.data.text import (  # noqa: E402
    POOLED,
    TEXT_EMB,
    encode_captions,
    has_empty_prompt,
    shard_text_name,
    unique_captions,
    validate_empty_prompt,
    write_empty_prompt,
    write_shard_text,
)
from distill_to_loop.io_utils import atomic_write_text  # noqa: E402
from distill_to_loop.training.distributed import barrier, cleanup, init_distributed  # noqa: E402
from dooop_zimage.teacher import TEACHER_MODEL_ID, TEACHER_REVISION, teacher_dir  # noqa: E402
from dooop_zimage.text import STORED_TEXT_ROWS, TEXT_MAX_SEQUENCE_LENGTH, ZImagePromptEncoder  # noqa: E402

MANIFEST_FILE = "extract_manifest.json"
COCO2017_TRAIN_IMAGES = 118_287  # images with a caption in captions_train2017.json, the released split
NON_BINDING_FIELDS = frozenset({"batch_size", "text_batch_size", "workers"})  # performance only


def check_manifest(out_dir: Path, manifest: dict, is_main: bool) -> None:
    """Write the manifest of a new output directory, or raise ``ValueError`` when an existing one records other binding settings."""
    path = out_dir / MANIFEST_FILE
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        diffs = {k: (recorded.get(k), manifest.get(k)) for k in set(recorded) | set(manifest) if k not in NON_BINDING_FIELDS and recorded.get(k) != manifest.get(k)}
        if diffs:
            raise ValueError(f"{out_dir} was written with other settings (recorded, now): {diffs}; use another --out")
    elif is_main:
        atomic_write_text(json.dumps(manifest, indent=2, sort_keys=True), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-root", required=True, help="the COCO 2017 train2017/ image directory")
    parser.add_argument("--annotations", required=True, help="captions_train2017.json")
    parser.add_argument("--out", required=True, help="the output split directory (the trainer's data.train_dir)")
    parser.add_argument("--limit", type=int, default=None, help="keep the first N images (smoke tests)")
    parser.add_argument("--allow-other-split", action="store_true",
                        help=f"accept an annotation file whose image count is not {COCO2017_TRAIN_IMAGES:,} (COCO 2017 train)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--caption-seed", type=int, default=0, help="seed of the one-caption-per-image pick")
    parser.add_argument("--batch", type=int, default=16, help="VAE encode batch size")
    parser.add_argument("--text-batch", type=int, default=32, help="prompt encode batch size (the pipeline pads every prompt to 512 tokens)")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers reading the images")
    parser.add_argument("--stored-text-rows", type=int, default=STORED_TEXT_ROWS, help="rows stored per caption; a longer caption stops the run")
    parser.add_argument("--teacher-revision", default=TEACHER_REVISION, help=f"the snapshot of {TEACHER_MODEL_ID} to encode with")
    parser.add_argument("--hf-cache", default=None, help="Hugging Face hub cache (default: HF_HUB_CACHE)")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="'cpu' forces the CPU")
    args = parser.parse_args(argv)
    if not 1 <= args.stored_text_rows <= TEXT_MAX_SEQUENCE_LENGTH:
        parser.error(f"--stored-text-rows must be in [1, {TEXT_MAX_SEQUENCE_LENGTH}]")

    dist_info = init_distributed(device="cpu" if args.device == "cpu" else None)
    rank, world_size, device = dist_info.rank, dist_info.world_size, dist_info.device

    def log(msg: str) -> None:
        print(f"[prepare rank {rank}/{world_size}] {msg}", flush=True)

    out_dir = Path(args.out)
    source = CocoTrainSource(args.images_root, args.annotations, image_size=args.image_size, seed=args.caption_seed)
    if len(source) != COCO2017_TRAIN_IMAGES and not args.allow_other_split:
        raise ValueError(
            f"{args.annotations} lists {len(source):,} images with a caption, but COCO 2017 train (captions_train2017.json) has "
            f"{COCO2017_TRAIN_IMAGES:,}; pass --allow-other-split to prepare another split"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit is not None:
        source.records = source.records[: args.limit]
    total = len(source)
    captions = source.captions()
    unique, caption_idx = unique_captions(captions)
    log(f"{total} images, {len(unique)} unique captions -> {out_dir}")

    from diffusers import AutoencoderKL

    model_dir = teacher_dir(args.teacher_revision, args.hf_cache)
    vae_cfg = AutoencoderKL.load_config(str(model_dir), subfolder="vae", local_files_only=True)
    factor = 2 ** (len(vae_cfg["block_out_channels"]) - 1)
    latent_shape = [int(vae_cfg["latent_channels"]), args.image_size // factor, args.image_size // factor]
    manifest = {
        "source": source.describe(), "total": total, "image_size": args.image_size, "latent_shape": latent_shape, "shard_size": args.shard_size,
        "model_id": TEACHER_MODEL_ID, "teacher_revision": args.teacher_revision, "stored_text_rows": args.stored_text_rows,
        "text_max_sequence_length": TEXT_MAX_SEQUENCE_LENGTH, "vae_compute_dtype": "float32", "text_compute_dtype": "bfloat16",
        "latent_storage_dtype": "float16", "text_storage_dtype": "bfloat16", "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__, "transformers_version": transformers.__version__, "batch_size": args.batch,
        "text_batch_size": args.text_batch, "workers": args.workers,
    }
    check_manifest(out_dir, manifest, dist_info.is_main)
    barrier()
    if (out_dir / INDEX_FILE).is_file():
        log(f"{out_dir / INDEX_FILE} exists: the split is complete, nothing to do")
        cleanup()
        return 0

    vae = VaeImageEncoder(str(model_dir), subfolder="vae", device=device, dtype=torch.float32)
    text_encoder = ZImagePromptEncoder.from_pretrained(model_dir, device, dtype=torch.bfloat16, stored_rows=args.stored_text_rows)

    def sync_fn() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    n_shards = num_shards(total, args.shard_size)
    shards_meta: list[dict] = []
    written = skipped = 0
    t0 = time.time()
    for k in range(n_shards):
        rows = shard_range(k, total, args.shard_size)
        shards_meta.append({"file": shard_name(k), "n": len(rows)})
        if k % world_size != rank:
            continue
        if has_shard_marker(out_dir, k):
            skipped += 1
            log(f"shard {k}/{n_shards}: committed, skipped")
            continue
        log(f"shard {k}/{n_shards}: encoding {len(rows)} images")
        mean, std, _, _ = encode_shard(source, rows, vae, args.batch, args.workers, log, sync_fn=sync_fn)
        write_shard(out_dir / shard_name(k), mean, std, torch.as_tensor(caption_idx[rows.start : rows.stop]))
        text = encode_captions(captions[rows.start : rows.stop], text_encoder, batch_size=args.text_batch, dtype=torch.bfloat16)
        write_shard_text(out_dir / shard_text_name(k), text)
        write_shard_marker(out_dir, k)
        written += 1
    log(f"shards written {written}, skipped {skipped}, {time.time() - t0:.1f} s")
    barrier()

    if dist_info.is_main:
        write_captions(out_dir, unique)
        write_meta(out_dir, (source.meta(i) for i in range(total)))
        if has_empty_prompt(out_dir):
            validate_empty_prompt(out_dir, args.teacher_revision, args.stored_text_rows)
        else:
            empty = encode_captions([""], text_encoder, batch_size=1, dtype=torch.bfloat16)
            write_empty_prompt(out_dir, empty[TEXT_EMB][0], empty[POOLED][0], model_snapshot=args.teacher_revision, max_sequence_length=args.stored_text_rows)
        write_index(out_dir, LatentIndex(
            total=total, shard_size=args.shard_size, latent_shape=latent_shape, scaling_factor=vae.scaling_factor, shift_factor=vae.shift_factor,
            shards=shards_meta, vae=f"{TEACHER_MODEL_ID}@{args.teacher_revision}#vae", text_encoder=f"{TEACHER_MODEL_ID}@{args.teacher_revision}#text_encoder",
            source=source.describe(), text_sharded=True,
        ))
        log(f"wrote {out_dir / INDEX_FILE}: {total} samples in {n_shards} shards")
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
