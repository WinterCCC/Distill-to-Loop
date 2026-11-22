#!/usr/bin/env python
"""Prepare the training split from COCO 2017 train: SD3 VAE latents and SD3 prompt encodings.

For every image with a caption (118,287 in ``captions_train2017.json``, sorted by image id; one caption per image picked with ``--caption-seed``)
the image is resized (short side) and center-cropped to ``--image-size``, encoded by the SD3-medium VAE in fp32 (the posterior mean and std are
stored in fp16), and its caption is encoded by the three SD3-medium text encoders in bf16 (``max_sequence_length`` 77; stored in bf16). The
empty prompt's encoding is stored next to them. Output layout: ``distill_to_loop.data.latents``.

Multi-process: launch with ``torchrun``, one rank per GPU; rank ``r`` encodes the shards ``k`` with ``k % world_size == r``. A rerun skips the
shards already committed (``shard_NNNNN.safetensors.done``) and refuses an output directory written with other settings
(``extract_manifest.json``); a finished directory (``index.json`` present) is left as it is.

Disk: the full COCO 2017 train split takes about 182 GB (119 shards of 1000 images).

Example::

    torchrun --nproc_per_node=2 scripts/prepare_coco.py --images-root /data/coco/train2017 \\
        --annotations /data/coco/annotations/captions_train2017.json --out /data/coco2017_512/train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    Sd3PromptEncoder,
    encode_captions,
    has_empty_prompt,
    resolve_snapshot,
    shard_text_name,
    unique_captions,
    validate_empty_prompt,
    write_empty_prompt,
    write_shard_text,
)
from distill_to_loop.hub import explain_missing_model  # noqa: E402
from distill_to_loop.io_utils import atomic_write_text  # noqa: E402
from distill_to_loop.training.distributed import barrier, cleanup, init_distributed  # noqa: E402

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
MANIFEST_FILE = "extract_manifest.json"
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
    parser.add_argument("--out", required=True, help="the output split directory (the trainer's data.train_dir; about 182 GB for the full split)")
    parser.add_argument("--limit", type=int, default=None, help="keep the first N images (smoke tests)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--caption-seed", type=int, default=0, help="seed of the one-caption-per-image pick")
    parser.add_argument("--batch", type=int, default=16, help="VAE encode batch size")
    parser.add_argument("--text-batch", type=int, default=64, help="prompt encode batch size")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers reading the images")
    parser.add_argument("--max-sequence-length", type=int, default=77)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-cache", default=None, help="Hugging Face hub cache (default: HF_HUB_CACHE)")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="'cpu' forces the CPU")
    args = parser.parse_args(argv)

    dist_info = init_distributed(device="cpu" if args.device == "cpu" else None)
    rank, world_size, device = dist_info.rank, dist_info.world_size, dist_info.device

    def log(msg: str) -> None:
        print(f"[prepare rank {rank}/{world_size}] {msg}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = CocoTrainSource(args.images_root, args.annotations, image_size=args.image_size, seed=args.caption_seed)
    if args.limit is not None:
        source.records = source.records[: args.limit]
    total = len(source)
    captions = source.captions()
    unique, caption_idx = unique_captions(captions)
    log(f"{total} images, {len(unique)} unique captions -> {out_dir}")

    from diffusers import AutoencoderKL

    with explain_missing_model(args.model_id, args.hf_cache):
        vae_cfg = AutoencoderKL.load_config(args.model_id, subfolder="vae", cache_dir=args.hf_cache, local_files_only=True)
    snapshot = resolve_snapshot(args.model_id, cache_dir=args.hf_cache)
    latent_shape = [int(vae_cfg["latent_channels"]), args.image_size // 8, args.image_size // 8]
    manifest = {
        "source": source.describe(), "total": total, "image_size": args.image_size, "latent_shape": latent_shape, "shard_size": args.shard_size,
        "model_id": args.model_id, "model_snapshot": snapshot, "max_sequence_length": args.max_sequence_length, "vae_compute_dtype": "float32",
        "text_compute_dtype": "bfloat16", "latent_storage_dtype": "float16", "text_storage_dtype": "bfloat16", "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__, "transformers_version": transformers.__version__, "batch_size": args.batch,
        "text_batch_size": args.text_batch, "workers": args.workers,
    }
    check_manifest(out_dir, manifest, dist_info.is_main)
    barrier()
    if (out_dir / INDEX_FILE).is_file():
        log(f"{out_dir / INDEX_FILE} exists: the split is complete, nothing to do")
        cleanup()
        return 0

    vae = VaeImageEncoder(args.model_id, subfolder="vae", device=device, dtype=torch.float32, cache_dir=args.hf_cache)
    text_encoder = Sd3PromptEncoder(args.model_id, device=device, dtype=torch.bfloat16, max_sequence_length=args.max_sequence_length, cache_dir=args.hf_cache)

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
            validate_empty_prompt(out_dir, snapshot, args.max_sequence_length)
        else:
            empty = encode_captions([""], text_encoder, batch_size=1, dtype=torch.bfloat16)
            write_empty_prompt(out_dir, empty[TEXT_EMB][0], empty[POOLED][0], model_snapshot=snapshot, max_sequence_length=args.max_sequence_length)
        write_index(out_dir, LatentIndex(
            total=total, shard_size=args.shard_size, latent_shape=latent_shape, scaling_factor=vae.scaling_factor, shift_factor=vae.shift_factor,
            shards=shards_meta, vae=f"{args.model_id}#vae", text_encoder=f"{args.model_id}#text_towers(max_seq_len={args.max_sequence_length})",
            source=source.describe(), text_sharded=True,
        ))
        log(f"wrote {out_dir / INDEX_FILE}: {total} samples in {n_shards} shards")
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
