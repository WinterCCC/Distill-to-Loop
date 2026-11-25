#!/usr/bin/env python
"""Generate images with an exported Z-Image student: a prompts file in, one PNG per (prompt, sample) out.

The prompts file holds one prompt per line (empty lines are skipped), or JSON lines with a ``"prompt"`` field (``.jsonl``). Images are written
to ``<out>/<prompt index, 5 digits>_<sample index, 2 digits>.png``, with ``<out>/prompts.txt`` (the prompts in index order) and
``<out>/generation.json`` (the settings). The prompts and the empty negative prompt are encoded once with Z-Image's Qwen3 text encoder;
sampling is ``ZImagePipeline`` on its own sigma grid with the student running ``--K`` loop rounds; image ``(p, s)`` starts from noise seeded by
``(--seed, p, s)``.

``--guidance-scale`` is ``ZImagePipeline``'s guidance_scale g, which combines ``pos + g (pos - neg)``: the usual CFG weight is ``w = g + 1``
(the default g = 8.0 is w = 9). ``--guidance-scale 0`` runs the conditional branch only.

Example::

    python zimage/scripts/generate.py --student exported/zimage_dooop_20k --prompts prompts.txt --out samples/ --K 4 --steps 28 --size 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ZIMAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ZIMAGE_ROOT), str(ZIMAGE_ROOT.parent)]

import torch  # noqa: E402

from dooop_zimage.inference import (  # noqa: E402
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LOOP_ROUNDS,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
    encode_prompts,
    iter_images,
    load_student,
    load_vae_and_scheduler_factory,
    read_student_metadata,
    write_generation_record,
)

# Z-Image: VAE downsampling factor 8 x patch size 2. Checked before any model is loaded; latent_shape checks it again on the loaded models.
SIZE_MULTIPLE = 16


def read_prompts(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".jsonl"):
        return [json.loads(line)["prompt"] for line in text.splitlines() if line.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True, help="an exported student directory (zimage/scripts/export_student.py)")
    parser.add_argument("--prompts", required=True, help="a text file with one prompt per line, or a .jsonl file with a 'prompt' field")
    parser.add_argument("--out", required=True, help="the output directory")
    parser.add_argument("--K", type=int, default=DEFAULT_LOOP_ROUNDS, help="loop rounds of the student (default: the trained count, 4)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="sampling steps on the pipeline's sigma grid")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help="ZImagePipeline's guidance_scale g: pos + g (pos - neg), CFG weight w = g + 1 (0: the conditional branch only)")
    parser.add_argument("--size", type=int, default=DEFAULT_RESOLUTION, help=f"square image size in pixels (a multiple of {SIZE_MULTIPLE})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the per-image noise")
    parser.add_argument("--num-samples", type=int, default=1, help="images per prompt")
    parser.add_argument("--micro-batch", type=int, default=4, help="images per sampling call")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda when available)")
    parser.add_argument("--hf-cache", default=None, help="Hugging Face hub cache holding the teacher (default: HF_HUB_CACHE)")
    args = parser.parse_args(argv)
    if min(args.K, args.steps, args.size, args.num_samples, args.micro_batch) < 1:
        parser.error("--K, --steps, --size, --num-samples and --micro-batch must be >= 1")
    if args.guidance_scale < 0:
        parser.error(f"--guidance-scale must be >= 0, got {args.guidance_scale}")
    if args.size % SIZE_MULTIPLE:
        parser.error(f"--size must be a positive multiple of {SIZE_MULTIPLE} (VAE factor 8 x patch size 2), got {args.size}")

    device = torch.device(args.device)
    metadata = read_student_metadata(args.student)
    revision = metadata["teacher_revision"]
    prompts = read_prompts(args.prompts)
    if not prompts:
        parser.error(f"{args.prompts} holds no prompt")
    out = Path(args.out)
    started = time.time()
    encodings = encode_prompts(prompts, device, revision=revision, cache_dir=args.hf_cache)
    student = load_student(args.student, device=device, loop_rounds=args.K)
    vae, scheduler_factory = load_vae_and_scheduler_factory(device, revision=revision, cache_dir=args.hf_cache)
    print(f"[generate] {len(prompts)} prompts x {args.num_samples} samples, K={args.K}, {args.steps} steps, guidance_scale g={args.guidance_scale} "
          f"(CFG w={args.guidance_scale + 1}), {args.size} px, seed {args.seed}, device {device}", flush=True)
    out.mkdir(parents=True, exist_ok=True)  # made only once the models are loaded
    n_images = 0
    for p, s, image in iter_images(  # each image is written as soon as its micro-batch is sampled
        student, vae, scheduler_factory, encodings, loop_rounds=args.K, steps=args.steps, guidance_scale=args.guidance_scale, resolution=args.size,
        seed=args.seed, num_samples=args.num_samples, micro_batch_size=args.micro_batch, device=device,
    ):
        image.save(out / f"{p:05d}_{s:02d}.png")
        n_images += 1
    (out / "prompts.txt").write_text("".join(f"{p}\n" for p in prompts), encoding="utf-8")
    write_generation_record(out, {
        "student": metadata, "K": args.K, "steps": args.steps, "guidance_scale": args.guidance_scale, "size": args.size, "seed": args.seed,
        "num_samples": args.num_samples, "micro_batch": args.micro_batch, "num_prompts": len(prompts),
    })
    print(f"[generate] wrote {n_images} images to {out} in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
