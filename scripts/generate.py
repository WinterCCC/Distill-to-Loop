#!/usr/bin/env python
"""Generate images with an exported student: a prompts file in, one PNG per (prompt, sample) out.

The prompts file holds one prompt per line (empty lines are skipped), or JSON lines with a ``"prompt"`` field (``.jsonl``). Images are written
to ``<out>/<prompt index, 5 digits>_<sample index, 2 digits>.png``, with ``<out>/prompts.txt`` (the prompts in index order) and
``<out>/generation.json`` (the settings). The prompts and the empty negative prompt are encoded once with SD3-medium's three text encoders;
sampling is FlowMatch Euler with CFG, the student running ``--K`` loop rounds; image ``(p, s)`` starts from noise seeded by ``(--seed, p, s)``.

Example::

    python scripts/generate.py --student exported/sd3m_d2l_100k --prompts prompts.txt --out samples/ --K 4 --steps 28 --cfg 7.0 --size 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from distill_to_loop.inference import (  # noqa: E402
    DEFAULT_CFG_SCALE,
    DEFAULT_LOOP_ROUNDS,
    DEFAULT_RESOLUTION,
    DEFAULT_STEPS,
    encode_prompts,
    generate_images,
    load_student,
    load_vae_and_scheduler_factory,
    read_student_metadata,
    write_generation_record,
)

# SD3-medium: VAE downsampling factor 8 x patch size 2. Checked before any model is loaded; latent_shape checks it again on the loaded models.
SIZE_MULTIPLE = 16


def read_prompts(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".jsonl"):
        return [json.loads(line)["prompt"] for line in text.splitlines() if line.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True, help="an exported student directory (scripts/export_student.py)")
    parser.add_argument("--prompts", required=True, help="a text file with one prompt per line, or a .jsonl file with a 'prompt' field")
    parser.add_argument("--out", required=True, help="the output directory")
    parser.add_argument("--K", type=int, default=DEFAULT_LOOP_ROUNDS, help="loop rounds of the student (default: the trained count, 4)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Euler steps")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG_SCALE, help="CFG scale (at most 1: the conditional branch only)")
    parser.add_argument("--size", type=int, default=DEFAULT_RESOLUTION, help=f"square image size in pixels (a multiple of {SIZE_MULTIPLE})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the per-image noise")
    parser.add_argument("--num-samples", type=int, default=1, help="images per prompt")
    parser.add_argument("--micro-batch", type=int, default=4, help="images per sampling call")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda when available)")
    parser.add_argument("--model-id", default=None, help="the teacher model providing the VAE, scheduler and text encoders (default: the one the export records)")
    parser.add_argument("--hf-cache", default=None, help="Hugging Face hub cache (default: HF_HUB_CACHE)")
    args = parser.parse_args(argv)
    if min(args.K, args.steps, args.size, args.num_samples, args.micro_batch) < 1:
        parser.error("--K, --steps, --size, --num-samples and --micro-batch must be >= 1")
    if args.size % SIZE_MULTIPLE:
        parser.error(f"--size must be a positive multiple of {SIZE_MULTIPLE} (VAE factor 8 x patch size 2), got {args.size}")

    device = torch.device(args.device)
    model_id = args.model_id or read_student_metadata(args.student)["teacher_model_id"]
    prompts = read_prompts(args.prompts)
    if not prompts:
        parser.error(f"{args.prompts} holds no prompt")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    encodings = encode_prompts(prompts, model_id, device, cache_dir=args.hf_cache)
    student = load_student(args.student, device=device, loop_rounds=args.K)
    vae, scheduler_factory = load_vae_and_scheduler_factory(model_id, device, cache_dir=args.hf_cache)
    print(f"[generate] {len(prompts)} prompts x {args.num_samples} samples, K={args.K}, {args.steps} steps, CFG {args.cfg}, {args.size} px, "
          f"seed {args.seed}, device {device}", flush=True)
    results = generate_images(
        student, vae, scheduler_factory, encodings, loop_rounds=args.K, steps=args.steps, cfg_scale=args.cfg, resolution=args.size, seed=args.seed,
        num_samples=args.num_samples, micro_batch_size=args.micro_batch, device=device,
    )
    for p, s, image in results:
        image.save(out / f"{p:05d}_{s:02d}.png")
    (out / "prompts.txt").write_text("".join(f"{p}\n" for p in prompts), encoding="utf-8")
    write_generation_record(out, {
        "student": read_student_metadata(args.student), "model_id": model_id, "K": args.K, "steps": args.steps, "cfg": args.cfg, "size": args.size,
        "seed": args.seed, "num_samples": args.num_samples, "micro_batch": args.micro_batch, "num_prompts": len(prompts),
    })
    print(f"[generate] wrote {len(results)} images to {out} in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
