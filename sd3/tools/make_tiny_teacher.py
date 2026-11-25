#!/usr/bin/env python
"""Write a small random SD3 transformer with SD3-medium's input and conditioning shapes, for smoke tests of the training pipeline.

The transformer reads the real prepared split (16 latent channels, 154 x 4096 prompt encodings, 2048-wide pooled encodings) but is only
``--heads x --head-dim`` wide with ``--layers`` layers (``(layers - n_pre - n_post) % n_loop == 0`` for the student's stride initialization;
the default 9 fits 2 + 5 + 2). Point the trainer at it with ``model.teacher_transformer_path=<out>``; the scheduler, VAE and text encoders stay
those of ``model.teacher_model_id``. The weights are random: a student trained from it learns nothing useful.

Example::

    python sd3/tools/make_tiny_teacher.py --out work/tiny_teacher
"""

from __future__ import annotations

import argparse

import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="the output directory (diffusers layout: config.json + safetensors)")
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    inner = args.heads * args.head_dim
    teacher = SD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, out_channels=16, num_layers=args.layers, attention_head_dim=args.head_dim,
        num_attention_heads=args.heads, joint_attention_dim=4096, caption_projection_dim=inner, pooled_projection_dim=2048, pos_embed_max_size=192,
    )
    teacher.to(torch.bfloat16).save_pretrained(args.out)
    print(f"[tiny teacher] {sum(p.numel() for p in teacher.parameters()):,} parameters -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
