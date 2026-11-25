#!/usr/bin/env python
"""Export the EMA weights of a Z-Image training checkpoint as a compact student directory.

Writes ``<out>/config.json`` (the student architecture) and ``<out>/diffusion_pytorch_model.safetensors`` (fp32 EMA weights; the teacher model
id, its revision and the training step in the file metadata). The architecture is read from the checkpoint's recorded config and the config of
the teacher transformer of the revision the checkpoint records (local Hugging Face cache); the checkpoint is memory-mapped and never modified.

Example::

    python zimage/scripts/export_student.py --checkpoint checkpoints/zimage_dooop/step_00020000.pt --out exported/zimage_dooop_20k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ZIMAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ZIMAGE_ROOT), str(ZIMAGE_ROOT.parent)]

from dooop_zimage.inference import export_student, load_student, read_student_metadata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="a training checkpoint (step_XXXXXXXX.pt) with EMA weights")
    parser.add_argument("--out", required=True, help="the output directory")
    parser.add_argument("--hf-cache", default=None, help="Hugging Face hub cache holding the teacher (default: HF_HUB_CACHE)")
    args = parser.parse_args(argv)
    path = export_student(args.checkpoint, args.out, hf_cache=args.hf_cache)
    student = load_student(args.out)  # strict reload of what was written
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[export] {path} ({path.stat().st_size / 2**30:.2f} GiB); {n_params:,} parameters; metadata {read_student_metadata(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
