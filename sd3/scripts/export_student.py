#!/usr/bin/env python
"""Export the EMA weights of a training checkpoint as a compact student directory.

Writes ``<out>/config.json`` (the student architecture) and ``<out>/diffusion_pytorch_model.safetensors`` (fp32 EMA weights; the teacher model
id and the training step in the file metadata). The architecture is read from the checkpoint's recorded config and the teacher transformer's
config (local Hugging Face cache); the checkpoint is memory-mapped and never modified.

Example::

    python sd3/scripts/export_student.py --checkpoint checkpoints/sd3m_dooop/step_00100000.pt --out exported/sd3m_dooop_100k

``--metadata KEY=VALUE`` (repeatable) adds an entry to the file metadata, e.g. how the weights were averaged after training; the standard
entries cannot be replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root (dooop)

from dooop.inference import STANDARD_METADATA_KEYS, export_student, load_student, read_student_metadata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="a training checkpoint (step_XXXXXXXX.pt) with EMA weights")
    parser.add_argument("--out", required=True, help="the output directory")
    parser.add_argument("--hf-cache", default=None, help="Hugging Face hub cache holding the teacher (default: HF_HUB_CACHE)")
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE", help="an extra entry of the file metadata (repeatable)")
    args = parser.parse_args(argv)
    extra: dict[str, str] = {}
    for item in args.metadata:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--metadata takes KEY=VALUE, got {item!r}")
        if key in extra:
            parser.error(f"--metadata names {key!r} twice")
        if key in STANDARD_METADATA_KEYS:
            parser.error(f"--metadata cannot replace the standard entry {key!r}")
        extra[key] = value
    path = export_student(args.checkpoint, args.out, hf_cache=args.hf_cache, extra_metadata=extra)
    student = load_student(args.out)  # strict reload of what was written
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[export] {path} ({path.stat().st_size / 2**30:.2f} GiB); {n_params:,} parameters; metadata {read_student_metadata(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
