#!/usr/bin/env python
"""Train the looped student by distillation from SD3-medium (see ``distill_to_loop.training.trainer``).

Launch with ``torchrun``, one rank per GPU (``scripts/launch_train.sh`` wraps it). Checkpoints go to ``<output_dir>/<name>/``. A run directory
that already holds a checkpoint is continued only with ``--resume-from``, and only from its newest checkpoint: a run directory holding steps
after the resumed one is refused (resume an older checkpoint into a new run name instead).

Examples::

    # 2 GPUs, the released recipe
    torchrun --nproc_per_node=2 scripts/train.py --config configs/sd3m_dooop.yaml data.train_dir=/path/to/coco2017_512/train

    # continue the run from its newest checkpoint
    torchrun --nproc_per_node=2 scripts/train.py --config configs/sd3m_dooop.yaml data.train_dir=/path/to/coco2017_512/train \\
        --resume-from checkpoints/sd3m_dooop/step_00050000.pt

    # continue from an older checkpoint in a new run directory (checkpoints/sd3m_dooop_from40k/)
    torchrun --nproc_per_node=2 scripts/train.py --config configs/sd3m_dooop.yaml data.train_dir=/path/to/coco2017_512/train \\
        name=sd3m_dooop_from40k --resume-from checkpoints/sd3m_dooop/step_00040000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_to_loop.config import load_config  # noqa: E402
from distill_to_loop.training.distributed import cleanup, init_distributed  # noqa: E402
from distill_to_loop.training.trainer import run_training  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sd3m_dooop.yaml", help="the YAML config (schema: distill_to_loop.config.DistillConfig)")
    parser.add_argument("--resume-from", default=None, metavar="CHECKPOINT",
                        help="continue the run from this checkpoint (step_XXXXXXXX.pt); the run directory must hold no later step")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="'auto': the rank's GPU when CUDA is available; 'cpu': force the CPU")
    parser.add_argument("overrides", nargs="*", help="dotted key.path=value overrides of the config, e.g. train.max_steps=200")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    dist_info = init_distributed(device="cpu" if args.device == "cpu" else None, timeout_minutes=cfg.train.dist_timeout_minutes)
    if dist_info.is_main:
        print(f"[train] run={cfg.name} run_dir={cfg.run_dir} world_size={dist_info.world_size} device={dist_info.device} "
              f"resume_from={args.resume_from}", flush=True)
    try:
        run_training(cfg, dist_info, resume_from=args.resume_from)
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
