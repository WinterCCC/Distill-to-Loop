"""Metric logging: Weights & Biases on the main rank when enabled (off by default), a no-op otherwise.

The W&B run id is kept in ``<run_dir>/wandb_run_id``, so a resumed run continues the same W&B run. The training step is logged as the field
``train/step`` and declared the x-axis of every metric. Credentials are read by ``wandb`` itself; this module never touches them.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any, Mapping

from dooop.io_utils import atomic_write_text

STEP_METRIC = "train/step"
RUN_ID_FILE = "wandb_run_id"
_ID_SAFE = re.compile(r"[^A-Za-z0-9_\-]+")


def _with_step(metrics: Mapping[str, Any], step: int | None) -> dict[str, Any]:
    out = dict(metrics)
    if step is not None:
        out[STEP_METRIC] = int(step)
    return out


class NullLogger:
    """The no-op logger (logging disabled, and every rank but the main one)."""

    enabled = False

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger:
    """A thin wrapper over ``wandb.init`` / ``wandb.log`` that resumes the run whose id is stored in ``run_dir``."""

    enabled = True

    def __init__(self, name: str, project: str, entity: str | None, config: Mapping[str, Any], run_dir: str | Path, tags: list[str], notes: str) -> None:
        import wandb  # optional dependency, imported only when logging is enabled

        path = Path(run_dir) / RUN_ID_FILE
        run_id = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not run_id:
            run_id = f"{_ID_SAFE.sub('-', name).strip('-')[:48] or 'run'}-{secrets.token_hex(4)}"
            atomic_write_text(run_id + "\n", path)
        self._run = wandb.init(
            project=project, entity=entity, name=name, id=run_id, resume="allow", config=dict(config), dir=str(run_dir), tags=tags, notes=notes or None,
        )
        self._run.define_metric(STEP_METRIC)
        self._run.define_metric("*", step_metric=STEP_METRIC)

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self._run.log(_with_step(metrics, step))

    def finish(self) -> None:
        self._run.finish()


def create_logger(enabled: bool, is_main: bool, **kwargs: Any) -> WandbLogger | NullLogger:
    """:class:`WandbLogger` on the main rank when ``enabled``, else :class:`NullLogger`."""
    if enabled and is_main:
        return WandbLogger(**kwargs)
    return NullLogger()
