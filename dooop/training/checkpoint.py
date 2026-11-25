"""Training checkpoints: atomic writes, a ``latest`` pointer, per-rank RNG state, and the resume checks.

Layout inside a run directory::

    step_00001000.pt            {"step", "format_version", "model", "optimizer", "scheduler", "extra"}  (written by rank 0)
    step_00001000.pt.rng/rank_0.pt   RNG state of rank 0 (every rank writes its own file)
    latest                      the file name of the newest checkpoint

``extra`` holds the run's config, the world size, the per-GPU batch, the data position (``data_progress``) and the EMA weights (``ema``). Every
write goes through :func:`dooop.io_utils.atomic_write`, so an interrupted write never leaves a truncated file behind ``latest``. The
save is a two-phase commit across ranks (every rank writes its RNG file, then rank 0 the payload), each phase closed by
:func:`~dooop.training.distributed.all_reduce_all_ok`, so a failure on any rank makes every rank raise together.
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from dooop.io_utils import atomic_write, atomic_write_text
from dooop.seeding import seed_everything
from dooop.training.distributed import all_reduce_all_ok
from dooop.training.logger import RUN_ID_FILE

LATEST_POINTER = "latest"
RNG_SUFFIX = ".rng"
CHECKPOINT_FORMAT_VERSION = 2
EMA_CHECKPOINT_KEY = "ema"
REQUIRED_RESUME_EXTRA_FIELDS: tuple[str, ...] = ("config", "world_size", "per_gpu_batch_size", "data_progress")
# Config fields that may differ between the run that wrote a checkpoint and the run resuming it: run length, reporting cadence, memory-only
# options, logging, and locations (``name`` and ``output_dir`` together give the run directory; :func:`check_resume_target` decides which run
# directories a checkpoint may be resumed into). Every other field must be equal.
RESUME_WHITELIST: frozenset[str] = frozenset(
    {
        "train.max_steps", "train.checkpoint_every", "train.keep_last", "train.log_every", "train.dist_timeout_minutes",
        "train.loop_grad_checkpoint", "train.block_grad_checkpoint", "name", "output_dir", "paths.hf_cache", "data.train_dir",
        "model.teacher_transformer_path", "wandb.enabled", "wandb.project", "wandb.entity", "wandb.tags", "wandb.notes",
    }
)
_STEP_RE = re.compile(r"^step_(\d+)\.pt$")
_TMP_RE = re.compile(r"^\..*\.tmp$")


def atomic_torch_save(obj: Any, path: str | Path) -> Path:
    def _write(fh: BinaryIO) -> None:
        torch.save(obj, fh)

    return atomic_write(path, _write, binary=True)


# --------------------------------------------------------------------------- #
# RNG state (per process; the CUDA state of the current device only)
# --------------------------------------------------------------------------- #
def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def rng_path(ckpt_path: str | Path, rank: int) -> Path:
    """``step_XXXXXXXX.pt.rng/rank_{rank}.pt`` next to the checkpoint file."""
    ckpt_path = Path(ckpt_path)
    return ckpt_path.with_name(ckpt_path.name + RNG_SUFFIX) / f"rank_{int(rank)}.pt"


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / DataParallel / ``torch.compile`` wrappers so that the state_dict keys stay stable."""
    from torch._dynamo.eval_frame import OptimizedModule

    m = model
    while True:
        if isinstance(m, (DistributedDataParallel, DataParallel)):
            m = m.module
        elif isinstance(m, OptimizedModule):
            m = m._orig_mod
        else:
            return m


def checkpoint_name(step: int) -> str:
    return f"step_{int(step):08d}.pt"


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_checkpoint(
    ckpt_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
    keep_last: int | None = None,
    rank: int = 0,
) -> Path:
    """Write the checkpoint of ``step`` (two-phase commit, module docstring) and return the ``.pt`` path. Call on every rank."""
    ckpt_dir = Path(ckpt_dir)
    path = ckpt_dir / checkpoint_name(step)
    device = _model_device(model)
    rng_error: BaseException | None = None
    try:
        atomic_torch_save(get_rng_state(), rng_path(path, rank))
    except BaseException as exc:  # noqa: BLE001 -- reported to the peers, not raised alone
        rng_error = exc
    if not all_reduce_all_ok(rng_error is None, device):
        raise RuntimeError(f"checkpoint save at step {step} failed (rank {rank} RNG write): {rng_error}") from rng_error

    payload_error: BaseException | None = None
    if rank == 0:
        try:
            payload = {
                "step": int(step),
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "extra": dict(extra) if extra is not None else {},
            }
            atomic_torch_save(payload, path)
            atomic_write_text(path.name + "\n", ckpt_dir / LATEST_POINTER)
            if keep_last is not None:
                prune_checkpoints(ckpt_dir, keep_last=keep_last)
        except BaseException as exc:  # noqa: BLE001 -- reported to the peers, not raised alone
            payload_error = exc
    if not all_reduce_all_ok(payload_error is None, device):
        raise RuntimeError(f"checkpoint save at step {step} failed (rank 0 payload write): {payload_error}") from payload_error
    return path


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    """All ``step_*.pt`` files in ascending step order."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    found = []
    for p in ckpt_dir.iterdir():
        m = _STEP_RE.match(p.name)
        if m and p.is_file():
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """The checkpoint the ``latest`` pointer names; else the highest step file; else ``None``."""
    ckpt_dir = Path(ckpt_dir)
    pointer = ckpt_dir / LATEST_POINTER
    if pointer.is_file():
        candidate = ckpt_dir / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_file():
            return candidate
    files = list_checkpoints(ckpt_dir)
    return files[-1] if files else None


def read_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu", *, mmap: bool = False) -> dict[str, Any]:
    """A checkpoint's raw payload (``torch.load``), applied to nothing. ``mmap=True`` memory-maps the file, so only the tensors used are read."""
    return torch.load(Path(path), map_location=map_location, weights_only=False, mmap=mmap)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    rank: int = 0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Load ``path`` into ``model`` (strict), ``optimizer`` and ``scheduler`` and restore this rank's RNG state from its file next to the
    checkpoint; when that file is missing and ``seed`` is given, reseed with ``seed_everything(seed + step, rank)``. Returns ``{"step", "extra",
    "rng_restored", "format_version"}``."""
    path = Path(path)
    payload = read_checkpoint_payload(path, map_location="cpu")
    _unwrap(model).load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    step = int(payload["step"])
    rng_restored: bool | None = None
    rp = rng_path(path, rank)
    if rp.is_file():
        set_rng_state(torch.load(rp, map_location="cpu", weights_only=False))
        rng_restored = True
    elif seed is not None:
        seed_everything(seed + step, rank)
        rng_restored = False
    return {"step": step, "extra": dict(payload.get("extra", {})), "rng_restored": rng_restored, "format_version": int(payload.get("format_version", 1))}


def prune_checkpoints(ckpt_dir: str | Path, keep_last: int) -> list[Path]:
    """Keep the ``keep_last`` checkpoints with the highest steps (with their RNG directories) and remove leftover temp files; returns the removed
    paths. A run never resumes into a directory holding steps above its own or another run's checkpoints (:func:`check_resume_target`), so these
    are the run's newest."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    ckpt_dir = Path(ckpt_dir)
    files = list_checkpoints(ckpt_dir)
    removed = files[:-keep_last]
    for p in removed:
        p.unlink(missing_ok=True)
        shutil.rmtree(p.with_name(p.name + RNG_SUFFIX), ignore_errors=True)
    if ckpt_dir.is_dir():
        for p in ckpt_dir.iterdir():
            if p.is_file() and _TMP_RE.match(p.name):
                p.unlink(missing_ok=True)
    return removed


# --------------------------------------------------------------------------- #
# resume checks
# --------------------------------------------------------------------------- #
def _flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_differences(old_cfg: Mapping[str, Any], new_cfg: Mapping[str, Any], exempt: frozenset[str] = RESUME_WHITELIST) -> dict[str, tuple[Any, Any]]:
    """``{dotted field: (old, new)}`` of every field outside ``exempt`` on which two recorded configs differ."""
    old_flat, new_flat = _flatten(old_cfg), _flatten(new_cfg)
    keys = set(old_flat) | set(new_flat)
    return {k: (old_flat.get(k), new_flat.get(k)) for k in sorted(keys) if k not in exempt and old_flat.get(k) != new_flat.get(k)}


def check_resume_target(run_dir: str | Path, resume_from: str | Path) -> int:
    """Raise ``RuntimeError`` unless ``resume_from`` may be resumed into ``run_dir``: a run directory other than the checkpoint's own must hold no
    checkpoint and no W&B run id (``name`` may change on resume, and the resumed run would prune another run's checkpoints and log into its W&B
    run), and the checkpoint's own directory must hold no checkpoint with a higher step (the resumed run would overwrite those files, and
    pruning, which keeps the highest steps, would delete its own new checkpoints). Returns the step of ``resume_from``, read from its payload
    (memory-mapped). A file-system check only, so every rank reaches the same decision."""
    run_dir = Path(run_dir)
    step = int(read_checkpoint_payload(resume_from, map_location="cpu", mmap=True)["step"])
    if Path(resume_from).resolve().parent != run_dir.resolve():
        found = [p.name for p in list_checkpoints(run_dir)] + ([RUN_ID_FILE] if (run_dir / RUN_ID_FILE).is_file() else [])
        if found:
            raise RuntimeError(
                f"cannot resume {resume_from} into {run_dir}: the directory holds checkpoints of another run ({', '.join(found)}); pick a new name "
                f"(name=...) or another output_dir=... for the resumed run (it would prune those checkpoints and log into that run's W&B run)"
            )
        return step
    later = [p.name for p in list_checkpoints(run_dir) if int(_STEP_RE.match(p.name).group(1)) > step]
    if later:
        raise RuntimeError(
            f"cannot resume {resume_from} (step {step}) into {run_dir}: the directory already holds later checkpoints ({', '.join(later)}), which "
            f"the resumed run would overwrite or prune. Resume into a new run directory instead (a new name=..., or another output_dir=...; the "
            f"checkpoint may stay where it is), or move the checkpoints after step {step} out of {run_dir}"
        )
    return step


def check_resume(extra: Mapping[str, Any], format_version: int, step: int, config: Mapping[str, Any], world_size: int, per_gpu_batch_size: int) -> None:
    """Raise ``RuntimeError`` unless a checkpoint at ``step`` can be resumed by this run: the checkpoint format is current and ``extra`` carries
    every resume field, the recorded config equals this run's ``config`` outside :data:`RESUME_WHITELIST`, and the world size and the per-GPU
    batch are the recorded ones (the data position counts batches of that layout). A step-0 checkpoint is exempt."""
    if step <= 0:
        return
    missing = [f for f in REQUIRED_RESUME_EXTRA_FIELDS if extra.get(f) is None]
    if format_version < CHECKPOINT_FORMAT_VERSION or missing:
        raise RuntimeError(f"cannot resume at step {step}: checkpoint format_version={format_version}, missing extra fields {missing}")
    diffs = config_differences(extra["config"], config)
    if diffs:
        raise RuntimeError(f"resume config mismatch outside the fields that may change {sorted(RESUME_WHITELIST)}: {diffs}")
    recorded = (int(extra["world_size"]), int(extra["per_gpu_batch_size"]))
    if recorded != (world_size, per_gpu_batch_size):
        raise RuntimeError(
            f"cannot resume a checkpoint of world size {recorded[0]} x per-GPU batch {recorded[1]} with world size {world_size} x {per_gpu_batch_size}: "
            "launch the same number of ranks as the run that wrote it"
        )
