"""Resume and memory options of the real trainer on tiny components (CPU, one rank, ``data.num_workers=0``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from distill_to_loop.training.checkpoint import check_resume_target, checkpoint_name
from distill_to_loop.training.logger import RUN_ID_FILE
from tests import tiny

FOREIGN = "holds checkpoints of another run"


@pytest.fixture
def split(tmp_path: Path) -> Path:
    return tiny.write_tiny_split(tmp_path / "data" / "tiny" / "train", n=40)


def _step_file(run_dir: Path, step: int) -> Path:
    """A checkpoint file holding only its step, the one field :func:`check_resume_target` reads."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / checkpoint_name(step)
    torch.save({"step": step}, path)
    return path


def _files(run_dir: Path) -> dict[str, tuple[int, int]]:
    """Every file under ``run_dir`` with its size and modification time."""
    return {str(p.relative_to(run_dir)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(run_dir.rglob("*")) if p.is_file()}


def test_resume_reproduces_the_uninterrupted_run(split, tmp_path, monkeypatch, capsys):
    """One step, a resume from its checkpoint, one more step == two uninterrupted steps (weights, EMA, optimizer, data position)."""
    out = tmp_path / "ckpt"
    straight = tiny.run_training(out, split, 2, monkeypatch, name="straight")
    first = tiny.run_training(out, split, 1, monkeypatch, name="split")
    tiny.run_training(out, split, 2, monkeypatch, name="split", resume_from=Path(first) / "step_00000001.pt")
    tiny.assert_training_states_equal(tiny.read_payload(straight, 2), tiny.read_payload(first, 2))
    # 1 rank x 24 rows is not the recipe's global batch: the trainer says so
    assert "WARNING: 1 rank(s) x train.per_gpu_batch_size=24 = 24 rows per step" in capsys.readouterr().out


def test_resuming_an_older_checkpoint_is_refused_and_a_new_run_name_works(split, tmp_path, monkeypatch):
    out = tmp_path / "ckpt"
    run_dir = tiny.run_training(out, split, 3, monkeypatch, name="run")
    before = _files(run_dir)
    with pytest.raises(RuntimeError, match=r"later checkpoints \(step_00000002\.pt, step_00000003\.pt\).*name="):
        tiny.run_training(out, split, 3, monkeypatch, name="run", resume_from=run_dir / "step_00000001.pt")
    assert _files(run_dir) == before  # nothing overwritten, nothing pruned
    # The remedy the error names: the same checkpoint into a new run directory; the old one is left as it is.
    new_dir = tiny.run_training(out, split, 3, monkeypatch, name="from_step_1", resume_from=run_dir / "step_00000001.pt")
    assert sorted(p.name for p in new_dir.glob("step_*.pt")) == ["step_00000002.pt", "step_00000003.pt"]
    assert _files(run_dir) == before
    tiny.assert_training_states_equal(tiny.read_payload(run_dir, 3), tiny.read_payload(new_dir, 3))


def test_resume_target_rules(tmp_path, monkeypatch):
    """A checkpoint resumes into its own directory when no later step is there, or into a new empty directory; never into an older checkpoint's
    directory with later steps, nor into a directory that holds another run's checkpoints or W&B run id."""
    own = tmp_path / "ckpt" / "own"
    first, latest = _step_file(own, 1), _step_file(own, 2)
    assert check_resume_target(own, latest) == 2  # the latest checkpoint, in its own directory
    with pytest.raises(RuntimeError, match=r"later checkpoints \(step_00000002\.pt\)"):
        check_resume_target(own, first)  # an older one, in its own directory with later steps
    assert check_resume_target(tmp_path / "ckpt" / "new", first) == 1  # a new run directory
    other = tmp_path / "ckpt" / "other"
    _step_file(other, 1)
    with pytest.raises(RuntimeError, match=FOREIGN + r" \(step_00000001\.pt\); pick a new name"):
        check_resume_target(other, latest)
    wandb_only = tmp_path / "ckpt" / "wandb_only"  # a W&B-logged run that stopped before its first checkpoint
    wandb_only.mkdir()
    (wandb_only / RUN_ID_FILE).write_text("other-0123abcd\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=FOREIGN + r" \(wandb_run_id\); pick a new name"):
        check_resume_target(wandb_only, latest)
    # The default output_dir is relative; a checkpoint named by its absolute path is still in its own directory, and the reverse.
    monkeypatch.chdir(tmp_path)
    assert check_resume_target(Path("ckpt") / "own", latest) == 2
    assert check_resume_target(own, Path("ckpt") / "own" / latest.name) == 2
    with pytest.raises(RuntimeError, match=FOREIGN):
        check_resume_target(Path("ckpt") / "other", latest)


def test_resuming_into_the_directory_of_another_run_is_refused(split, tmp_path, monkeypatch):
    """The checkpoint of run ``bar`` resumed with ``name=foo``, whose directory holds foo's checkpoints: refused before any file is written (the
    resumed run would otherwise prune foo's checkpoints and log into foo's W&B run)."""
    out = tmp_path / "ckpt"
    bar = tiny.run_training(out, split, 3, monkeypatch, name="bar")
    foo = tiny.run_training(out, split, 2, monkeypatch, name="foo", overrides=["train.keep_last=2"])
    before = _files(foo)
    with pytest.raises(RuntimeError, match=FOREIGN + r" \(step_00000001\.pt, step_00000002\.pt\); pick a new name"):
        tiny.run_training(out, split, 5, monkeypatch, name="foo", resume_from=bar / "step_00000003.pt", overrides=["train.keep_last=2"])
    assert _files(foo) == before


def test_block_checkpointing_changes_no_number_and_may_change_on_resume(split, tmp_path, monkeypatch):
    calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def counting_checkpoint(*args, **kwargs):
        calls.append(1)
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting_checkpoint)
    out = tmp_path / "ckpt"
    plain = tiny.run_training(out, split, 2, monkeypatch, name="plain")
    calls_plain = len(calls)
    blocks = tiny.run_training(out, split, 2, monkeypatch, name="blocks", overrides=["train.block_grad_checkpoint=true"])
    assert len(calls) - calls_plain > calls_plain > 0  # round checkpointing in both runs, per-block checkpointing on top in the second
    for step in (1, 2):
        tiny.assert_training_states_equal(tiny.read_payload(plain, step), tiny.read_payload(blocks, step))
    # switched on at a resume: no config mismatch, the same numbers
    first = tiny.run_training(out, split, 1, monkeypatch, name="switch")
    tiny.run_training(out, split, 2, monkeypatch, name="switch", resume_from=first / "step_00000001.pt", overrides=["train.block_grad_checkpoint=true"])
    tiny.assert_training_states_equal(tiny.read_payload(plain, 2), tiny.read_payload(first, 2))
