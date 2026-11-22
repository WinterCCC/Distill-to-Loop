"""Atomic file writes: a temp file in the target directory, fsync, rename, directory fsync."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TextIO


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: str | Path, write_fn: Callable[[Any], None], binary: bool = True) -> Path:
    """Write via ``write_fn(file_handle)`` into a temp file in ``path.parent``, then rename. On any exception the temp file is removed and the
    exception re-raised. After the rename the directory entry is fsynced and the file mode set to 0644."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def atomic_write_text(text: str, path: str | Path) -> Path:
    def _write(fh: TextIO) -> None:
        fh.write(text)

    return atomic_write(path, _write, binary=False)
