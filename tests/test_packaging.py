"""``pyproject.toml`` declares the dependencies of ``requirements.txt``."""

from __future__ import annotations

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python >= 3.11

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_dependencies_are_the_requirements():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    assert project["dependencies"] == [r for r in requirements if not r.startswith("pytest")]
    assert project["optional-dependencies"]["test"] == [r for r in requirements if r.startswith("pytest")]
    assert project["version"] == "0.1.0" and project["requires-python"] == ">=3.10"
