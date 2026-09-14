"""A project on disk, loaded, in a scratch directory."""

from pathlib import Path

import pytest
from strata.project import PROJECT_ENV_VAR, Project


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    """Create a project on disk and return it, loaded.

    Project resolution reads the working directory (a bare name resolves
    under ``projects/``) and $STRATA_PROJECT, so tests run from a scratch
    directory with the variable unset.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(PROJECT_ENV_VAR, raising=False)

    def _make(name: str = "demo", *, under_projects: bool = True, **kwargs) -> Project:
        root = (tmp_path / "projects" / name) if under_projects else (tmp_path / name)
        root.mkdir(parents=True)
        kwargs.setdefault("classes", ["cat", "dog"])
        return Project.create(root, name=name, **kwargs)

    return _make


@pytest.fixture
def project(make_project) -> Project:
    return make_project()


def toml_of(project: Project) -> Path:
    return project.root / "project.toml"
