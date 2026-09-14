"""A project is the job: what it labels, from which catalog, with what model.

The one file the labeller and an experiment both read, so it lives below
both (``docs/adr/0016``). Everything here is about the job; how a tool
presents it is that tool's own, kept in a section this package carries
without reading.

**May import:** ``labels``, ``catalog`` and ``modelling``.
**May not import:** ``strata.labeller``, ``strata.experiment``, or Label
Studio.
"""

from .project import (
    PROJECT_ENV_VAR,
    PROJECT_FILE,
    PROJECTS_DIR,
    CatalogSpec,
    DataSpec,
    LabelSetSpec,
    ModelSpec,
    Project,
    ProjectError,
    list_projects,
    resolve_under,
    section,
)
from .settings import ModellingConfig, Settings

#: What another package may use is what is exported here; no module path
#: is promised (``docs/adr/0015``).
PUBLIC_MODULES: frozenset[str] = frozenset()

__all__ = [
    "PROJECTS_DIR",
    "PROJECT_ENV_VAR",
    "PROJECT_FILE",
    "CatalogSpec",
    "DataSpec",
    "LabelSetSpec",
    "ModelSpec",
    "ModellingConfig",
    "Project",
    "ProjectError",
    "Settings",
    "list_projects",
    "resolve_under",
    "section",
]
