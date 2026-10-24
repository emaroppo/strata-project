"""A project is the job: what it labels, from which catalog, with what model.

The one file the labeller and an experiment both read, so it lives below
both; it imports contracts, catalog and modelling, never labeller,
experiment or Label Studio. See ``docs/adr/0016``.
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
from .settings import (
    ModellingConfig,
    ModellingConfigError,
    ModellingHosts,
    Settings,
    settings_path,
)

#: Nothing beyond these exports is promised by module path. docs/adr/0015
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
    "ModellingConfigError",
    "ModellingHosts",
    "Project",
    "ProjectError",
    "Settings",
    "list_projects",
    "resolve_under",
    "section",
    "settings_path",
]
