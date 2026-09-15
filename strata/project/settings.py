"""Machine-level settings: the catalogs this host reaches, and where it trains.

Everything that belongs to a job lives in the project directory instead.
This file describes the machine; a project carries none of it
(``docs/adr/0016``).

What a catalog is, and which one this host uses, is read through
:mod:`strata.catalog.config`, which the blob server and the modelling host
read too. A tool with settings of its own extends :class:`Settings` and
reads its own section from the same file.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from strata.catalog.config import CONFIG_ENV, CatalogConfigError, Catalogs, read_catalogs

SETTINGS_FILE = "config.toml"


def settings_path(given: Path | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """The host file a command reads: what it was given, else what ``$STRATA_CONFIG``
    names, else ``./config.toml``.

    The variable is the one a service reads, so a host states its file
    once and a project directory needs no copy. A variable naming a file
    that does not exist is refused rather than defaulted, as it is for a
    service. See ``docs/adr/0019``.
    """
    if given is not None:
        return given
    environ = os.environ if environ is None else environ
    value = environ.get(CONFIG_ENV, "")
    if not value:
        return Path(SETTINGS_FILE)
    path = Path(value)
    if not path.exists():
        raise CatalogConfigError(f"${CONFIG_ENV} names {path}, which does not exist.")
    return path


@dataclass
class ModellingConfig:
    """Where training happens.

    Empty means in this process. A URL sends rounds to a host with the
    GPU, which materialises the dataset itself. See ``docs/adr/0007``.
    """

    url: str = ""
    #: Shared with the host, and read from the environment. docs/adr/0019
    token: str = ""


@dataclass
class Settings:
    #: The catalogs this host describes, and which one it uses by default.
    catalogs: Catalogs = field(default_factory=Catalogs)
    modelling: ModellingConfig = field(default_factory=ModellingConfig)

    @classmethod
    def load(cls, path: Path | None = None, environ: Mapping[str, str] | None = None):
        """The settings in ``path``, or in the file :func:`settings_path` resolves."""
        path = settings_path(path, environ)
        data: dict = {}
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
        settings = cls()
        settings._read(data, os.environ if environ is None else environ)
        return settings

    def _read(self, data: dict, environ: Mapping[str, str]) -> None:
        """Take what this class owns from the file; a subclass adds its own."""
        self.catalogs = read_catalogs(data.get("catalog", {}), environ)
        if "modelling" in data:
            self.modelling = ModellingConfig(**data["modelling"])
        # Credentials come from the environment. docs/adr/0019
        for name in ("url", "token"):
            value = environ.get(f"STRATA_MODELLING_{name.upper()}")
            if value:
                setattr(self.modelling, name, value)
