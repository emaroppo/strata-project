"""Machine-level settings: the catalogs this host reaches, and where it trains.

Everything that belongs to a job lives in the project directory instead.
This file describes the machine; a project carries none of it.

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

from strata.catalog.config import Catalogs, read_catalogs

SETTINGS_FILE = "config.toml"


@dataclass
class ModellingConfig:
    """Where training happens.

    Empty means in this process, which is what a single machine wants and
    what keeps a checkout runnable. A URL sends rounds to a host with the
    GPU: it materialises the dataset itself, so nothing but a dataset id
    travels.
    """

    url: str = ""
    #: Shared with the host. Out of the file by preference, the same
    #: argument as every other credential here.
    token: str = ""


@dataclass
class Settings:
    #: The catalogs this host describes, and which one it uses by default.
    catalogs: Catalogs = field(default_factory=Catalogs)
    modelling: ModellingConfig = field(default_factory=ModellingConfig)

    @classmethod
    def load(cls, path: Path = Path(SETTINGS_FILE), environ: Mapping[str, str] | None = None):
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
        # Credentials belong in the environment rather than in a file: a
        # config file gets pasted, backed up and copied, and a secret in it
        # goes everywhere it does
        for name in ("url", "token"):
            value = environ.get(f"STRATA_MODELLING_{name.upper()}")
            if value:
                setattr(self.modelling, name, value)
