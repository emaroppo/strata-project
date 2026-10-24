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
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# Which file a command reads is settled in the catalog, so that every
# command line, strata-catalog included, resolves it the same way. Both
# names stay importable from here. docs/adr/0019
from strata.catalog.config import (
    SETTINGS_FILE,  # noqa: F401
    CatalogConfigError,
    Catalogs,
    read_catalogs,
    settings_path,
)


class ModellingConfigError(CatalogConfigError):
    """``[modelling]`` describes no host this command can use."""


@dataclass
class ModellingConfig:
    """Where training happens.

    Empty means in this process. A URL sends rounds to a host with the
    GPU, which materialises the dataset itself. See ``docs/adr/0007``.
    """

    url: str = ""
    #: Shared with the host, and read from the environment. docs/adr/0019
    token: str = ""
    #: What config.toml calls it; empty for a host's single, unnamed one.
    name: str = ""

    def describe(self) -> str:
        """The host as a person would recognise it: its name and where it is."""
        where = self.url or "this machine"
        return f"{self.name} ({where})" if self.name else where

    def token_from(self) -> str:
        """Where its token is read from, for the error that says it is missing."""
        if not self.name:
            return "$STRATA_MODELLING_TOKEN"
        return f"${_token_variable(self.name)} or $STRATA_MODELLING_TOKEN"


#: What ``[modelling]`` and each ``[modelling.<name>]`` may say.
_MODELLING_KEYS = frozenset({"url", "token"})
#: Only in a ``[modelling.<name>]``: the catalogs whose projects it trains
#: when nothing names a host. docs/adr/0043
_DEFAULT_FOR = "default_for"


@dataclass
class ModellingHosts:
    """Every modelling host this machine describes, and which one it uses by default.

    The same shape as the catalogs, for the same reason: a run and its
    checkpoint live on the host that trained it, so asking another one for
    "the latest run" answers about a different chain, and nothing raises.
    See ``docs/adr/0043``.
    """

    #: The one used when nothing names another.
    default: ModellingConfig = field(default_factory=ModellingConfig)
    #: From ``[modelling.<name>]`` tables. Empty with a flat ``[modelling]`` only.
    by_name: dict[str, ModellingConfig] = field(default_factory=dict)
    #: Which name :attr:`default` resolved to. Empty for a flat ``[modelling]``,
    #: or when several are described and none is marked — see :meth:`named`.
    default_name: str = ""
    #: Catalog name to the host its projects train on when nothing names
    #: one, from each table's ``default_for``.
    by_catalog: dict[str, str] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.by_name)

    def all(self) -> list[ModellingConfig]:
        """Every host described, for the commands that ask each one."""
        return [self.by_name[n] for n in self.names()] or [self.default]

    def named(self, name: str = "") -> ModellingConfig:
        """One host by name, or the default when nothing is named.

        A name that does not exist is refused with the list of the ones
        that do, and several with no default is refused here rather than
        guessed — training in this process instead would be a guess too.
        """
        if not name:
            if not self.default_name and self.by_name:
                raise ModellingConfigError(
                    f"This machine has several modelling hosts "
                    f"({', '.join(self.names())}) and nothing says which is the "
                    f'default. Add [modelling] default = "<name>", name one in a '
                    f"project's [model] host, or pass --host."
                )
            return self.default
        try:
            return self.by_name[name]
        except KeyError:
            known = ", ".join(self.names()) or "none — [modelling] has no named hosts"
            raise ModellingConfigError(
                f"No modelling host named {name!r} on this machine. Configured: {known}."
            ) from None


def _token_variable(name: str) -> str:
    """The variable a named host's token comes from: ``gx10`` reads
    ``$STRATA_MODELLING_TOKEN_GX10``."""
    return "STRATA_MODELLING_TOKEN_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def read_modelling(section: dict, environ: Mapping[str, str]) -> ModellingHosts:
    """A flat ``[modelling]``, or ``[modelling.<name>]`` tables, or both.

    Flat keys are shared by every named table, which layers on top of them,
    as ``[catalog]`` does. Tokens come from the environment: a named host's
    own variable first, then ``$STRATA_MODELLING_TOKEN`` for them all.
    ``$STRATA_MODELLING_URL`` stands in for the flat url only, since with
    several hosts it could not say which it replaces. See ``docs/adr/0019``.
    """
    scalars = {k: v for k, v in section.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in section.items() if isinstance(v, dict)}
    chosen = scalars.pop("default", "")
    if _DEFAULT_FOR in scalars:
        raise ModellingConfigError(
            f"[modelling] {_DEFAULT_FOR} says which host a catalog's projects "
            f"train on, so it belongs in that host's [modelling.<name>] table."
        )
    claims: dict[str, str] = {}
    for name, table in tables.items():
        catalogs = table.get(_DEFAULT_FOR, [])
        if not isinstance(catalogs, list) or not all(isinstance(c, str) for c in catalogs):
            raise ModellingConfigError(
                f'[modelling.{name}] {_DEFAULT_FOR} is a list of catalog names: ["main"].'
            )
        for catalog in catalogs:
            if catalog in claims:
                raise ModellingConfigError(
                    f"Catalog {catalog!r} is in the {_DEFAULT_FOR} of both "
                    f"[modelling.{claims[catalog]}] and [modelling.{name}]; one "
                    f"host is its default."
                )
            claims[catalog] = name
    tables = {
        name: {k: v for k, v in table.items() if k != _DEFAULT_FOR}
        for name, table in tables.items()
    }
    for where, keys in [("modelling", scalars)] + [
        (f"modelling.{n}", t) for n, t in tables.items()
    ]:
        unknown = set(keys) - _MODELLING_KEYS
        if unknown:
            raise ModellingConfigError(
                f"Unknown key(s) in [{where}]: {', '.join(sorted(unknown))} "
                f"(known: {', '.join(sorted(_MODELLING_KEYS))})"
            )

    shared_token = environ.get("STRATA_MODELLING_TOKEN", "")
    env_url = environ.get("STRATA_MODELLING_URL", "")

    if not tables:
        if chosen:
            raise ModellingConfigError(
                f"[modelling] default = {chosen!r}, but no [modelling.<name>] "
                f"tables are defined. This machine has one modelling host."
            )
        base = ModellingConfig(**scalars)
        base.url = env_url or base.url
        base.token = shared_token or base.token
        return ModellingHosts(default=base)

    if env_url:
        raise ModellingConfigError(
            f"$STRATA_MODELLING_URL is set, and [modelling] names several hosts "
            f"({', '.join(sorted(tables))}), so it could replace any of them. Put "
            f"the url in that host's table instead."
        )
    by_name = {}
    for name, table in tables.items():
        config = ModellingConfig(**{**scalars, **table}, name=name)
        config.token = environ.get(_token_variable(name)) or shared_token or config.token
        by_name[name] = config

    if chosen:
        if chosen not in by_name:
            raise ModellingConfigError(
                f"[modelling] default = {chosen!r} names no host. "
                f"Configured: {', '.join(sorted(by_name))}."
            )
        return ModellingHosts(
            default=by_name[chosen], by_name=by_name, default_name=chosen, by_catalog=claims
        )
    if len(by_name) == 1:
        [(name, only)] = by_name.items()
        return ModellingHosts(default=only, by_name=by_name, default_name=name, by_catalog=claims)
    # Unresolved, and not an error yet: a project, a catalog's default or
    # --host naming one settles it
    return ModellingHosts(by_name=by_name, by_catalog=claims)


@dataclass
class Settings:
    #: The catalogs this host describes, and which one it uses by default.
    catalogs: Catalogs = field(default_factory=Catalogs)
    modelling: ModellingHosts = field(default_factory=ModellingHosts)

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
        self.modelling = read_modelling(data.get("modelling", {}), environ)
        known = self.catalogs.names()
        for catalog, host in sorted(self.modelling.by_catalog.items()):
            if catalog not in known:
                raise ModellingConfigError(
                    f"[modelling.{host}] {_DEFAULT_FOR} names catalog {catalog!r}, "
                    f"which this machine does not describe. Configured: {', '.join(known)}."
                )

    def modelling_for(self, catalog: str = "", host: str = "") -> ModellingConfig:
        """The host a project on ``catalog`` trains on.

        ``host`` when given (``--host``, else the project's ``[model]
        host``); otherwise the host whose ``default_for`` lists the catalog,
        since only a host that reads a catalog can train from it; otherwise
        the machine's default. An empty ``catalog`` is the machine's default
        catalog. See ``docs/adr/0043``.
        """
        if host:
            return self.modelling.named(host)
        claimed = self.modelling.by_catalog.get(catalog or self.catalogs.default_name)
        return self.modelling.named(claimed or "")
