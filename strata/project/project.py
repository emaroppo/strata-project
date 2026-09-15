"""The project construct: the durable job, as a file that names what it is made of.

A project is a directory whose ``project.toml`` names a catalog, the
collections it draws samples from, the label set it labels them under, and
the model it trains. The samples and the annotations live in the catalog,
machine-level settings in ``config.toml``, and what a tool needs of its own
in a section this package carries without reading.

    projects/my-project/
    ├── project.toml            # this file's schema: catalog, label set, model
    ├── model.py                # optional: a model carried by this project
    ├── data/raw/…              # a corpus before ingest; [data] root, may be elsewhere
    ├── data/source/…           # a corpus before conversion, when it needs one
    ├── datasets/               # materialised dataset versions, files by checksum
    ├── runs/                   # the run store: runs.db, checkpoints/
    └── experiments/            # the ledger of experiment files run over this project

Handing someone the directory hands them the job's definition and its
record, not its data. A project is addressed by name under ``projects/`` or
by path, and an experiment file references one the same way. See
``docs/adr/0016``.
"""

import os
import textwrap
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import tomlkit
from tomlkit.items import Table

from strata.labels import AnySchema, BBoxSchema, ClassificationSchema, SpanSchema
from strata.modelling import Model, ModelError, absolute, resolve

PROJECT_FILE = "project.toml"
PROJECT_ENV_VAR = "STRATA_PROJECT"
# Projects live side by side here, addressable by name: -p cats
PROJECTS_DIR = "projects"

#: What kind of annotation a job collects. The catalog's schema for each is
#: in ``strata.labels``; the media it is collected over is the sample type's.
TASKS = ("classification", "bbox", "span")


class ProjectError(Exception):
    """Raised for a missing, malformed, or inconsistent project."""


@dataclass
class LabelSetSpec:
    """Which classes this job labels, and what shape an annotation takes.

    The label set in the catalog is authoritative once it exists: this
    section seeds it, and ``add_classes`` extends this section while the
    tool widens the label set to match. ``docs/adr/0014`` for why the shape
    is declared here.
    """

    task: str = "classification"
    classes: list[str] = field(default_factory=list)
    #: Classification only: "single" for mutually exclusive classes.
    choice: str | None = None
    #: Span only. None means not declared, so setting either on another
    #: task is refused by name. docs/adr/0016
    multi_label: bool | None = None
    overlapping: bool | None = None

    @property
    def schema(self) -> AnySchema:
        """This label set as the catalog stores it."""
        if self.task == "classification":
            return ClassificationSchema(
                classes=list(self.classes), multiple=self.choice != "single"
            )
        if self.task == "span":
            return SpanSchema(
                classes=list(self.classes),
                multi_label=bool(self.multi_label),
                overlapping=bool(self.overlapping),
            )
        return BBoxSchema(classes=list(self.classes))


@dataclass
class ModelSpec:
    # "<file>.py:Class" resolves inside the project; "pkg.module:Class"
    # falls back to an installed package
    ref: str = "multilabel"
    params: dict = field(default_factory=dict)
    # Merged over params when a round starts cold. docs/adr/0025
    fresh_params: dict = field(default_factory=dict)

    def params_for(self, fresh: bool) -> dict:
        return {**self.params, **self.fresh_params} if fresh else dict(self.params)


@dataclass
class CatalogSpec:
    """Which catalog, and which label set inside it, this job uses.

    Where each catalog *is* lives in config.toml; which one this job draws
    from travels with the project. See ``docs/adr/0020``.
    """

    #: Which catalog on this host, by the name it has in config.toml. Empty
    #: means the host's default. docs/adr/0020
    name: str = ""
    #: Defaults to the project's own name, which is the label set ingest
    #: creates.
    label_set: str = ""
    #: The dataset name versions accumulate under; defaults to the label set.
    dataset: str = ""
    #: Which collections in the catalog this job draws from, as paths:
    #: ["sat_images"] takes everything under it, ["sat_images/2024"] one
    #: batch. Defaults to a collection named after the label set. Dropping
    #: one declares that data out of scope, training included.
    collections: list[str] = field(default_factory=list)
    #: A metadata key whose values stay on one side of a split: ``video``
    #: for frames. Empty means every sample is its own group. docs/adr/0023
    group_by: str = ""
    #: A split the corpus arrived with, read off a metadata key: ``key``,
    #: and which of its values are ``holdout`` and which ``val``. Empty
    #: means every side is drawn by ratio.
    split: dict = field(default_factory=dict)

    @property
    def given_split(self):
        """The split as the catalog reads it, or None when none is declared."""
        if not self.split:
            return None
        from strata.catalog import GivenSplit

        try:
            return GivenSplit(**self.split)
        except (TypeError, ValueError) as e:
            raise ProjectError(f"[catalog.split]: {e}") from None


@dataclass
class DataSpec:
    root: str = "data/raw"
    #: Where the corpus arrives, before anything has converted it.
    #: docs/adr/0010
    source_root: str = "data/source"
    #: Which conversion to run over it. Empty resolves by what the files
    #: are and what this project ingests, and refuses an ambiguity.
    preparer: str = ""
    #: A registered sample type: what these files are, which decides which
    #: extensions are allowed, what is recorded about each one, and how
    #: they group.
    type: str = ""
    #: What this job's model is told about a sample besides its bytes, each
    #: naming its source, ``label_set`` or ``metadata`` (``docs/adr/0011``):
    #:
    #:     [[data.features]]
    #:     name = "species"
    #:     source = "label_set"
    #:     ref = "plant-species"
    features: list[dict] = field(default_factory=list)


@dataclass
class Project:
    root: Path
    name: str
    label_set: LabelSetSpec = field(default_factory=LabelSetSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    catalog: CatalogSpec = field(default_factory=CatalogSpec)
    #: Sections the job does not own, by name, as read; the job carries
    #: them unread. docs/adr/0016
    extensions: dict[str, dict] = field(default_factory=dict)

    SECTIONS = ("label_set", "model", "data", "catalog")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None):
        root = _resolve_root(path)
        toml_path = root / PROJECT_FILE
        if not toml_path.exists():
            available = [p.name for p in list_projects()]
            hint = (
                f" Available: {', '.join(available)}."
                if available
                else f" Create one with 'strata-labeller new {root.name}'."
            )
            raise ProjectError(f"No {PROJECT_FILE} in {root}.{hint}")
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        stray = [k for k, v in data.items() if k != "name" and not isinstance(v, dict)]
        if stray:
            raise ProjectError(
                f"Unknown key(s) at the top of {toml_path}: {', '.join(sorted(stray))}"
            )
        project = cls(
            root=root,
            name=data.get("name", root.name),
            label_set=section(LabelSetSpec, data.get("label_set", {}), "label_set"),
            model=section(ModelSpec, data.get("model", {}), "model"),
            data=section(DataSpec, data.get("data", {}), "data"),
            catalog=section(CatalogSpec, data.get("catalog", {}), "catalog"),
            extensions={k: v for k, v in data.items() if k != "name" and k not in cls.SECTIONS},
        )
        project._validate()
        return project

    def _validate(self) -> None:
        spec = self.label_set
        if spec.task not in TASKS:
            raise ProjectError(
                f"[label_set] task must be one of {', '.join(TASKS)}; got '{spec.task}'"
            )
        if spec.choice not in {None, "single", "multiple"}:
            raise ProjectError(
                f"[label_set] choice must be 'single' or 'multiple', got '{spec.choice}'"
            )
        if spec.choice is not None and spec.task != "classification":
            raise ProjectError('[label_set] choice applies to task = "classification" only')
        if spec.task != "span" and (spec.multi_label is not None or spec.overlapping is not None):
            raise ProjectError(
                '[label_set] multi_label and overlapping apply to task = "span" only'
            )

    # ------------------------------------------------------------------
    # Names and paths
    # ------------------------------------------------------------------

    @property
    def label_set_name(self) -> str:
        return self.catalog.label_set or self.name

    @property
    def dataset_name(self) -> str:
        return self.catalog.dataset or self.label_set_name

    @property
    def collections(self) -> list[str]:
        """Where this job draws its samples from."""
        return self.catalog.collections or [self.label_set_name]

    @property
    def datasets_dir(self) -> Path:
        """Where materialised dataset versions are written."""
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        """The model catalog for this project: runs, metrics and checkpoints."""
        return self.root / "runs"

    @property
    def data_dir(self) -> Path:
        """Where this project's samples live, whatever kind of file they are."""
        return resolve_under(self.root, self.data.root)

    @property
    def source_dir(self) -> Path:
        """Where this project's corpus arrives, before it is converted."""
        return resolve_under(self.root, self.data.source_root)

    @property
    def feature_specs(self) -> list:
        """The declarations, validated. Empty for a project with none."""
        from strata.catalog.versions.features import FeatureSpec

        return [FeatureSpec.from_dict(raw) for raw in self.data.features]

    @property
    def sample_type_name(self) -> str:
        """Which registered type this project's files are."""
        if not self.data.type:
            raise ProjectError(
                "[data] type is not set: a registered sample type, see 'strata-catalog types'"
            )
        return self.data.type

    def sample_type(self):
        """The type itself, resolved from what is installed."""
        from strata.catalog.types.sample_types import SampleTypeError, resolve

        name = self.sample_type_name
        try:
            return resolve(name)()
        except SampleTypeError as e:
            raise ProjectError(f"[data] type: {e}") from None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def model_ref(self, ref: str | None = None) -> str:
        """A model reference anchored at the project, so it resolves from anywhere.

        ``ref`` is the project's own unless an experiment names another. See
        ``docs/adr/0016``.
        """
        return absolute(self.model.ref if ref is None else ref, self.root)

    def load_model(self) -> Model:
        """Instantiate the project's model with its configured parameters.

        Resolution is ``strata.modelling``'s: a short name goes through the
        registry, and anything with a ``:`` is a direct reference, either a
        ``file.py:Class`` carried by the project or an installed
        ``module:Class``. A missing ML framework surfaces here.
        """
        try:
            model_cls = resolve(self.model.ref, root=self.root)
        except ModelError as exc:
            raise ProjectError(str(exc)) from exc
        return model_cls(**self.model.params)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_classes(self, names: list[str], known: list[str] | None = None) -> list[str]:
        """Append classes to project.toml and return the new full list.

        Classes are append-only (``docs/adr/0005``). When the list is empty
        the classes in use are written
        out first (``known``), turning an inferred order into a pinned one.
        """
        classes = list(self.label_set.classes) or sorted(known or [])
        for name in names:
            if not name.strip():
                raise ProjectError("Class names cannot be empty")
            if name in classes:
                raise ProjectError(f"Class '{name}' already exists in {PROJECT_FILE}")
            classes.append(name)

        document = self._document()
        if "label_set" not in document:
            document.add("label_set", tomlkit.table())
        document["label_set"]["classes"] = classes
        self._write(document)
        self.label_set.classes = classes
        return classes

    def write_extension(self, name: str, values: dict) -> None:
        """Write a tool's own section, leaving the rest of the file as it is."""
        document = self._document()
        table = tomlkit.table()
        for key, value in values.items():
            table.add(key, value)
        document[name] = table
        self._write(document)
        self.extensions[name] = dict(values)

    def _document(self) -> tomlkit.TOMLDocument:
        return tomlkit.parse((self.root / PROJECT_FILE).read_text())

    def _write(self, document: tomlkit.TOMLDocument) -> None:
        (self.root / PROJECT_FILE).write_text(tomlkit.dumps(document))

    # ------------------------------------------------------------------
    # Scaffolding
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        name: str | None = None,
        classes: list[str] | None = None,
        task: str = "classification",
        choice: str = "multiple",
        sample_type: str = "image",
    ):
        """Write a commented project.toml and return the project, loaded."""
        if (root / PROJECT_FILE).exists():
            raise ProjectError(f"{root / PROJECT_FILE} already exists")
        if task not in TASKS:
            raise ProjectError(f"task must be one of {', '.join(TASKS)}; got '{task}'")
        name = name or root.resolve().name
        (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
        document = _scaffold(name, task, classes or [], choice, sample_type)
        (root / PROJECT_FILE).write_text(tomlkit.dumps(document))
        return cls.load(root)


_LABEL_SET_NOTE = """
            multi_label = true   # one region may carry several labels
            overlapping = true   # two regions may intersect
"""

_DATA_NOTE = """
        Where a corpus arrives if it needs converting first: mail, video.
        See 'strata-catalog preparers' and 'prepare'.
        source_root = "data/source"
"""

_CATALOG_NOTE = """
        A metadata key whose values stay on one side of a split:
        "video" for frames. Empty, every sample is its own group.
        group_by = "video"

        A split the corpus arrived with, read off a metadata key each
        sample carries: which of its values are held out, which are
        validation. The rest is drawn by ratio.
        [catalog.split]
        key = "benchmark_split"
        holdout = ["test"]
        val = ["dev"]
"""

_MODEL_NOTE = """
        Applied over the above when a round has nothing to continue
        from. A cold start on an increment's schedule undertrains,
        and the result then reads as a baseline.
"""


def _scaffold(
    name: str, task: str, classes: list[str], choice: str, sample_type: str
) -> tomlkit.TOMLDocument:
    doc = tomlkit.document()
    doc.add("name", name)

    label_set = tomlkit.table()
    label_set.add("task", tomlkit.item(task).comment(", ".join(TASKS)))
    label_set.add("classes", classes)
    if task == "classification":
        label_set.add(
            "choice", tomlkit.item(choice).comment('"single" for mutually exclusive classes')
        )
    if task == "span":
        _note(label_set, _LABEL_SET_NOTE)
    doc.add("label_set", label_set)

    data = tomlkit.table()
    data.add("root", tomlkit.item("data/raw").comment("files live here; may be an absolute path"))
    data.add(
        "type",
        tomlkit.item(sample_type).comment("a registered sample type; see 'strata-catalog types'"),
    )
    _note(data, _DATA_NOTE)
    doc.add("data", data)

    catalog = tomlkit.table()
    _note(catalog, _CATALOG_NOTE)
    doc.add("catalog", catalog)

    model = tomlkit.table()
    _note(model, '"model.py:MyModel" to use a model carried by this project')
    model.add("ref", tomlkit.item("multilabel").comment('a registered name, or "model.py:MyModel"'))
    params = tomlkit.table()
    params.add("num_epochs", 4)
    params.add("batch_size", 16)
    params.add("lr", 5e-5)
    model.add("params", params)
    model.add(tomlkit.nl())
    _note(model, _MODEL_NOTE)
    fresh = tomlkit.table()
    fresh.add("num_epochs", 8)
    model.add("fresh_params", fresh)
    doc.add("model", model)
    return doc


def _note(table: Table, text: str) -> None:
    """A comment of one or more lines, into ``table`` where it stands."""
    for line in textwrap.dedent(text).strip("\n").splitlines():
        table.add(tomlkit.comment(line) if line else tomlkit.nl())


def resolve_under(root: Path, value: str) -> Path:
    """``value`` as a path: relative to ``root`` unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def list_projects(base: Path | None = None) -> list[Path]:
    """Every project directory under ``projects/`` (or another base)."""
    base = base if base is not None else Path(PROJECTS_DIR)
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if (d / PROJECT_FILE).exists())


def _resolve_named(path: Path) -> Path:
    """A path to a project directory, or a bare name under ``projects/``."""
    if path.name == PROJECT_FILE:
        path = path.parent
    if (path / PROJECT_FILE).exists():
        return path.resolve()
    named = Path(PROJECTS_DIR) / path
    if (named / PROJECT_FILE).exists():
        return named.resolve()
    # Absolute, so nothing handed on depends on cwd (docs/adr/0016); load()
    # raises the missing-project.toml error from here
    return path.resolve()


def _resolve_root(path: Path | None) -> Path:
    if path is not None:
        return _resolve_named(path)

    env = os.environ.get(PROJECT_ENV_VAR)
    if env:
        return _resolve_named(Path(env))
    if (Path(".") / PROJECT_FILE).exists():
        return Path(".").resolve()

    # Bare command inside a repo with a projects/ folder: unambiguous only
    # when there is exactly one project
    candidates = list_projects()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ProjectError(
            f"Several projects found; pass --project NAME (one of: {names}) "
            f"or set ${PROJECT_ENV_VAR}."
        )
    return Path(".").resolve()


def section(spec: type, data: dict, name: str):
    """``data`` as ``spec``, refusing a key the section does not have."""
    known = {f.name for f in fields(spec)}
    unknown = set(data) - known
    if unknown:
        raise ProjectError(
            f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))} "
            f"(known: {', '.join(sorted(known))})"
        )
    return spec(**data)
