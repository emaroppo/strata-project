# strata-project

A project is the durable job: a directory whose `project.toml` names a
catalog, the collections it draws samples from, the label set it labels
them under, and the model it trains. The labeller runs the job and
experiments vary it; both read the same file through this package.

```toml
name = "my-project"

[label_set]
task = "classification"              # classification, bbox or span
classes = ["cat", "dog"]
choice = "multiple"                  # "single" for mutually exclusive classes

[data]
type = "image"                       # a registered sample type

[catalog]
name = "images"                      # which catalog on this host
collections = ["my_images"]          # which collections this job draws from

[model]
ref = "multilabel"                   # a registered name, or model.py:MyModel

[model.params]
num_epochs = 4
```

A tool keeps what is its own in a section of its own, which this package
carries without reading: the labeller's `[label_studio]`, for instance.
Machine-level settings are `Settings`, read from the file `$STRATA_CONFIG`
names, or `config.toml` in the working directory: where the catalogs and
the modelling host are on this machine, and nothing about any job.

## Decisions

`docs/adr/NNNN`, wherever this package's code says it, is a record in the strata umbrella repository: https://github.com/emaroppo/strata/tree/main/docs/adr.

## Tests

```bash
.github/sibling-wheels.sh labels common catalog modelling   # the strata packages this one needs, until they are on an index
uv sync --find-links dist --group dev --extra test
uv run pytest
```

Inside the strata workspace: `uv run pytest packages/project` from its root.
