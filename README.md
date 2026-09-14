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
Machine-level settings live in `config.toml` beside the checkout and are
read by `Settings`.
