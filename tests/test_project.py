"""The job: resolution, validation, paths, the schema it seeds, and edits to its file."""

from pathlib import Path

import pytest

from strata.labels import BBoxSchema, ClassificationSchema, SpanSchema
from strata.project import PROJECT_ENV_VAR, Project, ProjectError, list_projects

# ----------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------


def test_create_then_load_round_trip(project):
    reloaded = Project.load(project.root)
    assert reloaded.name == project.name
    assert reloaded.label_set.task == "classification"
    assert reloaded.label_set.classes == ["cat", "dog"]
    assert reloaded.data.type == "image"


def test_the_scaffold_is_a_commented_file_a_person_can_read(project):
    text = (project.root / "project.toml").read_text()
    assert text.startswith('name = "demo"\n')
    assert "[label_set]" in text and "[model.fresh_params]" in text
    assert "# " in text


def test_a_missing_project_says_what_is_available(make_project):
    make_project("alpha")
    with pytest.raises(ProjectError, match="Available: alpha"):
        Project.load(Path("nope"))


def test_a_section_the_job_does_not_own_is_carried_unread(project):
    # A tool's own section travels with the file; the job neither reads
    # nor refuses it, so the same file serves every tool
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text() + '\n[label_studio]\nconfig = "own.xml"\n')
    assert Project.load(project.root).extensions == {"label_studio": {"config": "own.xml"}}


def test_a_stray_key_at_the_top_is_refused(project):
    toml = project.root / "project.toml"
    toml.write_text("epochs = 3\n" + toml.read_text())
    with pytest.raises(ProjectError, match="Unknown key"):
        Project.load(project.root)


def test_an_unknown_key_inside_a_section_is_refused(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[data]", '[data]\nrooot = "typo"'))
    with pytest.raises(ProjectError, match="Unknown key"):
        Project.load(project.root)


def test_choice_must_be_single_or_multiple(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('choice = "multiple"', 'choice = "maybe"'))
    with pytest.raises(ProjectError, match="must be 'single' or 'multiple'"):
        Project.load(project.root)


def test_an_unknown_task_fails_at_load(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('task = "classification"', 'task = "segmentation"'))
    with pytest.raises(ProjectError, match="task must be one of"):
        Project.load(project.root)


def test_span_parameters_are_refused_on_another_task(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[label_set]", "[label_set]\nmulti_label = true"))
    with pytest.raises(ProjectError, match="span"):
        Project.load(project.root)


def test_choice_is_refused_on_another_task(make_project):
    project = make_project("spans", task="span", sample_type="text")
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[label_set]", '[label_set]\nchoice = "single"'))
    with pytest.raises(ProjectError, match="classification"):
        Project.load(project.root)


def test_a_project_names_its_sample_type(project):
    assert project.sample_type_name == "image"


def test_a_project_without_a_type_says_so_when_asked(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('type = "image"', 'type = ""'))
    with pytest.raises(ProjectError, match=r"\[data\] type is not set"):
        _ = Project.load(project.root).sample_type_name


def test_a_project_declares_the_split_its_corpus_arrived_with(project):
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text().replace(
            "# [catalog.split]\n", '[catalog.split]\nkey = "bench"\nholdout = ["test"]\n'
        )
    )
    given = Project.load(project.root).catalog.given_split
    assert (given.key, given.holdout, given.val) == ("bench", ["test"], [])
    assert project.catalog.given_split is None


def test_a_split_naming_one_set_on_both_sides_is_refused(project):
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text().replace(
            "# [catalog.split]\n",
            '[catalog.split]\nkey = "bench"\nholdout = ["test"]\nval = ["test"]\n',
        )
    )
    with pytest.raises(ProjectError, match="both held out and validation"):
        _ = Project.load(project.root).catalog.given_split


# ----------------------------------------------------------------------
# The schema the job seeds the catalog with
# ----------------------------------------------------------------------


def test_a_classification_job_seeds_a_classification_schema(make_project):
    single = make_project("single", choice="single")
    assert single.label_set.schema == ClassificationSchema(classes=["cat", "dog"], multiple=False)
    multiple = make_project("multiple")
    assert multiple.label_set.schema.multiple is True


def test_a_span_job_seeds_a_span_schema(make_project):
    project = make_project("spans", task="span", sample_type="text", classes=["PER"])
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[label_set]", "[label_set]\nmulti_label = true"))
    assert Project.load(project.root).label_set.schema == SpanSchema(
        classes=["PER"], multi_label=True, overlapping=False
    )


def test_a_bbox_job_seeds_a_bbox_schema(make_project):
    project = make_project("boxes", task="bbox")
    assert project.label_set.schema == BBoxSchema(classes=["cat", "dog"])


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------


def test_a_bare_name_resolves_under_projects(make_project):
    make_project("cats")
    assert Project.load(Path("cats")).name == "cats"


def test_a_path_to_the_toml_resolves_to_its_directory(make_project):
    project = make_project("cats")
    assert Project.load(project.root / "project.toml").name == "cats"


def test_a_single_project_is_found_without_being_named(make_project):
    make_project("only")
    assert Project.load().name == "only"


def test_several_projects_require_a_choice(make_project):
    make_project("alpha")
    make_project("beta")
    with pytest.raises(ProjectError, match="Several projects found"):
        Project.load()


def test_the_environment_variable_selects_a_project(make_project, monkeypatch):
    make_project("alpha")
    make_project("beta")
    monkeypatch.setenv(PROJECT_ENV_VAR, "beta")
    assert Project.load().name == "beta"


def test_list_projects_ignores_directories_without_a_project_file(make_project, tmp_path):
    make_project("real")
    (tmp_path / "projects" / "not-a-project").mkdir()
    assert [p.name for p in list_projects()] == ["real"]


# ----------------------------------------------------------------------
# Paths and names
# ----------------------------------------------------------------------


def test_an_absolute_data_root_is_used_as_is(project, tmp_path):
    elsewhere = tmp_path / "shared-images"
    elsewhere.mkdir()
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('root = "data/raw"', f'root = "{elsewhere}"'))
    assert Project.load(project.root).data_dir == elsewhere


def test_names_default_to_each_other(project):
    assert project.label_set_name == "demo"
    assert project.dataset_name == "demo"
    assert project.collections == ["demo"]


def test_a_file_model_ref_is_anchored_at_the_project(project):
    assert project.model_ref("model.py:Toy") == f"{project.root / 'model.py'}:Toy"
    # A registry name needs no anchoring, and the project's own is the default
    assert project.model_ref() == "multilabel"


# ----------------------------------------------------------------------
# Edits to the file
# ----------------------------------------------------------------------


def test_add_classes_appends_and_persists(project):
    assert project.add_classes(["bird"]) == ["cat", "dog", "bird"]
    # Append-only: a checkpoint maps output neurons to this list by position
    assert Project.load(project.root).label_set.classes == ["cat", "dog", "bird"]


def test_add_classes_refuses_a_duplicate(project):
    with pytest.raises(ProjectError, match="already exists"):
        project.add_classes(["cat"])


def test_add_classes_refuses_an_empty_name(project):
    with pytest.raises(ProjectError, match="cannot be empty"):
        project.add_classes(["   "])


def test_add_classes_pins_the_inferred_order_when_the_list_is_empty(make_project):
    project = make_project("fresh", classes=[])
    assert project.add_classes(["bird"], known=["dog", "cat"]) == ["cat", "dog", "bird"]
    assert Project.load(project.root).label_set.classes == ["cat", "dog", "bird"]


def test_add_classes_leaves_the_rest_of_the_file_alone(project):
    # Comments included: the file is a person's, and an edit that dropped
    # them would be a rewrite dressed up as an edit
    before = (project.root / "project.toml").read_text()
    project.add_classes(["bird"])
    after = (project.root / "project.toml").read_text()
    lines = zip(before.splitlines(), after.splitlines(), strict=True)
    changed = [(a, b) for a, b in lines if a != b]
    assert changed == [('classes = ["cat", "dog"]', 'classes = ["cat", "dog", "bird"]')]
    assert len(before.splitlines()) == len(after.splitlines())


def test_add_classes_writes_the_section_when_the_file_has_none(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    (root / "project.toml").write_text('name = "bare"\n\n[data]\ntype = "image"\n')
    project = Project.load(root)
    project.add_classes(["cat"])
    assert Project.load(root).label_set.classes == ["cat"]


def test_a_tool_writes_its_own_section_without_touching_the_rest(project):
    before = (project.root / "project.toml").read_text()
    project.write_extension("label_studio", {"config": "label_config.xml"})
    after = (project.root / "project.toml").read_text()
    assert after.startswith(before.rstrip("\n"))
    assert Project.load(project.root).extensions == {"label_studio": {"config": "label_config.xml"}}


def test_create_refuses_to_overwrite_an_existing_project(project):
    with pytest.raises(ProjectError, match="already exists"):
        Project.create(project.root, name="demo")
