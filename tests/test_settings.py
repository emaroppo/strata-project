"""Host settings: the catalogs a machine reaches, and where it trains."""

from pathlib import Path

import pytest

from strata.catalog.config import CatalogConfigError
from strata.project import Settings, settings_path


def _write(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def test_settings_read_the_catalogs_and_the_modelling_host(tmp_path):
    path = _write(
        tmp_path,
        """
[catalog]
root = "catalog"

[modelling]
url = "http://gpu:8082"
""",
    )
    settings = Settings.load(path, environ={})
    assert settings.catalogs.named("").root == "catalog"
    assert settings.modelling.named().url == "http://gpu:8082"
    assert settings.modelling.named().token == ""


def test_credentials_come_from_the_environment(tmp_path):
    settings = Settings.load(
        _write(tmp_path, '[modelling]\nurl = "http://gpu:8082"\n'),
        environ={"STRATA_MODELLING_TOKEN": "secret"},
    )
    assert settings.modelling.named().token == "secret"


def test_a_section_another_tool_owns_is_left_alone(tmp_path):
    path = _write(tmp_path, '[label_studio]\nurl = "http://ls:8080"\n')
    settings = Settings.load(path, environ={})
    assert settings.modelling.named().url == ""


def test_a_missing_file_is_the_defaults(tmp_path):
    assert Settings.load(tmp_path / "absent.toml", environ={}).modelling.named().url == ""


def test_an_explicit_path_beats_the_environment(tmp_path):
    given = _write(tmp_path, "")
    assert settings_path(given, environ={"STRATA_CONFIG": "/elsewhere.toml"}) == given


def test_the_environment_names_the_file(tmp_path):
    path = _write(tmp_path, '[modelling]\nurl = "http://gpu:8082"\n')
    environ = {"STRATA_CONFIG": str(path)}
    assert settings_path(environ=environ) == path
    assert Settings.load(environ=environ).modelling.named().url == "http://gpu:8082"


def test_the_environment_naming_a_missing_file_is_refused(tmp_path):
    with pytest.raises(CatalogConfigError, match="STRATA_CONFIG"):
        settings_path(environ={"STRATA_CONFIG": str(tmp_path / "absent.toml")})


def test_nothing_set_is_the_file_beside_the_command():
    assert settings_path(environ={}) == Path("config.toml")


SEVERAL = """
[modelling]
default = "gpu"

[modelling.gpu]
url = "http://gpu:8082"

[modelling.gx10]
url = "http://gx10:8082"
"""


def test_a_flat_modelling_section_is_one_unnamed_host(tmp_path):
    hosts = Settings.load(_write(tmp_path, '[modelling]\nurl = "http://gpu:8082"\n'), environ={})
    assert hosts.modelling.names() == []
    assert hosts.modelling.named("").name == ""


def test_named_hosts_and_the_default_among_them(tmp_path):
    hosts = Settings.load(_write(tmp_path, SEVERAL), environ={}).modelling
    assert hosts.names() == ["gpu", "gx10"]
    assert hosts.named("").url == "http://gpu:8082"
    assert hosts.named("gx10").url == "http://gx10:8082"
    assert hosts.named("gx10").describe() == "gx10 (http://gx10:8082)"


def test_a_named_hosts_own_token_beats_the_shared_one(tmp_path):
    environ = {"STRATA_MODELLING_TOKEN": "shared", "STRATA_MODELLING_TOKEN_GX10": "own"}
    hosts = Settings.load(_write(tmp_path, SEVERAL), environ=environ).modelling
    assert hosts.named("gx10").token == "own"
    assert hosts.named("gpu").token == "shared"


def test_an_unknown_host_is_refused_with_the_ones_that_exist(tmp_path):
    hosts = Settings.load(_write(tmp_path, SEVERAL), environ={}).modelling
    with pytest.raises(CatalogConfigError, match="gpu, gx10"):
        hosts.named("dgx")


def test_several_hosts_and_no_default_is_refused_when_asked_for_the_default(tmp_path):
    body = SEVERAL.replace('default = "gpu"\n', "")
    hosts = Settings.load(_write(tmp_path, body), environ={}).modelling
    with pytest.raises(CatalogConfigError, match="--host"):
        hosts.named("")
    assert hosts.named("gx10").url == "http://gx10:8082"


def test_one_named_host_is_the_default(tmp_path):
    body = '[modelling.gx10]\nurl = "http://gx10:8082"\n'
    assert Settings.load(_write(tmp_path, body), environ={}).modelling.named("").name == "gx10"


def test_a_default_naming_no_host_is_refused(tmp_path):
    with pytest.raises(CatalogConfigError, match="names no host"):
        Settings.load(_write(tmp_path, SEVERAL.replace('"gpu"\n', '"dgx"\n', 1)), environ={})


def test_a_url_from_the_environment_with_several_hosts_is_refused(tmp_path):
    with pytest.raises(CatalogConfigError, match="STRATA_MODELLING_URL"):
        Settings.load(_write(tmp_path, SEVERAL), environ={"STRATA_MODELLING_URL": "http://x"})


def test_an_unknown_modelling_key_is_refused(tmp_path):
    with pytest.raises(CatalogConfigError, match=r"modelling\.gx10"):
        Settings.load(_write(tmp_path, '[modelling.gx10]\nurl = "u"\nport = 1\n'), environ={})


CATALOG_DEFAULTS = """
[catalog]
default = "main"

[catalog.main]
root = "main"

[catalog.local]
root = "local"

[modelling]
default = "here"

[modelling.here]

[modelling.gx10]
url = "http://gx10:8082"
default_for = ["main"]
"""


def test_a_catalogs_projects_train_on_the_host_that_claims_it(tmp_path):
    settings = Settings.load(_write(tmp_path, CATALOG_DEFAULTS), environ={})
    assert settings.modelling_for("main").name == "gx10"
    # The default catalog is main, so a project naming none goes there too
    assert settings.modelling_for("").name == "gx10"
    assert settings.modelling_for("local").name == "here"
    assert settings.modelling_for("local").url == ""


def test_a_named_host_beats_the_catalogs_default(tmp_path):
    settings = Settings.load(_write(tmp_path, CATALOG_DEFAULTS), environ={})
    assert settings.modelling_for("main", "here").name == "here"


def test_default_for_naming_an_unknown_catalog_is_refused(tmp_path):
    body = CATALOG_DEFAULTS.replace('default_for = ["main"]', 'default_for = ["mian"]')
    with pytest.raises(CatalogConfigError, match="mian"):
        Settings.load(_write(tmp_path, body), environ={})


def test_two_hosts_claiming_one_catalog_is_refused(tmp_path):
    body = CATALOG_DEFAULTS.replace(
        "[modelling.here]\n", '[modelling.here]\ndefault_for = ["main"]\n'
    )
    with pytest.raises(CatalogConfigError, match="one host is its default"):
        Settings.load(_write(tmp_path, body), environ={})


def test_default_for_in_the_flat_section_is_refused(tmp_path):
    body = '[modelling]\nurl = "http://gpu:8082"\ndefault_for = ["default"]\n'
    with pytest.raises(CatalogConfigError, match="belongs in that host's"):
        Settings.load(_write(tmp_path, body), environ={})
