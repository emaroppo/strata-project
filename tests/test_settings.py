"""Host settings: the catalogs a machine reaches, and where it trains."""

from strata.project import Settings


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
    assert settings.modelling.url == "http://gpu:8082"
    assert settings.modelling.token == ""


def test_credentials_come_from_the_environment(tmp_path):
    settings = Settings.load(
        _write(tmp_path, '[modelling]\nurl = "http://gpu:8082"\n'),
        environ={"STRATA_MODELLING_TOKEN": "secret"},
    )
    assert settings.modelling.token == "secret"


def test_a_section_another_tool_owns_is_left_alone(tmp_path):
    path = _write(tmp_path, '[label_studio]\nurl = "http://ls:8080"\n')
    settings = Settings.load(path, environ={})
    assert settings.modelling.url == ""


def test_a_missing_file_is_the_defaults(tmp_path):
    assert Settings.load(tmp_path / "absent.toml", environ={}).modelling.url == ""
