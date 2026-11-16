import importlib.util
import logging
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location(
        "run_experiment", ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_mock_run_from_the_command_line(script, monkeypatch, tmp_path, data_settings, capsys):
    monkeypatch.setattr(script, "load_data_settings", lambda path, root: data_settings)
    code = script.main(["--run-name", "cli-test", "--mock", "--output-dir", str(tmp_path)])
    assert code == 0
    run_dir = tmp_path / "cli-test"
    assert (run_dir / "summary.md").is_file()
    assert (run_dir / "run.log").stat().st_size > 0
    assert "| Match rate |" in capsys.readouterr().out


def test_live_run_without_an_api_key_stops_before_any_call(
    script, monkeypatch, tmp_path, data_settings
):
    monkeypatch.setattr(script, "load_data_settings", lambda path, root: data_settings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert script.main(["--run-name", "no-key", "--output-dir", str(tmp_path)]) == 1
    assert not (tmp_path / "no-key" / "run.json").exists()


def test_odd_sample_size_is_rejected(script, tmp_path):
    args = ["--run-name", "odd", "--mock", "--sample-size", "3", "--output-dir", str(tmp_path)]
    assert script.main(args) == 1


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden"])
def test_run_name_must_be_a_plain_folder_name(script, tmp_path, name):
    with pytest.raises(SystemExit):
        script.main(["--run-name", name, "--mock", "--output-dir", str(tmp_path)])


def test_mock_and_provider_cannot_be_combined(script, tmp_path):
    with pytest.raises(SystemExit):
        script.main(
            ["--run-name", "x", "--mock", "--provider", "gemini", "--output-dir", str(tmp_path)]
        )
