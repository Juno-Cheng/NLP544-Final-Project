from pathlib import Path

import pytest
import yaml

from trustsim.config import (
    EXPERIMENTS,
    ROLES,
    ConfigError,
    check_experiments,
    load_data_settings,
    load_experiment_settings,
    with_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def write_variant(tmp_path: Path, name: str, change) -> Path:
    raw = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_repository_configs_load():
    data = load_data_settings(CONFIGS / "data.yaml", ROOT)
    settings = load_experiment_settings(CONFIGS / "experiment.yaml", ROOT)
    assert data.seeds_path == ROOT / "data" / "processed" / "seeds.jsonl"
    assert settings.experiments == EXPERIMENTS
    assert set(settings.llm.roles) == set(ROLES)
    assert all(role.prompt_path.is_file() for role in settings.llm.roles.values())


def test_unknown_key_is_rejected(tmp_path):
    path = write_variant(tmp_path, "data.yaml", lambda raw: raw.update(sample_sise=4))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_data_settings(path, ROOT)


def test_missing_key_is_rejected(tmp_path):
    path = write_variant(tmp_path, "experiment.yaml", lambda raw: raw.pop("max_turns"))
    with pytest.raises(ConfigError, match="missing"):
        load_experiment_settings(path, ROOT)


def test_odd_sample_size_cannot_hold_whole_pairs(tmp_path):
    path = write_variant(tmp_path, "data.yaml", lambda raw: raw.update(sample_size=5))
    with pytest.raises(ConfigError, match="even"):
        load_data_settings(path, ROOT)


def test_missing_prompt_file_is_rejected(tmp_path):
    def point_judge_at_missing_prompt(raw):
        raw["llm"]["roles"]["judge"]["prompt"] = "prompts/missing.txt"

    path = write_variant(tmp_path, "experiment.yaml", point_judge_at_missing_prompt)
    with pytest.raises(ConfigError, match="does not exist"):
        load_experiment_settings(path, ROOT)


def test_unknown_provider_is_rejected(experiment_settings):
    with pytest.raises(ConfigError, match="unknown provider"):
        with_overrides(experiment_settings, provider="acme")


def test_model_override_applies_to_every_role(experiment_settings):
    settings = with_overrides(experiment_settings, model="gemini-test")
    assert {role.model for role in settings.llm.roles.values()} == {"gemini-test"}


def test_experiments_are_put_in_standard_order_and_checked():
    assert check_experiments(["dynamic", "static"]) == ("static", "dynamic")
    with pytest.raises(ConfigError, match="unknown"):
        check_experiments(["static", "frozen"])
    with pytest.raises(ConfigError, match="more than once"):
        check_experiments(["static", "static"])
