"""Shared fixtures: the repository's personas and settings, and a small CGA-format dataset."""

from pathlib import Path

import pytest

from trustsim.config import (
    MOCK_PROVIDER,
    DataSettings,
    ExperimentSettings,
    load_experiment_settings,
    with_overrides,
)
from trustsim.data.loader import Example, load_examples
from trustsim.dialogue import Seed, Turn
from trustsim.personas.cast import Cast, load_cast

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def cast() -> Cast:
    return load_cast(ROOT / "configs" / "personas.yaml")


@pytest.fixture
def experiment_settings() -> ExperimentSettings:
    settings = load_experiment_settings(ROOT / "configs" / "experiment.yaml", ROOT)
    return with_overrides(settings, provider=MOCK_PROVIDER)


@pytest.fixture
def data_settings() -> DataSettings:
    return DataSettings(
        seeds_path=FIXTURES / "seeds.jsonl",
        labels_path=FIXTURES / "labels.jsonl",
        opening_turns=3,
        min_opening_turns=3,
        splits=("test",),
        verified_only=False,
        sample_size=None,
        keep_pairs=True,
        random_seed=544,
    )


@pytest.fixture
def examples(data_settings: DataSettings) -> list[Example]:
    return load_examples(data_settings)


@pytest.fixture
def seed() -> Seed:
    """A two-speaker opening. alice speaks first (persona A) and last, so bob replies first."""
    return Seed(
        "c1",
        (
            Turn("alice", "I removed the date because the source gives a different year."),
            Turn("bob", "That source is outdated."),
            Turn("alice", "Then please cite a newer one."),
        ),
    )
