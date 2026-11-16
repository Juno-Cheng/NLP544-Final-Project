from pathlib import Path

import pytest
import yaml

from trustsim.config import ConfigError
from trustsim.dialogue import Seed, Turn
from trustsim.personas.cast import assign_personas, load_cast

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "configs" / "personas.yaml"


def write_variant(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(PERSONAS.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "personas.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_personas_differ_but_share_one_starting_trust(cast):
    a, b = cast.personas["A"], cast.personas["B"]
    assert a.temperament != b.temperament
    assert (a.alpha_up, a.alpha_down, a.forgiveness) != (b.alpha_up, b.alpha_down, b.forgiveness)
    assert 0 <= cast.starting_trust <= 1


@pytest.mark.parametrize(
    ("trust", "band"),
    [
        (1.0, "high"),
        (0.70, "high"),
        (0.6999, "middle"),
        (0.40, "middle"),
        (0.3999, "low"),
        (0.15, "low"),
        (0.1499, "very_low"),
        (0.0, "very_low"),
    ],
)
def test_trust_falls_in_the_expected_mood_band(cast, trust, band):
    assert cast.band(trust) == band


def test_each_persona_has_its_own_mood_for_the_same_trust(cast):
    band_a, mood_a = cast.mood("A", 0.1)
    band_b, mood_b = cast.mood("B", 0.1)
    assert band_a == band_b == "very_low"
    assert mood_a != mood_b


def test_agent_profile_does_not_include_trust_settings(cast):
    assert set(cast.personas["A"].profile()) == {"role", "background", "temperament"}


def test_first_speaker_plays_a_and_the_other_plays_b(seed):
    assert assign_personas(seed) == {"alice": "A", "bob": "B"}


def test_conversation_without_exactly_two_speakers_is_rejected():
    seed = Seed("solo", (Turn("x", "one"), Turn("x", "two")))
    with pytest.raises(ValueError, match="exactly 2"):
        assign_personas(seed)


def test_missing_mood_for_a_band_is_rejected(tmp_path):
    path = write_variant(tmp_path, lambda raw: raw["personas"]["B"]["moods"].pop("low"))
    with pytest.raises(ConfigError, match="missing"):
        load_cast(path)


def test_bands_must_reach_zero(tmp_path):
    path = write_variant(tmp_path, lambda raw: raw["mood_bands"][-1].update(min=0.05))
    with pytest.raises(ConfigError, match="start at 0"):
        load_cast(path)


def test_bands_must_be_listed_highest_first(tmp_path):
    path = write_variant(tmp_path, lambda raw: raw["mood_bands"].reverse())
    with pytest.raises(ConfigError, match="highest"):
        load_cast(path)


def test_zero_alpha_is_rejected(tmp_path):
    path = write_variant(tmp_path, lambda raw: raw["personas"]["A"].update(alpha_up=0))
    with pytest.raises(ConfigError, match="above 0"):
        load_cast(path)


def test_unknown_persona_attribute_is_rejected(tmp_path):
    path = write_variant(tmp_path, lambda raw: raw["personas"]["A"].update(mood="angry"))
    with pytest.raises(ConfigError, match="unknown"):
        load_cast(path)
