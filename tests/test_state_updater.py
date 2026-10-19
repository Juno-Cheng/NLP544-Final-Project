"""Tests for src/persona.py and src/state_updater.py.

Uses a tiny inline fake client so the tests run offline before src/llm_client.py
(Arya's MockLLMClient) is merged; swap to MockLLMClient once it is on main.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.persona import assign_personas, describe_trust, initialize_personas, load_persona_configs
from src.schemas import ConversationSeed, DialogueTurn, UpdaterConfig, from_dict
from src.state_updater import TrustUpdater, load_config

FIXTURES = Path(__file__).parent / "fixtures"
STATIC = UpdaterConfig("static", "trust_v1")
DYNAMIC = UpdaterConfig("dynamic", "trust_v1")


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, instructions, payload, config):
        self.requests.append({"instructions": instructions, "payload": payload, "config": config})
        return SimpleNamespace(text=self.replies.pop(0))


@pytest.fixture
def seed():
    return from_dict(ConversationSeed, json.loads((FIXTURES / "sample_seed.json").read_text(encoding="utf-8")))


@pytest.fixture
def states():
    return initialize_personas(assign_personas(_seed_with("A", "B"), load_persona_configs()))


def _seed_with(*speakers):
    turns = [{"turn_id": i, "speaker_id": s, "text": f"comment {i}"} for i, s in enumerate(speakers)]
    return from_dict(ConversationSeed, {"conversation_id": "c1", "source": "cga-wiki", "opening_turns": turns})


def _updater(*replies, alpha=0.3):
    return TrustUpdater(FakeClient(*replies), "mock", {**load_config(), "alpha": alpha})


def test_persona_config_defines_both_roles_with_same_traits():
    by_role = load_persona_configs()
    assert set(by_role) == {"A", "B"}
    assert by_role["A"].stable_traits == by_role["B"].stable_traits
    assert by_role["A"].initial_trust == by_role["B"].initial_trust


def test_assign_personas_maps_real_usernames_by_first_appearance():
    by_role = load_persona_configs()
    personas = assign_personas(_seed_with("Sirex98", "Jayjg", "Sirex98"), by_role)
    assert {sid: cfg.persona_id for sid, cfg in personas.items()} == {"Sirex98": "A", "Jayjg": "B"}


@pytest.mark.parametrize("speakers", [("A", "A"), ("A", "B", "C")])
def test_assign_personas_rejects_non_two_speaker_openings(speakers):
    with pytest.raises(ValueError, match="exactly 2"):
        assign_personas(_seed_with(*speakers), load_persona_configs())


def test_initialize_personas_gives_each_arm_independent_copies(seed):
    personas = assign_personas(seed, load_persona_configs())
    static_arm, dynamic_arm = initialize_personas(personas), initialize_personas(personas)
    assert static_arm == dynamic_arm
    assert all(state.trust == personas[sid].initial_trust for sid, state in static_arm.items())
    static_arm["A"].stable_traits["role"] = "mutated"
    assert dynamic_arm["A"].stable_traits["role"] != "mutated"
    assert personas["A"].stable_traits["role"] != "mutated"


@pytest.mark.parametrize(
    ("trust", "level"),
    [(1.0, "high"), (0.67, "high"), (0.66, "guarded"), (0.34, "guarded"), (0.33, "low"), (0.0, "low")],
)
def test_describe_trust_picks_band_at_boundaries(trust, level):
    described = describe_trust(trust)
    assert described["level"] == level
    assert described["behavior"].strip()


def test_describe_trust_rejects_bands_that_miss_zero():
    with pytest.raises(ValueError, match="lowest min"):
        describe_trust(0.1, [{"min": 0.5, "level": "high", "behavior": "b"}])


def test_static_trust_never_changes_and_skips_rater(seed, states):
    updater = _updater()
    history = list(seed.opening_turns)
    result = updater(states["A"], history[-1], history, STATIC)
    assert result.state == states["A"]
    assert result.previous_trust == states["A"].trust
    assert updater.client.requests == []


def test_dynamic_trust_moves_toward_rating(seed, states):
    history = list(seed.opening_turns)
    hostile = _updater("0").update_state(states["A"], history[-1], history, DYNAMIC)
    friendly = _updater("1").update_state(states["A"], history[-1], history, DYNAMIC)
    assert hostile.state.trust == pytest.approx(0.7 * 0.7)
    assert friendly.state.trust == pytest.approx(0.7 * 0.7 + 0.3)
    assert hostile.previous_trust == 0.7
    assert states["A"].trust == 0.7


def test_dynamic_trust_stays_bounded_over_many_turns(seed, states):
    history = list(seed.opening_turns)
    updater = _updater(*(["1", "0"] * 20), alpha=1.0)
    state = states["A"]
    for _ in range(40):
        state = updater(state, history[-1], history, DYNAMIC).state
        assert 0.0 <= state.trust <= 1.0


def test_rater_sees_dialogue_but_not_trust(seed, states):
    history = list(seed.opening_turns)
    updater = _updater("0.5")
    updater(states["A"], history[-1], history, DYNAMIC)
    payload = json.loads(updater.client.requests[0]["payload"])
    assert set(payload) == {"reader_id", "history", "observed_turn"}
    assert "trust" not in updater.client.requests[0]["payload"]
    assert payload["observed_turn"]["speaker_id"] == "B"


def test_rejects_update_from_own_turn(states):
    own = DialogueTurn(0, "A", "my own comment")
    with pytest.raises(ValueError, match="own turn"):
        _updater().update_state(states["A"], own, [own], DYNAMIC)


def test_rejects_observed_turn_that_is_not_latest(seed, states):
    history = list(seed.opening_turns) + [DialogueTurn(2, "A", "later")]
    with pytest.raises(ValueError, match="latest"):
        _updater().update_state(states["A"], history[1], history, STATIC)


@pytest.mark.parametrize("reply", ["high", "1.5", "-0.1", "nan", ""])
def test_bad_rating_raises_instead_of_defaulting(seed, states, reply):
    history = list(seed.opening_turns)
    with pytest.raises(ValueError, match="rat"):
        _updater(reply).update_state(states["A"], history[-1], history, DYNAMIC)


def test_updater_version_mismatch_raises(seed, states):
    history = list(seed.opening_turns)
    with pytest.raises(ValueError, match="trust_v9"):
        _updater().update_state(states["A"], history[-1], history, UpdaterConfig("dynamic", "trust_v9"))


def test_default_config_builds_rater_call():
    updater = TrustUpdater(FakeClient(), "mock")
    assert updater.rater_config.prompt_version == "trust_update_v1"
    assert updater.instructions.strip()
