"""Tests for src/persona.py and src/state_updater.py.

Uses a tiny inline fake client so the tests run offline before src/llm_client.py
(Arya's MockLLMClient) is merged; swap to MockLLMClient once it is on main.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.persona import assign_personas, describe_trust, initialize_personas, load_persona_configs
from src.persona import load_config as load_persona_file  # name clash with state_updater.load_config
from src.schemas import ConversationSeed, DialogueTurn, UpdaterConfig, from_dict
from src.state_updater import TrustUpdater, load_config, sensitivity_by_speaker

FIXTURES = Path(__file__).parent / "fixtures"
STATIC = UpdaterConfig("static", "trust_v4")
DYNAMIC = UpdaterConfig("dynamic", "trust_v4")


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, instructions, payload, config):
        self.requests.append({"instructions": instructions, "payload": payload, "config": config})
        return SimpleNamespace(text=self.replies.pop(0), metadata={"provider": "fake", "token_usage": {"total": 7}})


@pytest.fixture
def seed():
    return from_dict(ConversationSeed, json.loads((FIXTURES / "sample_seed.json").read_text(encoding="utf-8")))


@pytest.fixture
def states():
    return initialize_personas(assign_personas(_seed_with("A", "B"), load_persona_configs()))


def _seed_with(*speakers):
    turns = [{"turn_id": i, "speaker_id": s, "text": f"comment {i}"} for i, s in enumerate(speakers)]
    return from_dict(ConversationSeed, {"conversation_id": "c1", "source": "cga-wiki", "opening_turns": turns})


def _updater(*replies, alpha_up=0.25, alpha_down=0.5, sensitivity=None, on_rating=None):
    settings = {**load_config(), "alpha_up": alpha_up, "alpha_down": alpha_down}
    return TrustUpdater(FakeClient(*replies), "mock", settings, sensitivity=sensitivity, on_rating=on_rating)


def test_persona_config_defines_both_roles_with_same_traits_and_different_sensitivity():
    by_role = load_persona_configs()
    assert set(by_role) == {"A", "B"}
    assert by_role["A"].stable_traits == by_role["B"].stable_traits
    assert by_role["A"].initial_trust == by_role["B"].initial_trust
    assert by_role["A"].trust_sensitivity != by_role["B"].trust_sensitivity
    for persona in by_role.values():
        assert persona.trust_sensitivity["alpha_down"] > persona.trust_sensitivity["alpha_up"]


def test_mood_comes_from_trust_bands_not_fixed_traits():
    traits = load_persona_configs()["A"].stable_traits
    assert "temperament" not in traits
    assert describe_trust(0.75)["behavior"].startswith("Courteous")
    assert describe_trust(0.3)["behavior"].startswith("Irritated")


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
    [
        (1.0, "trusting"),
        (0.9, "trusting"),
        (0.89, "cordial"),
        (0.75, "open"),
        (0.5, "cautious"),
        (0.49, "wary"),
        (0.1, "adversarial"),
        (0.09, "accusatory"),
        (0.0, "accusatory"),
    ],
)
def test_describe_trust_picks_band_at_boundaries(trust, level):
    described = describe_trust(trust)
    assert described["level"] == level
    assert described["behavior"].strip()


def test_ten_distinct_levels_and_start_is_open():
    levels = load_persona_file()["trust_levels"]
    assert len(levels) == 10
    assert len({band["level"] for band in levels}) == 10
    assert describe_trust(load_persona_configs()["A"].initial_trust)["level"] == "open"


@pytest.mark.parametrize("levels", [[{"min": 0.5, "level": "high", "behavior": "b"}], []])
def test_describe_trust_rejects_bands_that_miss_zero(levels):
    with pytest.raises(ValueError, match="lowest min"):
        describe_trust(0.1, levels)


def test_static_trust_never_changes_and_skips_rater(seed, states):
    updater = _updater()
    history = list(seed.opening_turns)
    result = updater(states["A"], history[-1], history, STATIC)
    assert result.state == states["A"]
    assert result.previous_trust == states["A"].trust
    assert updater.client.requests == []


def test_dynamic_trust_shifts_by_rating_around_neutral(seed, states):
    history = list(seed.opening_turns)
    hostile = _updater("0").update_state(states["A"], history[-1], history, DYNAMIC)
    curt = _updater("0.25").update_state(states["A"], history[-1], history, DYNAMIC)
    warm = _updater("1").update_state(states["A"], history[-1], history, DYNAMIC)
    assert hostile.state.trust == pytest.approx(0.5)
    assert curt.state.trust == pytest.approx(0.625)
    assert warm.state.trust == pytest.approx(0.875)
    assert hostile.previous_trust == 0.75
    assert states["A"].trust == 0.75


def test_trust_falls_faster_than_it_recovers(seed, states):
    history = list(seed.opening_turns)
    drop = 0.75 - _updater("0").update_state(states["A"], history[-1], history, DYNAMIC).state.trust
    rise = _updater("1").update_state(states["A"], history[-1], history, DYNAMIC).state.trust - 0.75
    assert drop == pytest.approx(0.25)
    assert rise == pytest.approx(0.125)


def test_low_trust_recovers_after_cooperative_messages(seed, states):
    history = list(seed.opening_turns)
    updater = _updater("0", "0", "1", "1", "1", "1")
    state = states["A"]
    trajectory = []
    for _ in range(6):
        state = updater.update_state(state, history[-1], history, DYNAMIC).state
        trajectory.append(state.trust)
    assert trajectory == pytest.approx([0.5, 0.25, 0.375, 0.5, 0.625, 0.75])


def test_neutral_rating_leaves_trust_unchanged(seed, states):
    history = list(seed.opening_turns)
    assert _updater("0.5").update_state(states["A"], history[-1], history, DYNAMIC).state.trust == 0.75


def test_on_rating_reports_usage_for_cost_logging(seed, states):
    history = list(seed.opening_turns)
    logged = []
    _updater("0.3", on_rating=logged.append).update_state(states["A"], history[-1], history, DYNAMIC)
    assert logged == [
        {
            "provider": "fake",
            "token_usage": {"total": 7},
            "turn_id": history[-1].turn_id,
            "reader_id": "A",
            "raw_text": "0.3",
            "prompt_version": "trust_update_v2",
        }
    ]


def test_on_rating_reports_failed_ratings_too(seed, states):
    history = list(seed.opening_turns)
    logged = []
    with pytest.raises(ValueError):
        _updater("Rating: 0.4", on_rating=logged.append).update_state(states["A"], history[-1], history, DYNAMIC)
    assert logged[0]["raw_text"] == "Rating: 0.4"
    assert logged[0]["token_usage"] == {"total": 7}


def test_trust_lands_on_band_edge_despite_float_error(seed, states):
    history = list(seed.opening_turns)
    state = states["A"]
    for rating in ("0.15", "0.15"):
        state = _updater(rating).update_state(state, history[-1], history, DYNAMIC).state
    assert state.trust == 0.4
    assert describe_trust(state.trust)["level"] == "wary"


@pytest.mark.parametrize("name", ["alpha_up", "alpha_down"])
@pytest.mark.parametrize("alpha", [0.0, -0.5, float("nan"), 1.5])
def test_bad_alpha_fails_at_construction(name, alpha):
    with pytest.raises(ValueError, match=name):
        _updater(**{name: alpha})


def test_dynamic_trust_stays_bounded_over_many_turns(seed, states):
    history = list(seed.opening_turns)
    updater = _updater(*(["1", "0"] * 20), alpha_up=1.0, alpha_down=1.0)
    state = states["A"]
    for _ in range(40):
        state = updater(state, history[-1], history, DYNAMIC).state
        assert 0.0 <= state.trust <= 1.0


def test_rater_sees_dialogue_but_not_trust(seed, states):
    history = list(seed.opening_turns)
    updater = _updater("0.5")
    updater(states["A"], history[-1], history, DYNAMIC)
    payload = json.loads(updater.client.requests[0]["payload"])
    assert set(payload) == {"reader_id", "history"}
    assert "trust" not in updater.client.requests[0]["payload"]
    assert payload["history"][-1]["speaker_id"] == "B"


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


def test_each_speaker_moves_by_its_own_sensitivity(seed, states):
    history = list(seed.opening_turns)
    sensitivity = sensitivity_by_speaker(assign_personas(_seed_with("A", "B"), load_persona_configs()))
    observed_by_b = DialogueTurn(9, "A", "a later comment")
    a = _updater("0", sensitivity=sensitivity).update_state(states["A"], history[-1], history, DYNAMIC)
    b = _updater("0", sensitivity=sensitivity).update_state(
        states["B"], observed_by_b, history + [observed_by_b], DYNAMIC
    )
    assert a.state.trust == pytest.approx(0.75 - 0.4 * 0.5)
    assert b.state.trust == pytest.approx(0.75 - 0.6 * 0.5)


def test_speaker_without_sensitivity_uses_defaults(seed, states):
    history = list(seed.opening_turns)
    updater = _updater("0", sensitivity={"A": None, "B": {"alpha_up": 0.1, "alpha_down": 0.1}})
    assert updater.update_state(states["A"], history[-1], history, DYNAMIC).state.trust == pytest.approx(0.5)


def test_unknown_speaker_is_an_error_once_sensitivity_is_set(seed, states):
    history = list(seed.opening_turns)
    with pytest.raises(ValueError, match="no trust sensitivity"):
        _updater("0", sensitivity={"B": None}).update_state(states["A"], history[-1], history, DYNAMIC)


def test_bad_persona_sensitivity_fails_at_construction():
    with pytest.raises(ValueError, match="A alpha_down"):
        _updater(sensitivity={"A": {"alpha_up": 0.2, "alpha_down": 0.0}})


def test_replay_opening_rates_each_turn_from_the_receiver_side():
    seed = _seed_with("A", "B", "A")
    updater = _updater("0", "1", "0", sensitivity={"A": None, "B": None})
    start = initialize_personas(assign_personas(seed, load_persona_configs()))
    after = updater.replay_opening(start, seed.opening_turns)
    # B reads turns 0 and 2 (both 0), A reads turn 1 (rating 1).
    assert after["B"].trust == pytest.approx(0.25)
    assert after["A"].trust == pytest.approx(0.875)
    assert start["A"].trust == start["B"].trust == 0.75
    payloads = [json.loads(request["payload"]) for request in updater.client.requests]
    assert [p["reader_id"] for p in payloads] == ["B", "A", "B"]
    assert [len(p["history"]) for p in payloads] == [1, 2, 3]


def test_replay_opening_is_on_by_default():
    assert TrustUpdater(FakeClient(), "mock").replay_opening_enabled


def test_default_config_builds_rater_call():
    updater = TrustUpdater(FakeClient(), "mock")
    assert updater.version == "trust_v4"
    assert updater.alpha_up == 0.25
    assert updater.alpha_down == 0.5
    assert updater.rater_config.prompt_version == "trust_update_v2"
    assert updater.instructions.strip()
