"""Offline generator contract tests using shared schema fixtures."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.llm_client import GenerationError, MockLLMClient
from src.persona import describe_trust
from src.response_generator import PersonaResponseGenerator
from src.schemas import (
    AgentState,
    ConversationSeed,
    DialogueTurn,
    GenerationConfig,
    from_dict,
)


@pytest.fixture
def history():
    path = Path(__file__).parent / "fixtures" / "sample_seed.json"
    return from_dict(ConversationSeed, json.loads(path.read_text())).opening_turns


@pytest.fixture(params=["reply_v1", "reply_v2"])
def config(request):
    return GenerationConfig("mock", 0.3, 256, request.param, 544)


@pytest.mark.parametrize("speaker", ["A", "B"])
def test_reply_uses_actor_and_next_turn_id(history, config, speaker):
    client = MockLLMClient(["  Please explain the edit.  "])
    actor = AgentState(speaker, 0.7, {"role": "editor"})
    reply = PersonaResponseGenerator(client)(history, actor, config)
    assert reply == DialogueTurn(
        history[-1].turn_id + 1, speaker, "Please explain the edit."
    )
    payload = json.loads(client.requests[0]["payload"])
    assert payload["actor"] == {
        "speaker_id": speaker,
        "trust": 0.7,
        "stable_traits": {"role": "editor"},
    }
    assert [turn["text"] for turn in payload["history"]] == [
        turn.text for turn in history
    ]
    expected_keys = {"actor", "history"}
    if config.prompt_version == "reply_v2":
        expected_keys.add("trust_level")
        assert payload["trust_level"] == describe_trust(actor.trust)
    assert set(payload) == expected_keys


def test_updated_trust_reaches_later_reply_without_mutation(history, config):
    actor = AgentState("A", 0.7, {"goals": ["accuracy"]})
    original = deepcopy((history, actor))
    client = MockLLMClient(["First reply", "Second reply"])
    generator = PersonaResponseGenerator(client)
    first = generator(history, actor, config)
    generator(history + [first], replace(actor, trust=0.2), config)
    requests = [json.loads(request["payload"]) for request in client.requests]
    assert [request["actor"]["trust"] for request in requests] == [0.7, 0.2]
    assert requests[1]["history"][-1]["text"] == "First reply"
    assert (history, actor) == original


def test_identical_inputs_build_identical_requests(history, config):
    client = MockLLMClient(["One", "Two"])
    generator = PersonaResponseGenerator(client)
    for _ in range(2):
        generator(history, AgentState("A", 0.7, {}), config)
    assert client.requests[0] == client.requests[1]
    assert not hasattr(config, "condition")


def test_metadata_callback_is_separate_from_dialogue(history, config):
    events = []
    reply = PersonaResponseGenerator(
        MockLLMClient(["Reply"]), on_generation=events.append
    )(history, AgentState("A", 0.7, {}), config)
    assert reply.text == "Reply"
    assert events[0]["turn_id"] == reply.turn_id
    assert events[0]["provider"] == "mock"
    assert len(events[0]["prompt_sha256"]) == 64
    assert "history" not in events[0]
    assert "trust" not in events[0]
    assert "trust_level" not in events[0]


@pytest.mark.parametrize("text", ["", " \n ", None])
def test_empty_or_invalid_output_fails(history, config, text):
    with pytest.raises(GenerationError):
        PersonaResponseGenerator(MockLLMClient([text]))(
            history, AgentState("A", 0.7, {}), config
        )


@pytest.mark.parametrize(
    "kind", ["empty", "duplicate", "reversed", "unknown_actor", "blank"]
)
def test_bad_history_fails_before_call(history, config, kind):
    actor = AgentState("A", 0.7, {})
    if kind == "empty":
        history = []
    elif kind == "duplicate":
        history = history + [history[0]]
    elif kind == "reversed":
        history = list(reversed(history))
    elif kind == "unknown_actor":
        actor = replace(actor, speaker_id="C")
    else:
        history = [replace(history[0], text=" "), *history[1:]]
    client = MockLLMClient(["unused"])
    with pytest.raises(ValueError):
        PersonaResponseGenerator(client)(history, actor, config)
    assert client.requests == []


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), 2.1])
def test_invalid_temperature_fails_before_call(history, config, temperature):
    with pytest.raises(ValueError, match="temperature"):
        PersonaResponseGenerator(MockLLMClient([]))(
            history, AgentState("A", 0.7, {}), replace(config, temperature=temperature)
        )


def test_strict_parser_rejects_boolean_trust(history, config):
    with pytest.raises(ValueError):
        PersonaResponseGenerator(MockLLMClient([]))(
            history, AgentState("A", True, {}), config
        )


@pytest.mark.parametrize("version", ["../reply_v1", "judge_rubric_v1", "reply_v0"])
def test_prompt_path_is_restricted(history, config, version):
    with pytest.raises(ValueError, match="prompt_version"):
        PersonaResponseGenerator(MockLLMClient([]))(
            history, AgentState("A", 0.7, {}), replace(config, prompt_version=version)
        )


def test_missing_prompt_fails_without_network(history, config):
    with pytest.raises(FileNotFoundError):
        PersonaResponseGenerator(MockLLMClient([]))(
            history,
            AgentState("A", 0.7, {}),
            replace(config, prompt_version="reply_v999"),
        )


def test_mock_exhaustion_propagates(history, config):
    with pytest.raises(GenerationError, match="exhausted"):
        PersonaResponseGenerator(MockLLMClient([]))(
            history, AgentState("A", 0.7, {}), config
        )


@pytest.mark.parametrize(
    ("trust", "level"),
    [
        (0.0, "accusatory"),
        (0.099999, "accusatory"),
        (0.1, "adversarial"),
        (0.2, "distrustful"),
        (0.3, "skeptical"),
        (0.4, "wary"),
        (0.5, "cautious"),
        (0.6, "reserved"),
        (0.7, "open"),
        (0.75, "open"),
        (0.8, "cordial"),
        (0.9, "trusting"),
        (1.0, "trusting"),
    ],
)
def test_v2_passes_all_trust_bands_and_numeric_value(history, trust, level):
    client = MockLLMClient(["A reply"])
    config = GenerationConfig("mock", 0.3, 256, "reply_v2", 544)
    PersonaResponseGenerator(client)(history, AgentState("A", trust, {}), config)
    payload = json.loads(client.requests[0]["payload"])
    assert payload["actor"]["trust"] == trust
    assert payload["trust_level"]["level"] == level
    assert payload["trust_level"]["behavior"] == describe_trust(trust)["behavior"]
    assert "trust_level.behavior" in client.requests[0]["instructions"]


def test_v1_does_not_load_trust_bands(history, monkeypatch):
    def unexpected_lookup(*args):
        pytest.fail("Legacy v1 must not depend on the v2 band configuration")

    monkeypatch.setattr("src.response_generator.describe_trust", unexpected_lookup)
    client = MockLLMClient(["Reply"])
    PersonaResponseGenerator(client)(
        history,
        AgentState("A", 0.75, {}),
        GenerationConfig("mock", 0.3, 256, "reply_v1", 544),
    )
    assert set(json.loads(client.requests[0]["payload"])) == {"actor", "history"}


def test_v2_recomputes_band_on_each_call_and_keeps_within_band_changes(history):
    client = MockLLMClient(["Reply"] * 3)
    generator = PersonaResponseGenerator(client)
    config = GenerationConfig("mock", 0.3, 256, "reply_v2", 544)
    for trust in (0.75, 0.71, 0.5):
        generator(history, AgentState("A", trust, {}), config)
    payloads = [json.loads(request["payload"]) for request in client.requests]
    assert [payload["trust_level"]["level"] for payload in payloads] == [
        "open",
        "open",
        "cautious",
    ]
    assert [payload["actor"]["trust"] for payload in payloads] == [0.75, 0.71, 0.5]
