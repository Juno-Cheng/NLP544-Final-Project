"""Offline production-simulator handoffs using real components and canned model replies."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.escalation_judge import score_escalation
from src.evaluate import evaluate_conditions, to_schema_summary
from src.llm_client import MockLLMClient
from src.persona import assign_personas, initialize_personas, load_persona_configs
from src.response_generator import PersonaResponseGenerator
from src.schemas import (
    ConversationSeed,
    EscalationScore,
    EvaluationSummary,
    GenerationConfig,
    JudgeConfig,
    LabeledExample,
    RunMetadata,
    SimulationConfig,
    SimulationResult,
    UpdaterConfig,
    from_dict,
    to_dict,
)
from src.simulator import run_simulation
from src.state_updater import TrustUpdater


@pytest.fixture
def seed():
    path = Path(__file__).parent / "fixtures" / "sample_seed.json"
    original = from_dict(ConversationSeed, json.loads(path.read_text(encoding="utf-8")))
    names = {"A": "EditorOne", "B": "EditorTwo"}
    return replace(
        original,
        opening_turns=[
            replace(turn, speaker_id=names[turn.speaker_id])
            for turn in original.opening_turns
        ],
    )


def paired_components(seed):
    """Exercise four turns per arm through the production simulator."""
    personas = assign_personas(seed, load_persona_configs())
    config = GenerationConfig("mock", 0.3, 256, "reply_v2", 544)
    results, traces = [], {}
    for condition in ("static", "dynamic"):
        reply_client = MockLLMClient(["Please explain the edit."] * 4)
        rater_client = MockLLMClient(["0", "0.25", "0.5", "1"])
        generation_events, rating_events = [], []
        generator = PersonaResponseGenerator(
            reply_client, on_generation=generation_events.append
        )
        updater = TrustUpdater(
            rater_client, config.model_id, on_rating=rating_events.append
        )
        metadata = RunMetadata(
            experiment_id="component-test",
            paired_run_id=seed.conversation_id,
            model_id=config.model_id,
            prompt_version=config.prompt_version,
            updater_version=updater.version,
            max_generated_turns=4,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
            random_seed=config.random_seed,
        )
        results.append(
            run_simulation(
                seed,
                personas,
                SimulationConfig(
                    condition,
                    config.model_id,
                    4,
                    config.temperature,
                    config.max_output_tokens,
                    config.random_seed,
                ),
                generator,
                updater,
                run_metadata=metadata,
                run_id=f"{seed.conversation_id}-{condition}",
            )
        )
        assert results[-1].status == "ok", results[-1].run_metadata.error
        traces[condition] = (
            reply_client.requests,
            rater_client.requests,
            generation_events,
            rating_events,
        )
    return results, traces


def test_paired_trust_updates_reach_next_receiver_reply(seed):
    original = to_dict(seed)
    (static, dynamic), traces = paired_components(seed)
    assert to_dict(seed) == original
    assert static.state_history[0] == dynamic.state_history[0]
    assert all(
        state.trust == 0.75 for state in static.state_history[0].speaker_states.values()
    )
    assert all(
        state.trust == 0.75
        for snapshot in static.state_history
        for state in snapshot.speaker_states.values()
    )
    assert [t.speaker_id for t in dynamic.generated_turns] == [
        "EditorOne",
        "EditorTwo",
    ] * 2
    assert [t.turn_id for t in dynamic.generated_turns] == [2, 3, 4, 5]
    # At each step only the receiver can change. These are post-update snapshots.
    for before, after, turn in zip(
        dynamic.state_history, dynamic.state_history[1:], dynamic.generated_turns
    ):
        assert (
            before.speaker_states[turn.speaker_id]
            == after.speaker_states[turn.speaker_id]
        )
    assert dynamic.state_history[1].speaker_states["EditorTwo"].trust == 0.5
    assert dynamic.state_history[2].speaker_states["EditorOne"].trust == 0.625
    assert dynamic.state_history[-1].speaker_states["EditorOne"].trust == 0.875
    payloads = [json.loads(request["payload"]) for request in traces["dynamic"][0]]
    assert [p["actor"]["trust"] for p in payloads] == [0.75, 0.5, 0.625, 0.5]
    assert [p["trust_level"]["level"] for p in payloads] == [
        "open",
        "cautious",
        "reserved",
        "cautious",
    ]
    assert traces["static"][1] == []  # No rater calls for static or the opening.
    assert len(traces["dynamic"][1]) == 4
    for index, request in enumerate(traces["dynamic"][1]):
        payload = json.loads(request["payload"])
        assert set(payload) == {"reader_id", "history"}
        assert len(payload["history"]) == len(seed.opening_turns) + index + 1
        assert payload["reader_id"] != payload["history"][-1]["speaker_id"]
    for index in range(4):
        assert (
            traces["static"][0][index]["config"]
            == traces["dynamic"][0][index]["config"]
        )
        assert (
            traces["static"][0][index]["instructions"]
            == traces["dynamic"][0][index]["instructions"]
        )
    for result in (static, dynamic):
        assert (
            from_dict(SimulationResult, json.loads(json.dumps(to_dict(result))))
            == result
        )
        assert result.run_metadata.max_output_tokens == 256


def test_new_results_judge_and_evaluation_schema_handoff(seed):
    results, traces = paired_components(seed)
    judge_inputs = []

    def judge(prompt):
        judge_inputs.append(prompt)
        return '{"score": 0.25}'

    scores = [
        from_dict(
            EscalationScore,
            score_escalation(
                result.run_id,
                result.transcript,
                JudgeConfig("mock-judge", "v1", judge_callable=judge),
            ),
        )
        for result in results
    ]
    for prompt in judge_inputs:
        dialogue = json.loads(
            prompt.split("<TRANSCRIPT_JSON>\n")[1].split("\n</TRANSCRIPT_JSON>")[0]
        )
        assert all(set(turn) == {"speaker_id", "text"} for turn in dialogue)
        assert len(dialogue) == 6
    # The label only enters here, after generation, state updates and judging.
    example = LabeledExample(seed, 0, "synthetic label for component test", "train")
    summary = to_schema_summary(evaluate_conditions([example], results, scores))
    assert summary.coverage.n_completed_pairs == 1
    assert summary.excluded == []
    assert summary.unverified_pair_settings == []
    assert (
        summary.metrics.roc_auc_static is None
    )  # One synthetic example is not evidence.
    assert (
        from_dict(EvaluationSummary, json.loads(json.dumps(to_dict(summary))))
        == summary
    )
    for replies, _, events, _ in traces.values():
        assert len(events) == len(replies) == 4
        assert all(event["prompt_version"] == "reply_v2" for event in events)
        assert all(
            "trust_level" not in event and "history" not in event for event in events
        )


@pytest.mark.parametrize("rating", ["not a score", "nan", "2"])
def test_invalid_rating_propagates_without_changing_receiver(seed, rating):
    states = initialize_personas(assign_personas(seed, load_persona_configs()))
    reply_client = MockLLMClient(["Please explain."])
    generator = PersonaResponseGenerator(reply_client)
    turn = generator(
        seed.opening_turns,
        states["EditorOne"],
        GenerationConfig("mock", 0.3, 256, "reply_v2", 544),
    )
    history = [*seed.opening_turns, turn]
    logged = []
    updater = TrustUpdater(MockLLMClient([rating]), "mock", on_rating=logged.append)
    with pytest.raises(ValueError):
        updater(
            states["EditorTwo"],
            turn,
            history,
            UpdaterConfig("dynamic", updater.version),
        )
    assert states["EditorTwo"].trust == 0.75
    assert logged[0]["raw_text"] == rating
    assert len(reply_client.requests) == 1
