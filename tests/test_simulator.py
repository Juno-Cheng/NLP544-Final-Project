"""Offline simulator tests using shared fixtures and Magnus's actual trust updater."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.escalation_judge import score_escalation
from src.evaluate import evaluate_conditions
from src.persona import assign_personas, load_persona_configs
from src.schemas import (
    ConversationSeed,
    DialogueTurn,
    GenerationConfig,
    LabeledExample,
    RunMetadata,
    SimulationConfig,
    SimulationResult,
    StateUpdateResult,
    from_dict,
    to_dict,
)
from src.simulator import run_simulation
from src.state_updater import TrustUpdater

FIXTURES = Path(__file__).parent / "fixtures"


class RecordingGenerator:
    """Record the visible inputs and return one deterministic reply."""

    def __init__(self):
        self.calls = []

    def __call__(self, history, actor, config):
        self.calls.append((deepcopy(history), deepcopy(actor), config))
        return DialogueTurn(history[-1].turn_id + 1, actor.speaker_id, f"Reply {len(self.calls)}")


class FakeClient:
    """Supply rater completions without the unfinished live model client."""

    def __init__(self, rating="0"):
        self.rating = rating
        self.requests = []

    def complete(self, instructions, payload, config):
        self.requests.append(json.loads(payload))
        return SimpleNamespace(text=self.rating, metadata={})


@pytest.fixture
def simulation_inputs():
    seed = from_dict(
        ConversationSeed, json.loads((FIXTURES / "sample_seed.json").read_text(encoding="utf-8"))
    )
    config = SimulationConfig("dynamic", "mock", 6, 0.0, 256, 544)
    updater = TrustUpdater(FakeClient(), config.model_id)
    return {
        "seed": seed,
        "personas": assign_personas(seed, load_persona_configs()),
        "config": config,
        "generator": RecordingGenerator(),
        "updater": updater,
        "run_metadata": RunMetadata(
            experiment_id="experiment-test",
            paired_run_id="pair-example_001",
            model_id=config.model_id,
            prompt_version="reply_v2",
            updater_version=updater.version,
            max_generated_turns=config.max_generated_turns,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
            random_seed=config.random_seed,
        ),
    }


def test_alternates_after_last_opening_speaker_and_preserves_opening(simulation_inputs):
    result = run_simulation(**simulation_inputs)
    seed = simulation_inputs["seed"]
    assert result.status == "ok"
    assert result.conversation_id == seed.conversation_id
    assert result.transcript[:2] == seed.opening_turns
    assert result.transcript is not seed.opening_turns
    assert [turn.speaker_id for turn in result.generated_turns] == ["A", "B"] * 3
    assert [turn.turn_id for turn in result.transcript] == list(range(8))
    assert result.generated_turns == result.transcript[2:]


def test_dynamic_receiver_update_reaches_the_next_generation(simulation_inputs):
    result = run_simulation(**simulation_inputs)
    calls = simulation_inputs["generator"].calls
    assert [actor.trust for _, actor, _ in calls] == [0.75, 0.5, 0.5, 0.25, 0.25, 0.0]
    assert [len(history) for history, _, _ in calls] == list(range(2, 8))
    for index, (history, _, _) in enumerate(calls):
        assert history == result.transcript[:2 + index]
    requests = simulation_inputs["updater"].client.requests
    assert [request["reader_id"] for request in requests] == ["B", "A"] * 3
    assert [len(request["history"]) for request in requests] == list(range(3, 9))
    assert result.state_history[1].speaker_states["A"].trust == 0.75
    assert result.state_history[1].speaker_states["B"].trust == 0.5


def test_static_trust_stays_fixed_without_any_rater_calls(simulation_inputs):
    simulation_inputs["config"] = replace(simulation_inputs["config"], condition="static")
    result = run_simulation(**simulation_inputs)
    assert result.status == "ok"
    assert result.condition == "static"
    assert simulation_inputs["updater"].client.requests == []
    assert all(actor.trust == 0.75 for _, actor, _ in simulation_inputs["generator"].calls)
    assert all(
        state.trust == 0.75
        for snapshot in result.state_history
        for state in snapshot.speaker_states.values()
    )


def test_neutral_dynamic_messages_can_leave_trust_unchanged(simulation_inputs):
    simulation_inputs["updater"].client.rating = "0.5"
    result = run_simulation(**simulation_inputs)
    assert result.status == "ok"
    assert len(simulation_inputs["updater"].client.requests) == 6
    assert all(actor.trust == 0.75 for _, actor, _ in simulation_inputs["generator"].calls)


def test_records_initial_state_and_independent_post_update_snapshots(simulation_inputs):
    result = run_simulation(**simulation_inputs)
    assert [snapshot.turn_id for snapshot in result.state_history] == list(range(1, 8))
    assert all(state.trust == 0.75 for state in result.state_history[0].speaker_states.values())
    for snapshot in result.state_history:
        assert set(snapshot.speaker_states) == set(simulation_inputs["personas"])
        for speaker_id, state in snapshot.speaker_states.items():
            assert state.stable_traits == simulation_inputs["personas"][speaker_id].stable_traits
    result.state_history[-1].speaker_states["A"].stable_traits["role"] = "changed after run"
    assert result.state_history[0].speaker_states["A"].stable_traits["role"] == "Wikipedia editor"
    assert simulation_inputs["personas"]["A"].stable_traits["role"] == "Wikipedia editor"


def test_runs_start_fresh_and_keep_paired_generation_settings(simulation_inputs):
    static = run_simulation(
        **{**simulation_inputs, "config": replace(simulation_inputs["config"], condition="static")}
    )
    dynamic = run_simulation(**simulation_inputs)
    assert static.run_id != dynamic.run_id
    assert static.state_history[0] == dynamic.state_history[0]
    assert static.run_metadata.paired_run_id == dynamic.run_metadata.paired_run_id
    calls = simulation_inputs["generator"].calls
    assert calls[0][1].trust == calls[6][1].trust == 0.75
    assert all(config == calls[0][2] for _, _, config in calls)
    assert calls[0][2] == GenerationConfig("mock", 0.0, 256, "reply_v2", 544)


@pytest.mark.parametrize("max_generated_turns", [1, 3])
def test_obeys_configurable_turn_budget(simulation_inputs, max_generated_turns):
    simulation_inputs["config"] = replace(
        simulation_inputs["config"], max_generated_turns=max_generated_turns
    )
    simulation_inputs["run_metadata"] = replace(
        simulation_inputs["run_metadata"], max_generated_turns=max_generated_turns
    )
    result = run_simulation(**simulation_inputs)
    assert result.status == "ok"
    assert len(result.generated_turns) == max_generated_turns


def test_preserves_real_usernames_and_continues_nonconsecutive_ids(simulation_inputs):
    seed = replace(simulation_inputs["seed"], opening_turns=[
        DialogueTurn(10, "Sirex98", "Opening"),
        DialogueTurn(20, "Jayjg", "Response"),
        DialogueTurn(30, "Sirex98", "Follow-up"),
    ])
    simulation_inputs["seed"] = seed
    simulation_inputs["personas"] = assign_personas(seed, load_persona_configs())
    result = run_simulation(**simulation_inputs)
    assert result.status == "ok"
    assert [turn.speaker_id for turn in result.generated_turns] == ["Jayjg", "Sirex98"] * 3
    assert [turn.turn_id for turn in result.generated_turns] == list(range(31, 37))


def test_result_serializes_with_existing_shared_schema(simulation_inputs):
    simulation_inputs["run_id"] = "explicit-run-id"
    result = run_simulation(**simulation_inputs)
    assert result.run_id == "explicit-run-id"
    assert from_dict(SimulationResult, json.loads(json.dumps(to_dict(result)))) == result
    assert result.run_metadata.latency_ms >= 0
    assert result.run_metadata.error is None
    assert simulation_inputs["run_metadata"].latency_ms is None


def test_results_are_accepted_by_existing_judge_and_evaluator(simulation_inputs):
    example = from_dict(LabeledExample, json.loads(
        (FIXTURES / "sample_labeled_examples.json").read_text(encoding="utf-8")
    )[0])
    runs = [
        run_simulation(**{
            **simulation_inputs,
            "config": replace(simulation_inputs["config"], condition=condition),
        })
        for condition in ("static", "dynamic")
    ]
    judge_config = {
        "judge_model_id": "mock-judge",
        "rubric_version": "v1",
        "judge_callable": lambda _: '{"score": 0.2}',
    }
    scores = [score_escalation(run.run_id, run.transcript, judge_config) for run in runs]
    summary = evaluate_conditions([example], runs, scores)
    assert summary["coverage"]["n_completed_pairs"] == 1
    assert summary["pairs"][0]["conversation_id"] == example.seed.conversation_id
    assert summary["pairs"][0]["static_run_id"] == runs[0].run_id
    assert summary["pairs"][0]["dynamic_run_id"] == runs[1].run_id


def test_callback_inputs_contain_no_labels_or_condition_for_generator(simulation_inputs):
    run_simulation(**simulation_inputs)
    for history, actor, config in simulation_inputs["generator"].calls:
        assert set(to_dict(config)) == {
            "model_id", "temperature", "max_output_tokens", "prompt_version", "random_seed"
        }
        assert set(to_dict(actor)) == {"speaker_id", "trust", "stable_traits"}
        for turn in history:
            assert set(to_dict(turn)) == {"turn_id", "speaker_id", "text"}
    for request in simulation_inputs["updater"].client.requests:
        assert set(request) == {"reader_id", "history"}


def test_generator_cannot_mutate_owned_history_or_states(simulation_inputs):
    generator = simulation_inputs["generator"]
    original_seed = deepcopy(simulation_inputs["seed"])
    original_personas = deepcopy(simulation_inputs["personas"])

    def mutating_generator(history, actor, config):
        turn = generator(history, actor, config)
        history.clear()
        actor.stable_traits["role"] = "mutated inside generator"
        return turn

    result = run_simulation(**{**simulation_inputs, "generator": mutating_generator})
    assert result.status == "ok"
    assert len(result.transcript) == 8
    assert simulation_inputs["seed"] == original_seed
    assert simulation_inputs["personas"] == original_personas
    assert all(actor.stable_traits["role"] == "Wikipedia editor" for _, actor, _ in generator.calls)


def test_generator_failure_keeps_previous_turns_and_state_history(simulation_inputs):
    generator = simulation_inputs["generator"]

    def failing_generator(history, actor, config):
        if len(generator.calls) == 2:
            raise TimeoutError("model timed out")
        return generator(history, actor, config)

    result = run_simulation(**{**simulation_inputs, "generator": failing_generator})
    assert result.status == "failed"
    assert len(result.generated_turns) == 2
    assert len(result.transcript) == 4
    assert len(result.state_history) == 3
    assert "generation failed at turn 4" in result.run_metadata.error
    assert "TimeoutError: model timed out" in result.run_metadata.error
    assert from_dict(SimulationResult, to_dict(result)) == result


def test_invalid_rating_keeps_new_message_but_no_completed_update_snapshot(simulation_inputs):
    simulation_inputs["updater"].client.rating = "not a rating"
    result = run_simulation(**simulation_inputs)
    assert result.status == "failed"
    assert len(result.generated_turns) == 1
    assert len(result.transcript) == 3
    assert [snapshot.turn_id for snapshot in result.state_history] == [1]
    assert "state update failed at turn 2" in result.run_metadata.error
    assert "non-numeric" in result.run_metadata.error
    assert len(simulation_inputs["generator"].calls) == 1


@pytest.mark.parametrize("bad_reply, message", [
    ("plain text", "DialogueTurn"),
    (DialogueTurn(2, "B", "Wrong speaker"), "expected 'A'"),
    (DialogueTurn(1, "A", "Repeated ID"), "expected 2"),
    (DialogueTurn(2, "A", " \n "), "empty reply"),
])
def test_invalid_generated_reply_is_reported_as_failed(simulation_inputs, bad_reply, message):
    simulation_inputs["generator"] = lambda *_: bad_reply
    result = run_simulation(**simulation_inputs)
    assert result.status == "failed"
    assert message in result.run_metadata.error
    assert result.generated_turns == []
    assert simulation_inputs["updater"].client.requests == []


@pytest.mark.parametrize("violation, message", [
    ("wrong_type", "StateUpdateResult"),
    ("speaker_id", "speaker_id"),
    ("stable_traits", "stable_traits"),
    ("previous_trust", "previous_trust"),
    ("static_trust", "static condition"),
])
def test_invalid_update_is_reported_without_committing_state(simulation_inputs, violation, message):
    simulation_inputs["config"] = replace(simulation_inputs["config"], condition="static")

    def invalid_updater(current_state, observed_turn, visible_history, config):
        if violation == "wrong_type":
            return current_state
        changes = {
            "speaker_id": {"speaker_id": observed_turn.speaker_id},
            "stable_traits": {"stable_traits": {"role": "changed"}},
            "static_trust": {"trust": 0.5},
        }
        return StateUpdateResult(
            replace(current_state, **changes.get(violation, {})),
            0.0 if violation == "previous_trust" else current_state.trust,
        )

    result = run_simulation(**{**simulation_inputs, "updater": invalid_updater})
    assert result.status == "failed"
    assert message in result.run_metadata.error
    assert len(result.generated_turns) == 1
    assert len(result.state_history) == 1
    assert result.state_history[0].speaker_states["B"].trust == 0.75


@pytest.mark.parametrize("speakers", [("A", "A"), ("A", "B", "C")])
def test_rejects_openings_without_exactly_two_speakers(simulation_inputs, speakers):
    simulation_inputs["seed"] = replace(simulation_inputs["seed"], opening_turns=[
        DialogueTurn(index, speaker, "Comment") for index, speaker in enumerate(speakers)
    ])
    with pytest.raises(ValueError, match="exactly 2"):
        run_simulation(**simulation_inputs)
    assert simulation_inputs["generator"].calls == []


def test_rejects_unordered_opening(simulation_inputs):
    simulation_inputs["seed"] = replace(
        simulation_inputs["seed"], opening_turns=list(reversed(simulation_inputs["seed"].opening_turns))
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        run_simulation(**simulation_inputs)


def test_rejects_persona_keys_that_do_not_match_real_speakers(simulation_inputs):
    simulation_inputs["personas"] = {"someone_else": simulation_inputs["personas"]["A"]}
    with pytest.raises(ValueError, match="speaker_ids"):
        run_simulation(**simulation_inputs)


@pytest.mark.parametrize("field, value", [
    ("model_id", "another-model"),
    ("max_generated_turns", 4),
    ("max_output_tokens", 128),
    ("temperature", 0.8),
    ("random_seed", 123),
])
def test_rejects_metadata_settings_that_disagree_with_config(simulation_inputs, field, value):
    simulation_inputs["run_metadata"] = replace(simulation_inputs["run_metadata"], **{field: value})
    with pytest.raises(ValueError, match=field):
        run_simulation(**simulation_inputs)
    assert simulation_inputs["generator"].calls == []


def test_updater_version_mismatch_is_reported_as_failed(simulation_inputs):
    simulation_inputs["run_metadata"] = replace(
        simulation_inputs["run_metadata"], updater_version="trust_wrong"
    )
    result = run_simulation(**simulation_inputs)
    assert result.status == "failed"
    assert "trust_wrong" in result.run_metadata.error
    assert simulation_inputs["updater"].client.requests == []
