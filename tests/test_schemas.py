"""Tests for src/schemas.py.

Fixture JSON is the shared contract. A change to those files changes what
other modules must produce.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.data_loader import load_and_prepare, seeds_only
from src.escalation_judge import score_escalation
from src.evaluate import evaluate_conditions
from src.schemas import (
    AgentState,
    ConversationSeed,
    CoverageReport,
    DataConfig,
    DialogueTurn,
    EscalationScore,
    EvaluationMetrics,
    EvaluationSummary,
    ExcludedPair,
    GenerationConfig,
    JudgeConfig,
    LabeledExample,
    PersonaConfig,
    SimulationConfig,
    SimulationResult,
    StateUpdateResult,
    StateSnapshot,
    UpdaterConfig,
    from_dict,
    judge_payload,
    to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_sample_seed_parses():
    seed = from_dict(ConversationSeed, load_fixture("sample_seed.json"))
    assert seed.conversation_id == "example_001"
    assert [turn.speaker_id for turn in seed.opening_turns] == ["A", "B"]


def test_labeled_example_keeps_outcome_off_the_seed():
    example = from_dict(LabeledExample, load_fixture("sample_labeled_examples.json")[0])
    assert example.real_outcome == 1
    assert example.verified is True
    assert "real_outcome" not in to_dict(example.seed)


def test_verified_defaults_false_when_omitted():
    payload = load_fixture("sample_labeled_examples.json")[0]
    del payload["verified"]
    assert from_dict(LabeledExample, payload).verified is False


def test_seed_rejects_outcome_label():
    payload = load_fixture("sample_seed.json")
    payload["real_outcome"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        from_dict(ConversationSeed, payload)


def test_paired_results_share_opening_and_settings():
    static = from_dict(SimulationResult, load_fixture("sample_result_static.json"))
    dynamic = from_dict(SimulationResult, load_fixture("sample_result_dynamic.json"))
    assert static.condition == "static"
    assert dynamic.condition == "dynamic"
    assert static.conversation_id == dynamic.conversation_id
    assert static.transcript[:2] == dynamic.transcript[:2]
    assert static.run_metadata.paired_run_id == dynamic.run_metadata.paired_run_id
    assert static.run_metadata.model_id == dynamic.run_metadata.model_id
    assert static.run_metadata.prompt_version == dynamic.run_metadata.prompt_version
    assert static.run_metadata.temperature == dynamic.run_metadata.temperature
    assert static.run_metadata.random_seed == dynamic.run_metadata.random_seed
    assert static.run_metadata.max_generated_turns == dynamic.run_metadata.max_generated_turns
    assert static.run_metadata.max_output_tokens == dynamic.run_metadata.max_output_tokens


def test_fixture_trust_matches_condition():
    static = from_dict(SimulationResult, load_fixture("sample_result_static.json"))
    dynamic = from_dict(SimulationResult, load_fixture("sample_result_dynamic.json"))
    static_trust = [
        state.trust
        for snapshot in static.state_history
        for state in snapshot.speaker_states.values()
    ]
    dynamic_trust = [
        state.trust
        for snapshot in dynamic.state_history
        for state in snapshot.speaker_states.values()
    ]
    assert static_trust == [0.7, 0.7, 0.7, 0.7]
    assert len(set(dynamic_trust)) > 1


def test_scores_parse_and_reference_run_ids():
    scores = [from_dict(EscalationScore, row) for row in load_fixture("sample_scores.json")]
    assert [score.run_id for score in scores] == ["example_001-static", "example_001-dynamic"]
    assert all(0.0 <= score.score <= 1.0 for score in scores)


def test_simulation_result_json_round_trip():
    original = from_dict(SimulationResult, load_fixture("sample_result_dynamic.json"))
    encoded = json.loads(json.dumps(to_dict(original), ensure_ascii=False))
    assert from_dict(SimulationResult, encoded) == original


def test_dialogue_turn_requires_speaker():
    with pytest.raises(ValueError, match="speaker_id"):
        from_dict(DialogueTurn, {"turn_id": 0, "text": "hi"})


def test_trust_out_of_range_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        AgentState(speaker_id="A", trust=1.2, stable_traits={})


def test_score_out_of_range_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        EscalationScore(run_id="r", score=1.5, judge_model_id="judge", rubric_version="v1")


def test_condition_must_be_static_or_dynamic():
    payload = load_fixture("sample_result_static.json")
    payload["condition"] = "experimental"
    with pytest.raises(ValueError, match="condition"):
        from_dict(SimulationResult, payload)


def test_generated_turns_must_be_transcript_tail():
    payload = load_fixture("sample_result_static.json")
    payload["generated_turns"][0]["text"] = "a different reply"
    with pytest.raises(ValueError, match="ordered tail"):
        from_dict(SimulationResult, payload)


def test_state_snapshot_speaker_key_must_match():
    with pytest.raises(ValueError, match="speaker_states"):
        StateSnapshot(turn_id=1, speaker_states={"A": AgentState("B", 0.7, {})})


def test_judge_payload_is_dialogue_only():
    result = from_dict(SimulationResult, load_fixture("sample_result_dynamic.json"))
    payload = judge_payload(result.transcript)
    assert [set(turn) for turn in payload] == [{"turn_id", "speaker_id", "text"}] * len(payload)
    dumped = json.dumps(payload)
    for leaked in ("trust", "real_outcome", "model_id", "static", "dynamic"):
        assert leaked not in dumped


def test_data_yaml_matches_data_config():
    raw = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    config = from_dict(DataConfig, raw)
    assert config.source == "cga-wiki"
    assert config.splits == ["test"]
    assert config.sample_size == 20
    assert config.keep_pairs is True
    assert config.min_opening_turns == 3


def test_evaluator_output_matches_evaluation_summary():
    examples = [
        {"seed": {"conversation_id": "attack"}, "real_outcome": 1},
        {"seed": {"conversation_id": "civil"}, "real_outcome": 0},
    ]
    settings = {
        "model_id": "mock",
        "prompt_version": "reply_v2",
        "temperature": 0.0,
        "max_generated_turns": 6,
        "max_output_tokens": 256,
        "random_seed": 544,
    }
    values = {
        "attack-static": 0.2,
        "attack-dynamic": 0.9,
        "civil-static": 0.8,
        "civil-dynamic": 0.1,
    }
    runs, scores = [], []
    for conversation_id in ("attack", "civil"):
        for condition in ("static", "dynamic"):
            run_id = f"{conversation_id}-{condition}"
            runs.append({
                "run_id": run_id,
                "conversation_id": conversation_id,
                "condition": condition,
                "status": "ok",
                "run_metadata": dict(settings),
            })
            scores.append({"run_id": run_id, "score": values[run_id]})
    summary = from_dict(EvaluationSummary, evaluate_conditions(examples, runs, scores))
    encoded = json.loads(json.dumps(to_dict(summary)))
    assert from_dict(EvaluationSummary, encoded) == summary
    assert summary.coverage.n_examples == 2
    assert summary.metrics.roc_auc_dynamic == 1.0
    assert summary.unverified_pair_settings == []


def test_prepared_examples_match_labeled_example_schema(tmp_path):
    seeds_path = tmp_path / "seeds.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    seed = {
        "conversation_id": "c1",
        "source": "cga-wiki",
        "opening_turns": [
            {"turn_id": 0, "speaker_id": "A", "text": "hi"},
            {"turn_id": 1, "speaker_id": "B", "text": "hello"},
        ],
    }
    label = {
        "conversation_id": "c1",
        "real_outcome": 0,
        "label_definition": "conversation_has_personal_attack",
        "split": "test",
        "pair_id": None,
        "verified": False,
    }
    seeds_path.write_text(json.dumps(seed) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
    examples = load_and_prepare(
        "cga-wiki",
        {
            "source": "cga-wiki",
            "seeds_path": str(seeds_path),
            "labels_path": str(labels_path),
            "splits": ["test"],
            "verified_only": False,
            "min_opening_turns": 2,
            "min_speakers": 2,
            "max_speakers": 2,
            "keep_pairs": False,
            "sample_size": None,
            "random_seed": 544,
        },
    )
    parsed = [from_dict(LabeledExample, example) for example in examples]
    agent_seeds = [from_dict(ConversationSeed, seed) for seed in seeds_only(examples)]
    assert parsed[0].real_outcome == 0
    assert agent_seeds == [parsed[0].seed]
    assert "real_outcome" not in json.dumps(to_dict(agent_seeds[0]))


def test_failed_run_keeps_status_and_error():
    payload = load_fixture("sample_result_static.json")
    payload["status"] = "failed"
    payload["generated_turns"] = []
    payload["transcript"] = payload["transcript"][:2]
    payload["state_history"] = []
    payload["run_metadata"]["error"] = "model timeout"
    result = from_dict(SimulationResult, payload)
    assert result.status == "failed"
    assert result.generated_turns == []
    assert result.transcript[0].text == "Why did you remove my edit?"
    assert result.run_metadata.error == "model timeout"


def test_paired_simulation_configs_differ_only_by_condition():
    shared = {
        "model_id": "mock",
        "max_generated_turns": 6,
        "temperature": 0.0,
        "max_output_tokens": 256,
        "random_seed": 544,
    }
    static = to_dict(SimulationConfig(condition="static", **shared))
    dynamic = to_dict(SimulationConfig(condition="dynamic", **shared))
    assert static.pop("condition") == "static"
    assert dynamic.pop("condition") == "dynamic"
    assert static == dynamic


def test_generation_config_rejects_condition():
    with pytest.raises(ValueError, match="unexpected fields"):
        from_dict(
            GenerationConfig,
            {
                "model_id": "mock",
                "temperature": 0,
                "max_output_tokens": 128,
                "prompt_version": "reply_v1",
                "random_seed": 544,
                "condition": "dynamic",
            },
        )


def test_judge_config_rejects_condition_and_label():
    with pytest.raises(ValueError, match="unexpected fields"):
        from_dict(
            JudgeConfig,
            {
                "judge_model_id": "mock-judge",
                "rubric_version": "v1",
                "condition": "static",
                "real_outcome": 1,
            },
        )


def test_judge_yaml_parses_and_runtime_callable_scores():
    raw = yaml.safe_load((ROOT / "configs" / "judge.yaml").read_text(encoding="utf-8"))
    config = from_dict(JudgeConfig, raw)
    assert config.rubric_path == "prompts/judge_rubric_v1.txt"
    assert config.judge_callable is None
    with pytest.raises(ValueError, match="judge_callable"):
        score_escalation("example_001-static", [DialogueTurn(0, "A", "hello")], config)
    scored = score_escalation(
        "example_001-static",
        [DialogueTurn(0, "A", "hello")],
        JudgeConfig(
            judge_model_id=config.judge_model_id,
            rubric_version=config.rubric_version,
            rubric_path=config.rubric_path,
            judge_callable=lambda prompt: '{"score": 0.4}',
        ),
    )
    assert from_dict(EscalationScore, scored).score == 0.4


def test_updater_condition_must_be_static_or_dynamic():
    assert UpdaterConfig(condition="static", updater_version="trust_v1").condition == "static"
    with pytest.raises(ValueError, match="condition"):
        UpdaterConfig(condition="both", updater_version="trust_v1")


def test_state_update_records_trust_before_and_after():
    updated = StateUpdateResult(
        state=AgentState(speaker_id="A", trust=0.5, stable_traits={"role": "editor"}),
        previous_trust=0.7,
    )
    assert updated.previous_trust == 0.7
    assert updated.state.trust == 0.5
    with pytest.raises(ValueError, match="previous_trust"):
        StateUpdateResult(state=updated.state, previous_trust=1.4)


def test_persona_trust_accepts_bounds_and_rejects_outside():
    assert PersonaConfig(persona_id="A", stable_traits={}, initial_trust=0).initial_trust == 0
    assert PersonaConfig(persona_id="A", stable_traits={}, initial_trust=1).initial_trust == 1
    with pytest.raises(ValueError, match="initial_trust"):
        PersonaConfig(persona_id="A", stable_traits={}, initial_trust=-0.1)


def test_agent_state_cannot_be_reassigned():
    state = AgentState(speaker_id="A", trust=0.7, stable_traits={})
    with pytest.raises(AttributeError):
        state.trust = 0.2


def test_seed_requires_an_opening_and_unique_turn_ids():
    payload = load_fixture("sample_seed.json")
    payload["opening_turns"] = []
    with pytest.raises(ValueError, match="opening_turns"):
        from_dict(ConversationSeed, payload)
    payload = load_fixture("sample_seed.json")
    payload["opening_turns"][1]["turn_id"] = 0
    with pytest.raises(ValueError, match="turn_id"):
        from_dict(ConversationSeed, payload)


def test_real_outcome_must_be_zero_or_one():
    payload = load_fixture("sample_labeled_examples.json")[0]
    payload["real_outcome"] = 2
    with pytest.raises(ValueError, match="real_outcome"):
        from_dict(LabeledExample, payload)
    payload = load_fixture("sample_labeled_examples.json")[0]
    payload["real_outcome"] = True
    with pytest.raises(ValueError, match="int"):
        from_dict(LabeledExample, payload)


def test_excluded_pair_records_a_missing_arm():
    row = ExcludedPair(
        conversation_id="example_001",
        real_outcome=0,
        static_run_id="example_001-static",
        dynamic_run_id=None,
        reasons=["missing_dynamic_run"],
    )
    assert row.dynamic_run_id is None
    assert from_dict(ExcludedPair, to_dict(row)) == row


def test_evaluation_summary_must_account_for_every_example():
    coverage = CoverageReport(
        n_examples=1,
        n_attempted=0,
        n_completed_pairs=0,
        n_excluded=0,
        completion_rate=0.0,
        label_counts_selected={},
        label_counts_evaluated={},
    )
    metrics = EvaluationMetrics(
        n_pairs=0,
        label_counts={},
        roc_auc_static=None,
        roc_auc_dynamic=None,
        roc_auc_delta=None,
        spearman_static=None,
        spearman_dynamic=None,
        spearman_delta=None,
        mean_score_delta=None,
    )
    with pytest.raises(ValueError, match="every example"):
        EvaluationSummary(
            pairs=[],
            excluded=[],
            coverage=coverage,
            metrics=metrics,
            unverified_pair_settings=[],
            notes=[],
        )


def test_data_config_rejects_non_positive_sample_size():
    raw = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    raw["sample_size"] = 0
    with pytest.raises(ValueError, match="sample_size"):
        from_dict(DataConfig, raw)
