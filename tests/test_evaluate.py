"""Unit tests for pairing, integrity, pilot coverage, and honest metrics."""

import json
from copy import deepcopy
from dataclasses import dataclass

import pytest

from src.evaluate import EvaluationReportAdapter, evaluate_conditions, to_schema_summary
from src.schemas import EvaluationSummary, from_dict, to_dict


def sample():
    examples = [
        {"seed": {"conversation_id": "attack"}, "real_outcome": 1},
        {"seed": {"conversation_id": "civil"}, "real_outcome": 0},
    ]
    runs = [
        {
            "run_id": f"{cid}-{arm}",
            "conversation_id": cid,
            "condition": arm,
            "status": "completed",
        }
        for cid in ("attack", "civil")
        for arm in ("static", "dynamic")
    ]
    values = {
        "attack-static": 0.2,
        "attack-dynamic": 0.9,
        "civil-static": 0.8,
        "civil-dynamic": 0.1,
    }
    scores = [{"run_id": rid, "score": value} for rid, value in values.items()]
    return examples, runs, scores


def test_paired_scores_and_metrics():
    summary = evaluate_conditions(*sample())
    assert len(summary["pairs"]) == 2
    assert summary["metrics"]["roc_auc_static"] == 0.0
    assert summary["metrics"]["roc_auc_dynamic"] == 1.0
    assert summary["metrics"]["roc_auc_delta"] == 1.0
    assert summary["metrics"]["spearman_static"] == pytest.approx(-1.0)
    assert summary["metrics"]["spearman_dynamic"] == pytest.approx(1.0)
    assert summary["pairs"][0]["score_delta"] == pytest.approx(0.7)
    assert summary["coverage"]["completion_rate"] == 1.0
    assert summary["excluded"] == []


def test_missing_partner_fails():
    examples, runs, scores = sample()
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_conditions(examples, runs[:3], scores[:3])


def test_missing_judge_score_fails():
    examples, runs, scores = sample()
    with pytest.raises(ValueError, match="Missing judge score"):
        evaluate_conditions(examples, runs, scores[:3])


def test_invalid_score_fails():
    examples, runs, scores = sample()
    scores[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_conditions(examples, runs, scores)


def test_duplicate_condition_fails():
    examples, runs, scores = sample()
    runs.append(
        {
            "run_id": "attack-static-two",
            "conversation_id": "attack",
            "condition": "static",
        }
    )
    scores.append({"run_id": "attack-static-two", "score": 0.2})
    with pytest.raises(ValueError, match="Duplicate static"):
        evaluate_conditions(examples, runs, scores)


def test_auc_undefined_for_one_class():
    examples, runs, scores = sample()
    summary = evaluate_conditions(examples[:1], runs[:2], scores[:2])
    assert summary["metrics"]["roc_auc_static"] is None
    assert summary["metrics"]["spearman_dynamic"] is None
    assert any("ROC-AUC undefined" in n for n in summary["notes"])


def test_report_failed_run_with_no_score_and_keep_good_pair():
    examples, runs, scores = sample()
    runs[1]["status"] = "failed"
    scores = [s for s in scores if s["run_id"] != "attack-dynamic"]
    summary = evaluate_conditions(examples, runs, scores, on_incomplete="report")
    assert [p["conversation_id"] for p in summary["pairs"]] == ["civil"]
    assert summary["excluded"][0]["conversation_id"] == "attack"
    assert "dynamic_run_status:failed" in summary["excluded"][0]["reasons"]
    assert summary["coverage"]["n_completed_pairs"] == 1
    assert summary["coverage"]["n_excluded"] == 1
    assert summary["coverage"]["completion_rate"] == pytest.approx(0.5)
    assert summary["metrics"]["roc_auc_dynamic"] is None
    assert summary["coverage"]["label_counts_selected"] == {1: 1, 0: 1}
    assert summary["coverage"]["label_counts_evaluated"] == {0: 1}


def test_report_missing_partner_and_missing_score():
    examples, runs, scores = sample()
    runs = runs[:3]
    scores = scores[:2]  # attack complete but civil-static lacks a score
    summary = evaluate_conditions(examples, runs, scores, on_incomplete="report")
    assert [p["conversation_id"] for p in summary["pairs"]] == ["attack"]
    reasons = summary["excluded"][0]["reasons"]
    assert "missing_dynamic_run" in reasons
    assert "missing_judge_score:civil-static" in reasons


def test_report_wholly_missing_example():
    examples, runs, scores = sample()
    examples.append({"seed": {"conversation_id": "never-attempted"}, "real_outcome": 0})
    summary = evaluate_conditions(examples, runs, scores, on_incomplete="report")
    assert summary["coverage"]["n_examples"] == 3
    assert summary["coverage"]["n_attempted"] == 2
    assert summary["coverage"]["n_excluded"] == 1
    assert set(summary["excluded"][0]["reasons"]) == {
        "missing_static_run",
        "missing_dynamic_run",
    }


def test_mismatched_generation_settings_raise_or_report():
    examples, runs, scores = sample()
    for run in runs:
        run["run_metadata"] = {
            "simulation_config": {"model_id": "same", "temperature": 0.4}
        }
    runs[1]["run_metadata"]["simulation_config"]["model_id"] = "different"
    with pytest.raises(ValueError, match="Mismatched paired settings"):
        evaluate_conditions(examples, runs, scores)
    summary = evaluate_conditions(examples, runs, scores, on_incomplete="report")
    assert len(summary["pairs"]) == 1
    assert "mismatched_setting:model_id" in summary["excluded"][0]["reasons"]


def test_missing_settings_are_flagged_not_assumed_equal():
    summary = evaluate_conditions(*sample())
    assert "model_id" in summary["unverified_pair_settings"]
    assert "random_seed" in summary["pairs"][0]["unverified_pair_settings"]
    assert any("not fully verified" in n for n in summary["notes"])


def test_static_dynamic_difference_is_expected_not_a_setting_mismatch():
    examples, runs, scores = sample()
    for run in runs:
        run["run_metadata"] = {
            "config": {
                "model_id": "same",
                "temperature": 0.6,
                "condition": run["condition"],
            }
        }
    assert len(evaluate_conditions(examples, runs, scores)["pairs"]) == 2


def test_failed_run_strict_mode_raises_even_if_unscored():
    examples, runs, scores = sample()
    runs[0]["status"] = "failed"
    scores = scores[1:]
    with pytest.raises(ValueError, match="Failed or incomplete"):
        evaluate_conditions(examples, runs, scores)


def test_unexpected_run_status_is_not_treated_as_success():
    examples, runs, scores = sample()
    runs[0]["status"] = "running"
    summary = evaluate_conditions(examples, runs, scores, on_incomplete="report")
    assert "static_run_status:running" in summary["excluded"][0]["reasons"]


def test_orphan_judge_score_always_raises():
    examples, runs, scores = sample()
    scores.append({"run_id": "unrecognized", "score": 0.8})
    with pytest.raises(ValueError, match="Scores without matching runs"):
        evaluate_conditions(examples, runs, scores, on_incomplete="report")


def test_duplicate_run_id_always_raises():
    examples, runs, scores = sample()
    runs[-1]["run_id"] = runs[0]["run_id"]
    with pytest.raises(ValueError, match="Duplicate run ID"):
        evaluate_conditions(examples, runs, scores, on_incomplete="report")


def test_invalid_label_and_bool_scores_raise():
    examples, runs, scores = sample()
    examples[0]["real_outcome"] = "1"
    with pytest.raises(ValueError, match="Invalid human label"):
        evaluate_conditions(examples, runs, scores)
    examples[0]["real_outcome"] = 1
    scores[0]["score"] = True
    with pytest.raises(ValueError, match="Judge score must be a number"):
        evaluate_conditions(examples, runs, scores)


def test_empty_inputs_are_explicit_not_fake_metrics():
    summary = evaluate_conditions([], [], [], on_incomplete="report")
    assert summary["coverage"]["completion_rate"] is None
    assert summary["metrics"]["n_pairs"] == 0
    assert summary["metrics"]["roc_auc_dynamic"] is None
    assert summary["metrics"]["mean_score_delta"] is None


def test_metric_ties_are_well_defined():
    examples, runs, scores = sample()
    for score in scores:
        score["score"] = 0.5
    summary = evaluate_conditions(examples, runs, scores)
    assert summary["metrics"]["roc_auc_static"] == 0.5
    assert summary["metrics"]["spearman_dynamic"] is None


def test_invalid_incomplete_policy_raises():
    with pytest.raises(ValueError, match="on_incomplete"):
        evaluate_conditions(*sample(), on_incomplete="ignore")


@dataclass
class Record:
    run_id: str
    score: float


def test_dataclass_score_accepted():
    examples, runs, scores = sample()
    scores = [Record(s["run_id"], s["score"]) for s in scores]
    assert len(evaluate_conditions(examples, runs, scores)["pairs"]) == 2


def test_schema_summary_conversion():
    from src.evaluate import to_schema_summary
    from src.schemas import EvaluationSummary, from_dict, to_dict

    report = evaluate_conditions(*sample())
    summary = to_schema_summary(report)

    assert isinstance(summary, EvaluationSummary)
    assert summary.coverage.n_examples == 2
    assert summary.coverage.n_completed_pairs == 2
    assert len(summary.pairs) == 2

    assert summary.metrics.roc_auc_static == 0.0
    assert summary.metrics.roc_auc_dynamic == 1.0

    # Verify the result can be serialized and reconstructed.
    assert from_dict(EvaluationSummary, to_dict(summary)) == summary


def test_schema_conversion_preserves_failed_pair_details():
    from src.evaluate import to_schema_summary
    from src.schemas import EvaluationSummary, from_dict, to_dict

    examples, runs, scores = sample()
    examples[0]["pair_id"] = "civil"
    runs[1]["status"] = "failed"
    scores = [s for s in scores if s["run_id"] != "attack-dynamic"]

    report = evaluate_conditions(examples, runs, scores, on_incomplete="report")
    original = deepcopy(report)
    summary = to_schema_summary(report)

    excluded = next(r for r in summary.excluded if r.conversation_id == "attack")
    assert excluded.static_run_id == "attack-static"
    assert excluded.dynamic_run_id == "attack-dynamic"
    assert excluded.reasons == ["dynamic_run_status:failed"]
    assert report == original
    assert report["excluded"][0]["pair_id"] == "civil"
    assert report["excluded"][0]["static_score"] == 0.2
    assert report["excluded"][0]["dynamic_score"] is None
    assert report["excluded"][0]["reasons"] == ["dynamic_run_status:failed"]
    assert from_dict(EvaluationSummary, to_dict(summary)) == summary


@pytest.fixture
def failed_report():
    examples, runs, scores = sample()
    runs[1]["status"] = "failed"
    scores = [score for score in scores if score["run_id"] != runs[1]["run_id"]]
    return evaluate_conditions(examples, runs, scores, on_incomplete="report")


def test_adapter_accepts_current_schema_only_report(failed_report):
    for row in failed_report["excluded"]:
        for name in ("pair_id", "static_score", "dynamic_score"):
            row.pop(name)
    expected = from_dict(EvaluationSummary, failed_report)
    assert to_schema_summary(failed_report) == expected


def test_adapter_preserves_completed_pair_id_and_all_metrics():
    examples, runs, scores = sample()
    examples[0]["pair_id"] = "civil"
    report = evaluate_conditions(examples, runs, scores)
    summary = to_schema_summary(report)
    assert summary.pairs[0].pair_id == "civil"
    assert to_dict(summary.metrics) == report["metrics"]
    assert summary.pairs[0].score_delta == pytest.approx(0.7)
    assert summary.unverified_pair_settings == report["unverified_pair_settings"]


def test_adapter_empty_report_and_json_roundtrip():
    report = evaluate_conditions([], [], [], on_incomplete="report")
    summary = to_schema_summary(report)
    assert summary.coverage.n_examples == 0
    assert summary.metrics.roc_auc_static is None
    assert (
        from_dict(EvaluationSummary, json.loads(json.dumps(to_dict(summary))))
        == summary
    )


@pytest.mark.parametrize("field", ["pair_id", "static_score", "dynamic_score"])
def test_adapter_validates_details_before_projecting(failed_report, field):
    failed_report["excluded"][0][field] = True
    original = deepcopy(failed_report)
    with pytest.raises(ValueError):
        to_schema_summary(failed_report)
    assert failed_report == original


@pytest.mark.parametrize("location", ["root", "excluded", "pairs", "metrics"])
def test_adapter_rejects_unrecognized_fields(failed_report, location):
    target = failed_report if location == "root" else failed_report[location]
    if isinstance(target, list):
        target = target[0]
    target["unreviewed_field"] = "must not be silently lost"
    with pytest.raises(ValueError, match="unexpected fields"):
        to_schema_summary(failed_report)


def test_adapter_requires_schema_fields_and_consistent_counts(failed_report):
    broken = deepcopy(failed_report)
    del broken["pairs"][0]["score_delta"]
    with pytest.raises(ValueError, match="score_delta"):
        to_schema_summary(broken)
    broken = deepcopy(failed_report)
    broken["coverage"]["n_excluded"] = 0
    with pytest.raises(ValueError, match="n_excluded"):
        to_schema_summary(broken)


def test_custom_detail_validator_is_injected_without_global_changes(failed_report):
    def validate_review_note(value):
        if not isinstance(value, str):
            raise TypeError("review_note must be a string")

    failed_report["excluded"][0]["review_note"] = "Retry next pilot"
    adapter = EvaluationReportAdapter(
        excluded_detail_validators={"review_note": validate_review_note}
    )
    original = deepcopy(failed_report)
    summary = to_schema_summary(failed_report, adapter=adapter)
    assert summary.coverage.n_excluded == 1
    assert failed_report == original
    with pytest.raises(ValueError, match="unexpected fields"):
        to_schema_summary(failed_report)
    failed_report["excluded"][0]["review_note"] = 123
    with pytest.raises(TypeError, match="review_note"):
        to_schema_summary(failed_report, adapter=adapter)


def test_adapter_cannot_drop_required_schema_fields():
    with pytest.raises(ValueError, match="schema fields"):
        EvaluationReportAdapter(
            excluded_detail_validators={"reasons": lambda value: None}
        )


@pytest.mark.parametrize("validators", [{"note": None}, {"": lambda value: None}])
def test_adapter_rejects_invalid_configuration(validators):
    with pytest.raises(ValueError, match="validators"):
        EvaluationReportAdapter(excluded_detail_validators=validators)


def test_adapter_detaches_typed_summary_from_report(failed_report):
    summary = to_schema_summary(failed_report)
    summary.notes.append("local annotation")
    summary.excluded[0].reasons.append("local reason")
    assert "local annotation" not in failed_report["notes"]
    assert "local reason" not in failed_report["excluded"][0]["reasons"]
