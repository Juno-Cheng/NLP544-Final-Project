"""Unit tests for pairing, integrity, pilot coverage, and honest metrics."""

from dataclasses import dataclass

import pytest

from src.evaluate import evaluate_conditions


def sample():
    examples = [
        {"seed": {"conversation_id": "attack"}, "real_outcome": 1},
        {"seed": {"conversation_id": "civil"}, "real_outcome": 0},
    ]
    runs = [
        {"run_id": f"{cid}-{arm}", "conversation_id": cid, "condition": arm, "status": "completed"}
        for cid in ("attack", "civil") for arm in ("static", "dynamic")
    ]
    values = {"attack-static": 0.2, "attack-dynamic": 0.9,
              "civil-static": 0.8, "civil-dynamic": 0.1}
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
    runs.append({"run_id": "attack-static-two", "conversation_id": "attack", "condition": "static"})
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
    assert set(summary["excluded"][0]["reasons"]) == {"missing_static_run", "missing_dynamic_run"}


def test_mismatched_generation_settings_raise_or_report():
    examples, runs, scores = sample()
    for run in runs:
        run["run_metadata"] = {"simulation_config": {"model_id": "same", "temperature": 0.4}}
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
        run["run_metadata"] = {"config": {"model_id": "same", "temperature": 0.6,
                                          "condition": run["condition"]}}
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
