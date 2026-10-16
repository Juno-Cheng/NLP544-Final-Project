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
    assert summary["metrics"]["spearman_static"] == pytest.approx(-1.0)
    assert summary["metrics"]["spearman_dynamic"] == pytest.approx(1.0)
    assert summary["pairs"][0]["score_delta"] == pytest.approx(0.7)


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
    examples = examples[:1]
    runs = runs[:2]
    scores = scores[:2]
    summary = evaluate_conditions(examples, runs, scores)
    assert summary["metrics"]["roc_auc_static"] is None
    assert summary["metrics"]["spearman_dynamic"] is None
