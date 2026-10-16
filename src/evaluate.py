"""Paired static/dynamic evaluation; dictionary contracts until schemas.py arrives."""

import math
from collections import Counter
from collections.abc import Mapping


def _record(value):
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Expected record-like value, got {type(value).__name__}")


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Judge score must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Judge score must be finite and in [0, 1]")
    return value


def _roc_auc(labels, values):
    positive = [s for y, s in zip(labels, values) if y == 1]
    negative = [s for y, s in zip(labels, values) if y == 0]
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + 1 + j) / 2
        for index in order[i:j]:
            result[index] = average_rank
        i = j
    return result


def _spearman(labels, values):
    if len(values) < 2:
        return None
    x, y = _ranks(labels), _ranks(values)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = sum((a - x_mean) ** 2 for a in x)
    y_norm = sum((b - y_mean) ** 2 for b in y)
    return numerator / math.sqrt(x_norm * y_norm) if x_norm and y_norm else None


def evaluate_conditions(examples, runs, scores):
    """Join each source to exactly one scored run per condition; never silently drop runs.

    Returns a provisional dict until Prakhar defines EvaluationSummary.
    """
    examples_by_id = {}
    for example in examples:
        example = _record(example)
        cid = _record(example["seed"])["conversation_id"]
        if cid in examples_by_id:
            raise ValueError(f"Duplicate example: {cid}")
        if type(example["real_outcome"]) is not int or example["real_outcome"] not in (0, 1):
            raise ValueError(f"Invalid human label: {cid}")
        examples_by_id[cid] = example

    scores_by_run = {}
    for raw in scores:
        score = _record(raw)
        rid = score["run_id"]
        if rid in scores_by_run:
            raise ValueError(f"Duplicate judge score for run: {rid}")
        scores_by_run[rid] = _score(score["score"])

    runs_by_id = {}
    used_run_ids = set()
    for raw in runs:
        run = _record(raw)
        cid, condition, rid = run["conversation_id"], run["condition"], run["run_id"]
        if cid not in examples_by_id:
            raise ValueError(f"Run with unknown conversation: {cid}")
        if condition not in ("static", "dynamic"):
            raise ValueError(f"Unknown condition: {condition}")
        if rid in used_run_ids:
            raise ValueError(f"Duplicate run ID: {rid}")
        used_run_ids.add(rid)
        if rid not in scores_by_run:
            raise ValueError(f"Missing judge score for run: {rid}")
        if str(run.get("status", "")).lower() in ("failed", "error"):
            raise ValueError(f"Failed run cannot be evaluated: {rid}")
        by_condition = runs_by_id.setdefault(cid, {})
        if condition in by_condition:
            raise ValueError(f"Duplicate {condition} condition for {cid}")
        by_condition[condition] = run

    unknown_scores = set(scores_by_run) - used_run_ids
    if unknown_scores:
        raise ValueError(f"Scores without matching runs: {sorted(unknown_scores)}")

    paired = []
    for cid, example in examples_by_id.items():
        conditions = runs_by_id.get(cid, {})
        if set(conditions) != {"static", "dynamic"}:
            raise ValueError(f"Incomplete static/dynamic pair for {cid}: {sorted(conditions)}")
        static_run, dynamic_run = conditions["static"], conditions["dynamic"]
        static_score = scores_by_run[static_run["run_id"]]
        dynamic_score = scores_by_run[dynamic_run["run_id"]]
        paired.append({
            "conversation_id": cid,
            "real_outcome": example["real_outcome"],
            "static_run_id": static_run["run_id"],
            "dynamic_run_id": dynamic_run["run_id"],
            "static_score": static_score,
            "dynamic_score": dynamic_score,
            "score_delta": dynamic_score - static_score,
            "status": "paired_complete",
        })

    labels = [r["real_outcome"] for r in paired]
    static_scores = [r["static_score"] for r in paired]
    dynamic_scores = [r["dynamic_score"] for r in paired]
    return {
        "pairs": paired,
        "metrics": {
            "n_pairs": len(paired),
            "label_counts": dict(Counter(labels)),
            "roc_auc_static": _roc_auc(labels, static_scores),
            "roc_auc_dynamic": _roc_auc(labels, dynamic_scores),
            "spearman_static": _spearman(labels, static_scores),
            "spearman_dynamic": _spearman(labels, dynamic_scores),
        },
    }
