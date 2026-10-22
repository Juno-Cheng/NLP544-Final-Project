"""Condition-paired evaluation of simulated conversations against held-out labels.

Keep the detailed dict report for audit records. Convert it to the shared
``EvaluationSummary`` through ``to_schema_summary`` at the integration boundary.

By default invalid or incomplete experiments raise a ValueError, preserving the
original fail-fast interface.  Use ``on_incomplete="report"`` for exploratory
smoke tests: valid pairs are evaluated and all incomplete pairs are accounted
for in ``excluded`` and ``coverage``. Structural corruption always raises.
"""

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields
from typing import Any

from src.schemas import EvaluationSummary, ExcludedPair, from_dict

LOGGER = logging.getLogger(__name__)

# These settings should match across arms, if recorded. The actual generated
# text, condition, trust history, run_id, latency, and token usage may differ.
PAIR_SETTING_KEYS = (
    "model_id",
    "prompt_version",
    "temperature",
    "max_generated_turns",
    "max_output_tokens",
    "random_seed",
)
SUCCESS_STATUSES = frozenset({"completed", "success", "succeeded", "ok"})


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
        raise ValueError("Judge score must be a number")  # noqa: TRY004 - preserve public validation contract
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Judge score must be finite and in [0, 1]")
    return value


def _roc_auc(labels, values):
    positive = [s for label, s in zip(labels, values) if label == 1]
    negative = [s for label, s in zip(labels, values) if label == 0]
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + 1 + j) / 2
        for index in order[i:j]:
            ranks[index] = average_rank
        i = j
    return ranks


def _spearman(labels, values):
    if len(values) < 2:
        return None
    x, y = _ranks(labels), _ranks(values)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = sum((a - x_mean) ** 2 for a in x)
    y_norm = sum((b - y_mean) ** 2 for b in y)
    return numerator / math.sqrt(x_norm * y_norm) if x_norm and y_norm else None


def _pair_settings(run):
    """Read recorded generation settings; absence is *not* proof of equality.

    Supports fields on the run, in run_metadata, or in common nested config
    dictionaries. A later integration pass should reconcile this with the
    final experiment runner's manifest format.
    """
    metadata = run.get("run_metadata") or {}
    metadata = _record(metadata)
    sources = [run, metadata]
    for name in ("config", "simulation_config", "generation_config"):
        nested = metadata.get(name)
        if nested is not None:
            sources.append(_record(nested))
    result = {}
    for key in PAIR_SETTING_KEYS:
        for source in sources:
            if key in source and source[key] is not None:
                result[key] = source[key]
                break
    return result


def _check_pair_settings(static_run, dynamic_run):
    static = _pair_settings(static_run)
    dynamic = _pair_settings(dynamic_run)
    mismatched = [
        key
        for key in PAIR_SETTING_KEYS
        if key in static and key in dynamic and static[key] != dynamic[key]
    ]
    unverified = [
        key for key in PAIR_SETTING_KEYS if key not in static or key not in dynamic
    ]
    return mismatched, unverified


def evaluate_conditions(examples, runs, scores, *, on_incomplete="raise"):
    """Join source labels to matched static/dynamic runs *after* judge scoring.

    ``on_incomplete="raise"`` (default) fails on missing, unscored, failed, or
    incomparable pairs. ``"report"`` instead returns good pairs plus an
    ``excluded`` entry for each problematic source conversation. Duplicate
    identifiers, unknown runs, invalid labels/scores, and other structural
    corruption always raise regardless of mode.

    AUC/Spearman are descriptive point estimates on *completed pairs only*;
    they do not establish statistical significance or calibrated prediction.
    """
    if on_incomplete not in ("raise", "report"):
        raise ValueError("on_incomplete must be 'raise' or 'report'")

    examples_by_id = {}
    for raw in examples:
        example = _record(raw)
        cid = _record(example["seed"])["conversation_id"]
        if cid in examples_by_id:
            raise ValueError(f"Duplicate example: {cid}")
        if type(example["real_outcome"]) is not int or example["real_outcome"] not in (
            0,
            1,
        ):
            raise ValueError(f"Invalid human label: {cid}")
        examples_by_id[cid] = example

    scores_by_run = {}
    for raw in scores:
        score = _record(raw)
        rid = score["run_id"]
        if rid in scores_by_run:
            raise ValueError(f"Duplicate judge score for run: {rid}")
        scores_by_run[rid] = _score(score["score"])

    runs_by_cid = {}
    seen_run_ids = set()
    for raw in runs:
        run = _record(raw)
        cid, condition, rid = run["conversation_id"], run["condition"], run["run_id"]
        if cid not in examples_by_id:
            raise ValueError(f"Run with unknown conversation: {cid}")
        if condition not in ("static", "dynamic"):
            raise ValueError(f"Unknown condition: {condition}")
        if rid in seen_run_ids:
            raise ValueError(f"Duplicate run ID: {rid}")
        seen_run_ids.add(rid)
        by_condition = runs_by_cid.setdefault(cid, {})
        if condition in by_condition:
            raise ValueError(f"Duplicate {condition} condition for {cid}")
        by_condition[condition] = run

    unknown_scores = set(scores_by_run) - seen_run_ids
    if unknown_scores:
        raise ValueError(f"Scores without matching runs: {sorted(unknown_scores)}")

    paired, excluded = [], []
    for cid, example in examples_by_id.items():
        arms = runs_by_cid.get(cid, {})
        reasons = []
        for condition in ("static", "dynamic"):
            run = arms.get(condition)
            if run is None:
                reasons.append(f"missing_{condition}_run")
                continue
            status = str(run.get("status", "missing")).lower()
            if status not in SUCCESS_STATUSES:
                reasons.append(f"{condition}_run_status:{status}")
            elif run["run_id"] not in scores_by_run:
                reasons.append(f"missing_judge_score:{run['run_id']}")

        unverified = []
        if set(arms) == {"static", "dynamic"}:
            mismatched, unverified = _check_pair_settings(
                arms["static"], arms["dynamic"]
            )
            for key in mismatched:
                reasons.append(f"mismatched_setting:{key}")

        if reasons:
            if on_incomplete == "raise":
                if any(r.startswith("missing_judge_score:") for r in reasons):
                    raise ValueError(f"Missing judge score for {cid}: {reasons}")
                if any("run_status:" in r for r in reasons):
                    raise ValueError(
                        f"Failed or incomplete run cannot be evaluated for {cid}: {reasons}"
                    )
                if any(r.startswith("mismatched_setting:") for r in reasons):
                    raise ValueError(f"Mismatched paired settings for {cid}: {reasons}")
                raise ValueError(f"Incomplete static/dynamic pair for {cid}: {reasons}")
            excluded.append(
                {
                    "conversation_id": cid,
                    "real_outcome": example["real_outcome"],
                    "static_run_id": arms.get("static", {}).get("run_id"),
                    "dynamic_run_id": arms.get("dynamic", {}).get("run_id"),
                    "reasons": reasons,
                }
            )
            continue

        static_run, dynamic_run = arms["static"], arms["dynamic"]
        static_score = scores_by_run[static_run["run_id"]]
        dynamic_score = scores_by_run[dynamic_run["run_id"]]
        paired.append(
            {
                "conversation_id": cid,
                "real_outcome": example["real_outcome"],
                "static_run_id": static_run["run_id"],
                "dynamic_run_id": dynamic_run["run_id"],
                "static_score": static_score,
                "dynamic_score": dynamic_score,
                "score_delta": dynamic_score - static_score,
                "status": "paired_complete",
                "unverified_pair_settings": unverified,
            }
        )

    # Preserve pair_id and partial scores in evaluation report rows.
    for row in paired:
        row["pair_id"] = examples_by_id[row["conversation_id"]].get("pair_id")
    for row in excluded:
        row["pair_id"] = examples_by_id[row["conversation_id"]].get("pair_id")
        row["static_score"] = scores_by_run.get(row.get("static_run_id"))
        row["dynamic_score"] = scores_by_run.get(row.get("dynamic_run_id"))

    labels = [row["real_outcome"] for row in paired]
    static_scores = [row["static_score"] for row in paired]
    dynamic_scores = [row["dynamic_score"] for row in paired]
    auc_static = _roc_auc(labels, static_scores)
    auc_dynamic = _roc_auc(labels, dynamic_scores)
    rho_static = _spearman(labels, static_scores)
    rho_dynamic = _spearman(labels, dynamic_scores)

    all_unverified = sorted(
        {key for row in paired for key in row["unverified_pair_settings"]}
    )
    notes = []
    if excluded:
        notes.append(
            "Metrics exclude the explicitly listed incomplete or incomparable pairs."
        )
    if len(paired) < 20:
        notes.append("Small pilot: descriptive metrics only; no significance claim.")
    if auc_static is None or auc_dynamic is None:
        notes.append(
            "ROC-AUC undefined unless evaluated pairs contain both outcome labels."
        )
    if rho_static is None or rho_dynamic is None:
        notes.append("Spearman undefined when labels or scores have no rank variation.")
    if all_unverified:
        notes.append(
            "Pair equivalence not fully verified: some generation settings were not recorded in both arms."
        )

    n_examples = len(examples_by_id)
    LOGGER.info(
        "evaluate: %s examples, %s completed pairs, %s excluded; auc static=%s dynamic=%s",
        n_examples,
        len(paired),
        len(excluded),
        auc_static,
        auc_dynamic,
    )
    return {
        "pairs": paired,
        "excluded": excluded,
        "coverage": {
            "n_examples": n_examples,
            "n_attempted": len(runs_by_cid),
            "n_completed_pairs": len(paired),
            "n_excluded": len(excluded),
            "completion_rate": len(paired) / n_examples if n_examples else None,
            "label_counts_selected": dict(
                Counter(e["real_outcome"] for e in examples_by_id.values())
            ),
            "label_counts_evaluated": dict(Counter(labels)),
        },
        "metrics": {
            "n_pairs": len(paired),
            "label_counts": dict(Counter(labels)),
            "roc_auc_static": auc_static,
            "roc_auc_dynamic": auc_dynamic,
            "roc_auc_delta": (
                auc_dynamic - auc_static
                if auc_static is not None and auc_dynamic is not None
                else None
            ),
            "spearman_static": rho_static,
            "spearman_dynamic": rho_dynamic,
            "spearman_delta": (
                rho_dynamic - rho_static
                if rho_static is not None and rho_dynamic is not None
                else None
            ),
            "mean_score_delta": (
                sum(row["score_delta"] for row in paired) / len(paired)
                if paired
                else None
            ),
        },
        "unverified_pair_settings": all_unverified,
        "notes": notes,
    }


def _optional_pair_id(value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("report-only pair_id must be a nonempty string or null")


def _optional_score(value: Any) -> None:
    if value is not None:
        _score(value)


class EvaluationReportAdapter:
    """Project a detailed report into shared schemas without mutating the report.

    Only explicitly registered, validated excluded-row details may remain outside
    the typed summary. Persist the original report too, so those details are kept.
    New schema-native fields pass directly to from_dict instead of being removed.
    """

    def __init__(
        self,
        *,
        excluded_detail_validators: Mapping[str, Callable[[Any], None]] | None = None,
    ) -> None:
        self._excluded_fields = frozenset(field.name for field in fields(ExcludedPair))
        custom = dict(excluded_detail_validators or {})
        if self._excluded_fields.intersection(custom):
            raise ValueError("Cannot register schema fields as report-only details")
        if any(
            not isinstance(name, str) or not name.strip() or not callable(validator)
            for name, validator in custom.items()
        ):
            raise ValueError(
                "Detail validators require nonempty field names and callable validators"
            )
        self._detail_validators = {
            "pair_id": _optional_pair_id,
            "static_score": _optional_score,
            "dynamic_score": _optional_score,
            **custom,
        }

    def to_summary(self, report: Mapping[str, Any]) -> EvaluationSummary:
        """Return a validated summary; unknown fields and missing data remain errors."""
        if not isinstance(report, Mapping):
            raise TypeError("evaluation report must be a mapping")
        payload = deepcopy(dict(report))
        excluded = payload.get("excluded")
        if isinstance(excluded, list):
            for row in excluded:
                if not isinstance(row, dict):
                    continue  # from_dict supplies the schema-path validation error.
                for name, validate in self._detail_validators.items():
                    if name in row and name not in self._excluded_fields:
                        validate(row.pop(name))
        return from_dict(EvaluationSummary, payload)


def to_schema_summary(
    report: Mapping[str, Any],
    *,
    adapter: EvaluationReportAdapter | None = None,
) -> EvaluationSummary:
    """Stable entry point for runners; inject an adapter for explicit PR-specific details."""
    converter = EvaluationReportAdapter() if adapter is None else adapter
    return converter.to_summary(report)
