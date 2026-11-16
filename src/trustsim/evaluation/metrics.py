"""Compare each experiment's judged conversations with the dataset's real outcomes."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from trustsim.data.loader import Label
from trustsim.judge.judge import WALK_AWAYS, Verdict
from trustsim.simulation.conversation import WALK_AWAY, Simulation

NOT_RUN = "not run yet"
RESULT_COLUMNS = (
    "personal_attack",
    "matches",
    "walk_away",
    "ended_by",
    "generated_turns",
    "lowest_trust",
)


@dataclass(frozen=True)
class Outcome:
    """One experiment's finished and judged conversation."""

    simulation: Simulation
    verdict: Verdict


def mcnemar_exact_p(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the conversations where exactly
    one of two paired experiments matched the dataset."""
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def experiment_metrics(
    labels: Mapping[str, Label], outcomes: Mapping[str, Outcome]
) -> dict[str, Any]:
    """How often one experiment's verdicts match the real outcomes, with supporting counts."""
    confusion = Counter(
        (labels[conversation_id].real_outcome, outcome.verdict.personal_attack)
        for conversation_id, outcome in outcomes.items()
    )
    attack_caught, attack_missed = confusion[(1, 1)], confusion[(1, 0)]
    civil_kept, civil_flagged = confusion[(0, 0)], confusion[(0, 1)]
    walk_aways = Counter(outcome.verdict.walk_away for outcome in outcomes.values())
    lowest_trust = {
        real: [
            outcome.simulation.lowest_trust()
            for conversation_id, outcome in outcomes.items()
            if labels[conversation_id].real_outcome == real
        ]
        for real in (1, 0)
    }
    return {
        "conversations": len(outcomes),
        "matches": attack_caught + civil_kept,
        "match_rate": _share(attack_caught + civil_kept, len(outcomes)),
        "judged_attacks": attack_caught + civil_flagged,
        "real_attacks_caught": attack_caught,
        "real_attacks": attack_caught + attack_missed,
        "civil_kept_civil": civil_kept,
        "real_civil": civil_kept + civil_flagged,
        "walk_aways": {value: walk_aways[value] for value in WALK_AWAYS},
        "ended_early": sum(
            outcome.simulation.ended_by == WALK_AWAY for outcome in outcomes.values()
        ),
        "mean_generated_turns": _mean(
            [len(outcome.simulation.turns) for outcome in outcomes.values()]
        ),
        "mean_lowest_trust": {
            "real_attack": _mean(lowest_trust[1]),
            "real_civil": _mean(lowest_trust[0]),
        },
    }


def paired_metrics(
    labels: Mapping[str, Label],
    first: Mapping[str, Outcome],
    second: Mapping[str, Outcome],
    names: tuple[str, str],
) -> dict[str, Any]:
    """Per-conversation agreement of two experiments with the dataset, and McNemar's test."""
    counts: Counter[tuple[bool, bool]] = Counter()
    verdicts_differ = 0
    for conversation_id, label in labels.items():
        first_attack = first[conversation_id].verdict.personal_attack
        second_attack = second[conversation_id].verdict.personal_attack
        counts[(first_attack == label.real_outcome, second_attack == label.real_outcome)] += 1
        verdicts_differ += first_attack != second_attack
    only_first, only_second = counts[(True, False)], counts[(False, True)]
    return {
        "both_match": counts[(True, True)],
        f"only_{names[0]}_matches": only_first,
        f"only_{names[1]}_matches": only_second,
        "neither_matches": counts[(False, False)],
        "verdicts_differ": verdicts_differ,
        "mcnemar_p": mcnemar_exact_p(only_first, only_second),
    }


def evaluate(
    labels: Mapping[str, Label],
    outcomes: Mapping[str, Mapping[str, Outcome]],
    problems: Mapping[tuple[str, str], str],
    experiments: Sequence[str],
) -> dict[str, Any]:
    """Metrics for each experiment over the conversations finished in all of them.

    outcomes maps experiment -> conversation id -> outcome; problems explains a missing
    outcome by (conversation id, experiment). Conversations missing from any experiment are
    left out of every experiment, so all of them are compared on the same conversations.
    """
    compared = {
        conversation_id: label
        for conversation_id, label in labels.items()
        if all(conversation_id in outcomes.get(name, {}) for name in experiments)
    }
    excluded = [
        {
            "conversation_id": conversation_id,
            "reasons": {
                name: problems.get((conversation_id, name), NOT_RUN)
                for name in experiments
                if conversation_id not in outcomes.get(name, {})
            },
        }
        for conversation_id in labels
        if conversation_id not in compared
    ]
    real_attacks = sum(label.real_outcome for label in compared.values())
    result: dict[str, Any] = {
        "selected_conversations": len(labels),
        "compared_conversations": len(compared),
        "real_outcomes": {"attack": real_attacks, "civil": len(compared) - real_attacks},
        "excluded": excluded,
        "experiments": {
            name: experiment_metrics(compared, {cid: outcomes[name][cid] for cid in compared})
            for name in experiments
        },
    }
    if len(experiments) == 2 and compared:
        first, second = experiments
        result["paired"] = paired_metrics(
            compared, outcomes[first], outcomes[second], (first, second)
        )
    return result


def result_rows(
    labels: Mapping[str, Label],
    outcomes: Mapping[str, Mapping[str, Outcome]],
    problems: Mapping[tuple[str, str], str],
    experiments: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per conversation with each experiment's verdict next to the real outcome."""
    rows = []
    for conversation_id, label in labels.items():
        row: dict[str, Any] = {
            "conversation_id": conversation_id,
            "pair_id": label.pair_id or "",
            "real_outcome": label.real_outcome,
        }
        for name in experiments:
            outcome = outcomes.get(name, {}).get(conversation_id)
            if outcome is None:
                row[f"{name}_status"] = problems.get((conversation_id, name), NOT_RUN)
                for column in RESULT_COLUMNS:
                    row[f"{name}_{column}"] = ""
                continue
            verdict, simulation = outcome.verdict, outcome.simulation
            row[f"{name}_status"] = "ok"
            row[f"{name}_personal_attack"] = verdict.personal_attack
            row[f"{name}_matches"] = int(verdict.personal_attack == label.real_outcome)
            row[f"{name}_walk_away"] = verdict.walk_away
            row[f"{name}_ended_by"] = simulation.ended_by
            row[f"{name}_generated_turns"] = len(simulation.turns)
            row[f"{name}_lowest_trust"] = simulation.lowest_trust()
        rows.append(row)
    return rows


def result_columns(experiments: Sequence[str]) -> list[str]:
    """Column order for results.csv."""
    columns = ["conversation_id", "pair_id", "real_outcome"]
    for name in experiments:
        columns.append(f"{name}_status")
        columns.extend(f"{name}_{column}" for column in RESULT_COLUMNS)
    return columns
