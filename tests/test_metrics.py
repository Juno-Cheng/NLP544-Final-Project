import pytest

from trustsim.data.loader import Label
from trustsim.dialogue import Turn
from trustsim.evaluation.metrics import (
    Outcome,
    evaluate,
    mcnemar_exact_p,
    result_columns,
    result_rows,
)
from trustsim.evaluation.report import render_summary
from trustsim.judge.judge import Verdict
from trustsim.simulation.conversation import TURN_LIMIT, WALK_AWAY, GeneratedTurn, Simulation

BOTH = ["static", "dynamic"]


def label(conversation_id: str, real_outcome: int) -> Label:
    return Label(conversation_id, real_outcome, "test", None, False)


LABELS = {
    "a1": label("a1", 1),
    "c1": label("c1", 0),
    "a2": label("a2", 1),
    "c2": label("c2", 0),
}


def outcome(
    attack: int,
    walk_away: str = "none",
    ended_by: str = TURN_LIMIT,
    turns: int = 10,
    lowest: float = 0.75,
) -> Outcome:
    turn = GeneratedTurn("x", "A", "Text.", False, "high", None, {"A": lowest, "B": 0.75})
    simulation = Simulation(
        "c", "static", {"x": "A", "y": "B"}, 0.75, (Turn("x", "Hi."),), (turn,) * turns, ended_by
    )
    return Outcome(simulation, Verdict(attack, walk_away, ""))


def test_match_rate_and_attack_counts():
    outcomes = {"static": {"a1": outcome(0), "c1": outcome(0), "a2": outcome(1), "c2": outcome(0)}}
    metrics = evaluate(LABELS, outcomes, {}, ["static"])
    static = metrics["experiments"]["static"]
    assert static["matches"] == 3
    assert static["match_rate"] == 0.75
    assert (static["real_attacks_caught"], static["real_attacks"]) == (1, 2)
    assert (static["civil_kept_civil"], static["real_civil"]) == (2, 2)
    assert static["judged_attacks"] == 1
    assert "paired" not in metrics


def test_conversation_missing_from_one_experiment_is_left_out_of_both():
    outcomes = {
        "static": {cid: outcome(0) for cid in LABELS},
        "dynamic": {cid: outcome(0) for cid in LABELS if cid != "a2"},
    }
    problems = {("a2", "dynamic"): "judge failed: UnusableAnswerError: no usable answer"}
    metrics = evaluate(LABELS, outcomes, problems, BOTH)
    assert metrics["compared_conversations"] == 3
    assert metrics["experiments"]["static"]["conversations"] == 3
    assert metrics["real_outcomes"] == {"attack": 1, "civil": 2}
    assert metrics["excluded"] == [
        {"conversation_id": "a2", "reasons": {"dynamic": problems[("a2", "dynamic")]}}
    ]


def test_conversation_never_run_is_reported_as_not_run():
    outcomes = {"static": {cid: outcome(0) for cid in LABELS if cid != "c2"}}
    metrics = evaluate(LABELS, outcomes, {}, ["static"])
    assert metrics["excluded"] == [{"conversation_id": "c2", "reasons": {"static": "not run yet"}}]


def test_paired_comparison_counts_and_mcnemar():
    static = {"a1": outcome(0), "c1": outcome(0), "a2": outcome(0), "c2": outcome(0)}
    dynamic = {"a1": outcome(1), "c1": outcome(0), "a2": outcome(1), "c2": outcome(1)}
    paired = evaluate(LABELS, {"static": static, "dynamic": dynamic}, {}, BOTH)["paired"]
    assert paired == {
        "both_match": 1,
        "only_static_matches": 1,
        "only_dynamic_matches": 2,
        "neither_matches": 0,
        "verdicts_differ": 3,
        "mcnemar_p": mcnemar_exact_p(1, 2),
    }


@pytest.mark.parametrize(
    ("only_first", "only_second", "p"),
    [(0, 0, 1.0), (5, 0, 0.0625), (0, 5, 0.0625), (3, 3, 1.0), (6, 1, 0.125)],
)
def test_mcnemar_exact_p_values(only_first, only_second, p):
    assert mcnemar_exact_p(only_first, only_second) == pytest.approx(p)


def test_walk_aways_early_endings_and_turns_are_counted():
    outcomes = {
        "dynamic": {
            "a1": outcome(0, "frustrated", WALK_AWAY, turns=4),
            "c1": outcome(0, "calm", WALK_AWAY, turns=6),
            "a2": outcome(1),
            "c2": outcome(0),
        }
    }
    dynamic = evaluate(LABELS, outcomes, {}, ["dynamic"])["experiments"]["dynamic"]
    assert dynamic["walk_aways"] == {"none": 2, "calm": 1, "frustrated": 1}
    assert dynamic["ended_early"] == 2
    assert dynamic["mean_generated_turns"] == 7.5


def test_frustrated_walk_away_without_an_attack_is_still_a_zero():
    outcomes = {"dynamic": {"a1": outcome(0, "frustrated", WALK_AWAY)}}
    dynamic = evaluate({"a1": LABELS["a1"]}, outcomes, {}, ["dynamic"])["experiments"]["dynamic"]
    assert dynamic["real_attacks_caught"] == 0
    assert dynamic["matches"] == 0


def test_lowest_trust_is_averaged_separately_for_real_attacks_and_civil():
    outcomes = {
        "dynamic": {
            "a1": outcome(1, lowest=0.2),
            "a2": outcome(1, lowest=0.4),
            "c1": outcome(0, lowest=0.6),
            "c2": outcome(0, lowest=0.7),
        }
    }
    lowest = evaluate(LABELS, outcomes, {}, ["dynamic"])["experiments"]["dynamic"][
        "mean_lowest_trust"
    ]
    assert lowest == {"real_attack": pytest.approx(0.3), "real_civil": pytest.approx(0.65)}


def test_result_rows_show_each_verdict_next_to_the_real_outcome():
    outcomes = {"static": {"a1": outcome(1)}, "dynamic": {}}
    problems = {("a1", "dynamic"): "simulation failed: LLMError: timeout"}
    rows = result_rows({"a1": LABELS["a1"]}, outcomes, problems, BOTH)
    assert list(rows[0]) == result_columns(BOTH)
    assert rows[0]["static_matches"] == 1
    assert rows[0]["dynamic_status"] == "simulation failed: LLMError: timeout"
    assert rows[0]["dynamic_personal_attack"] == ""


def test_summary_shows_both_experiments_the_paired_test_and_exclusions():
    outcomes = {
        "static": {cid: outcome(0) for cid in LABELS},
        "dynamic": {cid: outcome(1) for cid in LABELS if cid != "c2"},
    }
    problems = {("c2", "dynamic"): "judge failed: LLMError: timeout"}
    text = render_summary("demo", evaluate(LABELS, outcomes, problems, BOTH))
    assert "| | Static | Dynamic |" in text
    assert "| Matches with the dataset | 1 / 3 | 2 / 3 |" in text
    assert "Exact McNemar test" in text
    assert "- c2: dynamic: judge failed: LLMError: timeout" in text
