"""A readable Markdown summary of a run's metrics."""

from collections.abc import Callable, Mapping
from typing import Any


def _fraction(part: int, whole: int) -> str:
    return f"{part} / {whole}" if whole else "n/a"


def _number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


TABLE_ROWS: tuple[tuple[str, Callable[[Mapping[str, Any]], str]], ...] = (
    ("Matches with the dataset", lambda m: _fraction(m["matches"], m["conversations"])),
    ("Match rate", lambda m: _number(m["match_rate"])),
    (
        "Real attacks judged as attacks",
        lambda m: _fraction(m["real_attacks_caught"], m["real_attacks"]),
    ),
    (
        "Civil conversations judged civil",
        lambda m: _fraction(m["civil_kept_civil"], m["real_civil"]),
    ),
    (
        "Walk-aways, calm / frustrated",
        lambda m: f"{m['walk_aways']['calm']} / {m['walk_aways']['frustrated']}",
    ),
    ("Ended early by an agent leaving", lambda m: str(m["ended_early"])),
    ("Mean generated turns", lambda m: _number(m["mean_generated_turns"], 1)),
    (
        "Mean lowest trust, real attack / civil",
        lambda m: (
            f"{_number(m['mean_lowest_trust']['real_attack'])} / "
            f"{_number(m['mean_lowest_trust']['real_civil'])}"
        ),
    ),
)


def render_summary(run_name: str, metrics: Mapping[str, Any]) -> str:
    """Markdown with one column per experiment, the paired comparison, and any exclusions."""
    experiments = list(metrics["experiments"])
    real = metrics["real_outcomes"]
    lines = [
        f"# Run {run_name}",
        "",
        f"Compared {metrics['compared_conversations']} of {metrics['selected_conversations']} "
        f"conversations: {real['attack']} really ended in a personal attack, "
        f"{real['civil']} stayed civil.",
        "",
        "| | " + " | ".join(name.capitalize() for name in experiments) + " |",
        "|---|" + "---|" * len(experiments),
    ]
    for label, cell in TABLE_ROWS:
        cells = " | ".join(cell(metrics["experiments"][name]) for name in experiments)
        lines.append(f"| {label} | {cells} |")
    paired = metrics.get("paired")
    if paired:
        first, second = experiments
        lines += [
            "",
            "## Paired comparison",
            "",
            f"- Both match the dataset: {paired['both_match']}",
            f"- Only {first} matches: {paired[f'only_{first}_matches']}",
            f"- Only {second} matches: {paired[f'only_{second}_matches']}",
            f"- Neither matches: {paired['neither_matches']}",
            f"- Verdicts differ between the experiments: {paired['verdicts_differ']}",
            f"- Exact McNemar test: p = {paired['mcnemar_p']:.3f}",
        ]
    if metrics["excluded"]:
        lines += ["", "## Left out of the comparison", ""]
        for item in metrics["excluded"]:
            reasons = "; ".join(f"{name}: {reason}" for name, reason in item["reasons"].items())
            lines.append(f"- {item['conversation_id']}: {reasons}")
    return "\n".join(lines) + "\n"
