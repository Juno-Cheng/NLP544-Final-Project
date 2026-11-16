"""The whole run: simulate both experiments, judge each conversation blind, then evaluate.

A run lives in one folder. Every simulated conversation and every verdict is appended to a log
as soon as it is done, so a stopped run resumes where it left off; failed ones are retried.
"""

import csv
import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustsim.config import AGENT, CRITIC, DYNAMIC, EXPERIMENTS, JUDGE, ExperimentSettings
from trustsim.data.loader import Example
from trustsim.dialogue import Seed
from trustsim.evaluation.metrics import Outcome, evaluate, result_columns, result_rows
from trustsim.evaluation.report import render_summary
from trustsim.judge.judge import Judge, Verdict
from trustsim.llm.client import LLMClient, LLMError, LLMSetupError, UsageMeter
from trustsim.personas.cast import Cast
from trustsim.simulation.agent import ReplyWriter
from trustsim.simulation.conversation import Simulation, simulate
from trustsim.simulation.critic import Critic
from trustsim.storage import RecordLog, write_json

LOGGER = logging.getLogger(__name__)
MANIFEST = "run.json"
SIMULATIONS = "simulations.jsonl"
VERDICTS = "verdicts.jsonl"
METRICS = "metrics.json"
RESULTS = "results.csv"
SUMMARY = "summary.md"
OK = "ok"
ERROR = "error"


class RunMismatchError(RuntimeError):
    """A run folder was started with different settings from the current ones."""


@dataclass(frozen=True)
class Roles:
    """The three model-backed roles of a run."""

    writer: ReplyWriter
    critic: Critic
    judge: Judge


def build_roles(
    clients: Mapping[str, LLMClient], settings: ExperimentSettings
) -> tuple[Roles, dict[str, str]]:
    """The agent writer, critic, and judge, plus the prompt text each one uses."""
    prompts = {
        role: role_settings.prompt_path.read_text(encoding="utf-8")
        for role, role_settings in settings.llm.roles.items()
    }
    attempts = settings.attempts_per_answer
    roles = Roles(
        writer=ReplyWriter(clients[AGENT], prompts[AGENT], attempts),
        critic=Critic(clients[CRITIC], prompts[CRITIC], attempts),
        judge=Judge(clients[JUDGE], prompts[JUDGE], attempts),
    )
    return roles, prompts


def describe_run(
    examples: Sequence[Example],
    cast: Cast,
    settings: ExperimentSettings,
    prompts: Mapping[str, str],
) -> dict[str, Any]:
    """Everything that defines a run's results. Workers, retries, and the experiments chosen
    for one invocation are left out, so a run can be resumed with different values for them."""
    return {
        "conversations": [example.seed.conversation_id for example in examples],
        "max_turns": settings.max_turns,
        "provider": settings.llm.provider,
        "roles": {
            role: {
                "model": role_settings.model,
                "prompt": role_settings.prompt_path.name,
                "max_output_tokens": role_settings.max_output_tokens,
                "temperature": role_settings.temperature,
                "reasoning_effort": role_settings.reasoning_effort,
            }
            for role, role_settings in settings.llm.roles.items()
        },
        "prompt_sha256": {
            role: hashlib.sha256(text.encode("utf-8")).hexdigest() for role, text in prompts.items()
        },
        "personas": cast.summary(),
    }


def run_pipeline(
    run_dir: Path,
    examples: Sequence[Example],
    cast: Cast,
    settings: ExperimentSettings,
    roles: Roles,
    prompts: Mapping[str, str],
    meter: UsageMeter | None = None,
) -> dict[str, Any]:
    """Simulate and judge every conversation not finished yet in each chosen experiment, then
    evaluate everything finished in the run folder. Returns the metrics."""
    description = describe_run(examples, cast, settings, prompts)
    manifest = _open_run(run_dir, description)
    simulations = RecordLog(run_dir / SIMULATIONS, _unit_key)
    verdicts = RecordLog(run_dir / VERDICTS, _unit_key)
    simulated = {key: rec for key, rec in simulations.latest().items() if rec["status"] == OK}
    judged = {key for key, rec in verdicts.latest().items() if rec["status"] == OK}
    units = [
        (example.seed, experiment)
        for example in examples
        for experiment in settings.experiments
        if (example.seed.conversation_id, experiment) not in judged
    ]
    LOGGER.info(
        "%d of %d conversation runs left to do (%s)",
        len(units),
        len(examples) * len(settings.experiments),
        ", ".join(settings.experiments),
    )
    try:
        _run_units(units, cast, settings, roles, simulated, simulations, verdicts)
    except LLMSetupError:
        _close_run(run_dir, manifest, "stopped", meter)
        raise
    metrics = evaluate_run(run_dir, examples)
    _close_run(run_dir, manifest, "completed" if not metrics["excluded"] else "incomplete", meter)
    return metrics


def evaluate_run(run_dir: Path, examples: Sequence[Example]) -> dict[str, Any]:
    """Compare every finished conversation in a run folder with its real outcome, and write
    metrics.json, results.csv, and summary.md. This is the only step that reads labels."""
    labels = {example.seed.conversation_id: example.label for example in examples}
    simulation_records = RecordLog(run_dir / SIMULATIONS, _unit_key).latest()
    verdict_records = RecordLog(run_dir / VERDICTS, _unit_key).latest()
    experiments = [
        name for name in EXPERIMENTS if any(key[1] == name for key in simulation_records)
    ]
    outcomes: dict[str, dict[str, Outcome]] = {name: {} for name in experiments}
    problems: dict[tuple[str, str], str] = {}
    for key, record in simulation_records.items():
        conversation_id, name = key
        verdict = verdict_records.get(key)
        if conversation_id not in labels:
            continue
        if record["status"] != OK:
            problems[key] = f"simulation failed: {record['error']}"
        elif verdict is None:
            problems[key] = "not judged yet"
        elif verdict["status"] != OK:
            problems[key] = f"judge failed: {verdict['error']}"
        else:
            outcomes[name][conversation_id] = Outcome(
                Simulation.from_record(record["simulation"]), Verdict(**verdict["verdict"])
            )
    metrics = evaluate(labels, outcomes, problems, experiments)
    write_json(run_dir / METRICS, metrics)
    with (run_dir / RESULTS).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns(experiments))
        writer.writeheader()
        writer.writerows(result_rows(labels, outcomes, problems, experiments))
    (run_dir / SUMMARY).write_text(render_summary(run_dir.name, metrics), encoding="utf-8")
    return metrics


def _run_units(
    units: Sequence[tuple[Seed, str]],
    cast: Cast,
    settings: ExperimentSettings,
    roles: Roles,
    simulated: Mapping[tuple[str, str], dict[str, Any]],
    simulations: RecordLog,
    verdicts: RecordLog,
) -> None:
    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        futures = [
            pool.submit(
                _run_unit,
                seed,
                experiment,
                cast,
                roles,
                settings.max_turns,
                simulated.get((seed.conversation_id, experiment)),
                simulations,
                verdicts,
            )
            for seed, experiment in units
        ]
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                LOGGER.info("[%d/%d] %s", done, len(units), future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _run_unit(
    seed: Seed,
    experiment: str,
    cast: Cast,
    roles: Roles,
    max_turns: int,
    simulated: dict[str, Any] | None,
    simulations: RecordLog,
    verdicts: RecordLog,
) -> str:
    """Simulate (unless already done) and judge one conversation in one experiment.
    A failure is saved with its error and retried on the next run; it is never dropped."""
    key = {"conversation_id": seed.conversation_id, "experiment": experiment}
    label = f"{seed.conversation_id} {experiment}"
    record = simulated
    if record is None:
        critic = roles.critic if experiment == DYNAMIC else None
        try:
            simulation = simulate(seed, experiment, cast, roles.writer, critic, max_turns)
        except LLMSetupError:
            raise
        except (LLMError, ValueError) as exc:
            LOGGER.warning("%s: simulation failed: %s", label, exc)
            simulations.append({**key, "status": ERROR, "error": _error_text(exc)})
            return f"{label}: simulation failed"
        record = {**key, "status": OK, "simulation": simulation.to_record()}
        simulations.append(record)
    simulation = Simulation.from_record(record["simulation"])
    try:
        verdict = roles.judge.judge(simulation.transcript())
    except LLMSetupError:
        raise
    except (LLMError, ValueError) as exc:
        LOGGER.warning("%s: judge failed: %s", label, exc)
        verdicts.append({**key, "status": ERROR, "error": _error_text(exc)})
        return f"{label}: judge failed"
    verdicts.append({**key, "status": OK, "verdict": asdict(verdict)})
    return (
        f"{label}: {len(simulation.turns)} turns, ended by {simulation.ended_by}; "
        f"judge: personal_attack={verdict.personal_attack}, walk_away={verdict.walk_away}"
    )


def _unit_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return record["conversation_id"], record["experiment"]


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _fingerprint(description: Mapping[str, Any]) -> str:
    canonical = json.dumps(description, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _open_run(run_dir: Path, description: dict[str, Any]) -> dict[str, Any]:
    """The run's manifest, new or resumed. A folder started with other settings is refused."""
    fingerprint = _fingerprint(description)
    manifest_path = run_dir / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RunMismatchError(
                f"{run_dir} was started with different settings (conversations, personas, "
                "models, prompts, or max_turns). Use a new --run-name, or delete that folder "
                "to start over."
            )
        LOGGER.info("resuming the run in %s", run_dir)
    else:
        manifest = {
            "run_name": run_dir.name,
            "created_at": _now(),
            "git_commit": _git_commit(),
            "fingerprint": fingerprint,
            "settings": description,
            "usage": {},
        }
    manifest.update(status="running", updated_at=_now())
    write_json(manifest_path, manifest)
    return manifest


def _close_run(
    run_dir: Path, manifest: dict[str, Any], status: str, meter: UsageMeter | None
) -> None:
    usage = manifest.setdefault("usage", {})
    for role, totals in (meter.totals() if meter else {}).items():
        role_usage = usage.setdefault(role, {})
        for name, value in totals.items():
            role_usage[name] = role_usage.get(name, 0) + value
    manifest.update(status=status, updated_at=_now())
    write_json(run_dir / MANIFEST, manifest)
