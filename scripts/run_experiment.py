"""Run one paired static/dynamic experiment and save every artifact.

Stages, in order:
    1. Jon's load_and_prepare, or the committed fixture, supplies LabeledExample rows.
    2. Magnus's assign_personas maps each opening onto roles A and B. The seed is all it sees.
    3. Aditya's run_simulation runs twice per seed. Arya's PersonaResponseGenerator writes
       replies and Magnus's TrustUpdater applies the condition. Both arms share the settings
       in configs/simulation.yaml.
    4. Shadi's score_escalation scores each successful transcript. The payload is dialogue
       text only: no condition, trust, model id, or label.
    5. Shadi's evaluate_conditions joins the held-out labels, then to_schema_summary
       validates the shared EvaluationSummary.

Usage:
    python scripts/run_experiment.py --mode mock --limit 1
    python scripts/run_experiment.py --mode live --limit 5

Mock mode uses canned replies and tests/fixtures/sample_labeled_examples.json.
Live mode calls OpenAI and reads data/processed through configs/data.yaml.
Set OPENAI_API_KEY yourself; this script does not load .env.

Outputs under outputs/<experiment_id>/:
    manifest.json, simulations/, judge_inputs/, scores.jsonl,
    evaluation.json, evaluation_report.json, logs/events.jsonl, logs/model_calls.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from itertools import cycle, islice
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data_loader import load_and_prepare
from src.escalation_judge import make_llm_judge_callable, score_escalation
from src.evaluate import evaluate_conditions, to_schema_summary
from src.llm_client import MockLLMClient, OpenAILLMClient
from src.persona import assign_personas, load_persona_configs
from src.response_generator import PersonaResponseGenerator
from src.schemas import (
    CONDITIONS,
    EscalationScore,
    GenerationConfig,
    JudgeConfig,
    LabeledExample,
    RunMetadata,
    SimulationConfig,
    SimulationResult,
    from_dict,
    to_dict,
)
from src.simulator import run_simulation
from src.state_updater import TrustUpdater

SIMULATION_CONFIG = ROOT / "configs" / "simulation.yaml"
JUDGE_CONFIG = ROOT / "configs" / "judge.yaml"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "sample_labeled_examples.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

# JudgeConfig has no temperature field. The judge call stays deterministic and short.
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_OUTPUT_TOKENS = 64
# Neutral canned replies. Ratings of 0.5 leave dynamic trust unchanged. The judge score
# is fixed and is never taken from the held-out label.
MOCK_REPLIES = (
    "Could you point me to the source for that change?",
    "The sentence did not match the cited reference, so I removed it.",
    "I can restore it if we add a reliable source.",
    "Let us check the policy page and adjust the wording.",
)
MOCK_RATINGS = ("0.5",)
MOCK_JUDGE_RESPONSE = '{"score": 0.5}'


@dataclass(frozen=True)
class SimulationSettings:
    """Fields of configs/simulation.yaml. Both arms receive this same generation setup."""

    model_id: str
    prompt_version: str
    temperature: float
    max_generated_turns: int
    max_output_tokens: int
    random_seed: int


def load_simulation_settings(path: Path = SIMULATION_CONFIG) -> SimulationSettings:
    """Return the shared generation settings, rejecting a missing or mistyped field."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    known = set(SimulationSettings.__dataclass_fields__)
    unexpected = sorted(set(raw) - known)
    missing = sorted(known - set(raw))
    if unexpected or missing:
        raise ValueError(f"{path} fields mismatch; unexpected={unexpected} missing={missing}")
    settings = SimulationSettings(
        model_id=str(raw["model_id"]),
        prompt_version=str(raw["prompt_version"]),
        temperature=float(raw["temperature"]),
        max_generated_turns=int(raw["max_generated_turns"]),
        max_output_tokens=int(raw["max_output_tokens"]),
        random_seed=int(raw["random_seed"]),
    )
    prompt_path = ROOT / "prompts" / f"{settings.prompt_version}.txt"
    if not prompt_path.is_file():
        raise ValueError(f"missing reply prompt {prompt_path}")
    if not 0 <= settings.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if settings.max_generated_turns <= 0:
        raise ValueError("max_generated_turns must be positive")
    if settings.max_output_tokens < 16:
        raise ValueError("max_output_tokens must be at least 16")
    return settings


def load_judge_config(path: Path = JUDGE_CONFIG) -> JudgeConfig:
    """Return the rubric settings. The callable is attached later and is not in the yaml."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    return from_dict(JudgeConfig, raw)


def load_examples(source: str) -> list[LabeledExample]:
    """Load labeled rows. Processed data is Jon's loader; fixture is the committed sample."""
    if source == "fixture":
        rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    elif source == "processed":
        try:
            rows = load_and_prepare()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Processed seeds are missing. Run python scripts/prepare_data.py first."
            ) from exc
    else:
        raise ValueError("source must be 'fixture' or 'processed'")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} source produced no examples")
    return [from_dict(LabeledExample, row) for row in rows]


def resolve_model_id(mode: str, configured: str) -> str:
    """Keep the yaml model in mock mode. Live mode prefers OPENAI_MODEL."""
    if mode == "mock":
        return configured
    model_id = os.environ.get("OPENAI_MODEL") or configured
    if not model_id or model_id == "mock":
        raise RuntimeError(
            "live mode needs OPENAI_MODEL or a real model_id in configs/simulation.yaml"
        )
    return model_id


def resolve_judge_model(mode: str, configured: str) -> str:
    """Refuse the mock judge id once the run is calling a live model."""
    if mode == "live" and configured in {"", "mock", "mock-judge"}:
        raise RuntimeError("live mode needs a real judge_model_id in configs/judge.yaml")
    return configured


def current_commit() -> str | None:
    """Return HEAD when git is available. A missing commit does not stop the run."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one UTF-8 JSON object. Failed runs are recorded here rather than dropped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _mock_client(lines: Iterable[str], count: int) -> MockLLMClient:
    return MockLLMClient(islice(cycle(lines), count))


def build_live_client() -> OpenAILLMClient:
    """Open one OpenAI client. Credentials stay in the environment."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("live mode needs OPENAI_API_KEY in the environment")
    from openai import OpenAI

    return OpenAILLMClient(OpenAI())


def make_judge(
    mode: str, live: OpenAILLMClient | None, judge_model_id: str, random_seed: int
) -> Callable[[str], str]:
    """Return the function score_escalation calls. Mock mode never touches the network."""
    if mode == "mock":
        return lambda _prompt: MOCK_JUDGE_RESPONSE
    if live is None:
        raise RuntimeError("live mode is missing its OpenAI client")
    generation = GenerationConfig(
        judge_model_id,
        JUDGE_TEMPERATURE,
        JUDGE_MAX_OUTPUT_TOKENS,
        "judge_rubric_v1",
        random_seed,
    )
    return make_llm_judge_callable(live, generation)


def run_arm(
    example: LabeledExample,
    personas: dict,
    condition: str,
    settings: SimulationSettings,
    updater_version: str,
    experiment_id: str,
    mode: str,
    live: OpenAILLMClient | None,
    calls_path: Path,
) -> SimulationResult:
    """Run one condition. The simulator receives the seed, never the label."""
    run_id = f"{example.seed.conversation_id}-{condition}"
    turns = settings.max_generated_turns
    # Arya: one reply client per arm so a mock cursor is not shared with the other condition.
    generator = PersonaResponseGenerator(
        _mock_client(MOCK_REPLIES, turns) if mode == "mock" else live,
        on_generation=lambda event: append_jsonl(
            calls_path, {**event, "run_id": run_id, "stage": "reply"}
        ),
    )
    # Magnus: static trust never calls the rater. Dynamic trust uses updater.yaml.
    updater = TrustUpdater(
        _mock_client(MOCK_RATINGS, turns) if mode == "mock" else live,
        settings.model_id,
        on_rating=lambda event: append_jsonl(
            calls_path, {**event, "run_id": run_id, "stage": "rating"}
        ),
    )
    config = SimulationConfig(
        condition,
        settings.model_id,
        settings.max_generated_turns,
        settings.temperature,
        settings.max_output_tokens,
        settings.random_seed,
    )
    metadata = RunMetadata(
        experiment_id=experiment_id,
        paired_run_id=f"pair-{example.seed.conversation_id}",
        model_id=settings.model_id,
        prompt_version=settings.prompt_version,
        updater_version=updater_version,
        max_generated_turns=settings.max_generated_turns,
        max_output_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
        random_seed=settings.random_seed,
    )
    # Aditya: opening turns do not update trust. A callback failure is a saved result.
    return run_simulation(
        example.seed,
        personas,
        config,
        generator,
        updater,
        run_metadata=metadata,
        run_id=run_id,
    )


def score_arm(
    result: SimulationResult,
    judge_config: JudgeConfig,
    judge: Callable[[str], str],
    judge_dir: Path,
    events_path: Path,
) -> EscalationScore | None:
    """Score one successful transcript. The prompt saved here is the exact judge input."""
    captured: dict[str, str] = {}

    def recording_judge(prompt: str) -> str:
        captured["prompt"] = prompt
        (judge_dir / f"{result.run_id}.txt").write_text(prompt, encoding="utf-8")
        return judge(prompt)

    try:
        raw = score_escalation(
            result.run_id,
            result.transcript,
            judge_config,
            judge_callable=recording_judge,
        )
        return from_dict(EscalationScore, raw)
    except Exception as exc:  # noqa: BLE001 -- keep the transcript and record the judge failure.
        append_jsonl(
            events_path,
            {
                "stage": "judge",
                "run_id": result.run_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_saved": "prompt" in captured,
            },
        )
        return None


def run_experiment(
    *,
    mode: str,
    limit: int | None,
    experiment_id: str,
    source: str,
    output_root: Path,
) -> Path:
    """Execute both arms for each example and write the experiment directory."""
    if mode not in {"mock", "live"}:
        raise ValueError("mode must be 'mock' or 'live'")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    settings = load_simulation_settings()
    settings = SimulationSettings(
        model_id=resolve_model_id(mode, settings.model_id),
        prompt_version=settings.prompt_version,
        temperature=settings.temperature,
        max_generated_turns=settings.max_generated_turns,
        max_output_tokens=settings.max_output_tokens,
        random_seed=settings.random_seed,
    )
    judge_config = load_judge_config()
    judge_model_id = resolve_judge_model(mode, judge_config.judge_model_id)
    judge_config = JudgeConfig(
        judge_model_id, judge_config.rubric_version, judge_config.rubric_path
    )

    run_dir = output_root / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"experiment directory already exists: {run_dir}")
    simulations_dir = run_dir / "simulations"
    judge_dir = run_dir / "judge_inputs"
    events_path = run_dir / "logs" / "events.jsonl"
    calls_path = run_dir / "logs" / "model_calls.jsonl"
    for path in (simulations_dir, judge_dir, events_path.parent):
        path.mkdir(parents=True)

    # Version comes from Magnus's updater, which reads configs/updater.yaml.
    updater_version = TrustUpdater(MockLLMClient(()), settings.model_id).version
    prompt_sha = sha256((ROOT / "prompts" / f"{settings.prompt_version}.txt").read_bytes()).hexdigest()
    rubric_sha = sha256((ROOT / judge_config.rubric_path).read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "running",
        "mode": mode,
        "source": source,
        "git_commit": current_commit(),
        "limit": limit,
        "simulation": to_dict_settings(settings),
        "judge_model_id": judge_model_id,
        "rubric_version": judge_config.rubric_version,
        "updater_version": updater_version,
        "prompt_sha256": prompt_sha,
        "rubric_sha256": rubric_sha,
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)

    try:
        examples = load_examples(source)
        if limit is not None:
            examples = examples[:limit]
        # Personas are loaded once. Each arm copies them into fresh trust state internally.
        persona_configs = load_persona_configs()
        live = build_live_client() if mode == "live" else None
        judge = make_judge(mode, live, judge_model_id, settings.random_seed)
        runs: list[SimulationResult] = []
        scores: list[EscalationScore] = []

        for example in examples:
            # Magnus sees the opening only. real_outcome stays on the example until stage 5.
            personas = assign_personas(example.seed, persona_configs)
            for condition in CONDITIONS:
                try:
                    result = run_arm(
                        example,
                        personas,
                        condition,
                        settings,
                        updater_version,
                        experiment_id,
                        mode,
                        live,
                        calls_path,
                    )
                except Exception as exc:  # noqa: BLE001 -- one bad opening must not drop the rest.
                    append_jsonl(
                        events_path,
                        {
                            "stage": "simulation",
                            "conversation_id": example.seed.conversation_id,
                            "condition": condition,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue
                runs.append(result)
                write_json(simulations_dir / f"{result.run_id}.json", to_dict(result))
                append_jsonl(
                    events_path,
                    {
                        "stage": "simulation",
                        "run_id": result.run_id,
                        "conversation_id": result.conversation_id,
                        "condition": condition,
                        "status": result.status,
                        "error": result.run_metadata.error,
                    },
                )
                if result.status != "ok":
                    continue
                # Shadi's judge. Passing the transcript, not the LabeledExample, keeps the label out.
                score = score_arm(result, judge_config, judge, judge_dir, events_path)
                if score is not None:
                    scores.append(score)

        scores_path = run_dir / "scores.jsonl"
        scores_path.write_text("", encoding="utf-8")
        for score in scores:
            append_jsonl(scores_path, to_dict(score))
        # Labels enter here. on_incomplete keeps failed arms in the report instead of raising.
        report = evaluate_conditions(
            [to_dict(example) for example in examples],
            [to_dict(run) for run in runs],
            [to_dict(score) for score in scores],
            on_incomplete="report",
        )
        summary = to_schema_summary(report)
        write_json(run_dir / "evaluation_report.json", report)
        write_json(run_dir / "evaluation.json", to_dict(summary))
        manifest["status"] = "completed"
        manifest["n_examples"] = len(examples)
        manifest["n_runs"] = len(runs)
        manifest["n_scores"] = len(scores)
        manifest["conversation_ids"] = [example.seed.conversation_id for example in examples]
        manifest["coverage"] = report["coverage"]
        write_json(manifest_path, manifest)
    except Exception:
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise
    return run_dir


def to_dict_settings(settings: SimulationSettings) -> dict[str, Any]:
    """Plain settings for the manifest. This is not a schema dataclass."""
    return {
        "model_id": settings.model_id,
        "prompt_version": settings.prompt_version,
        "temperature": settings.temperature,
        "max_generated_turns": settings.max_generated_turns,
        "max_output_tokens": settings.max_output_tokens,
        "random_seed": settings.random_seed,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the mock and live commands from SPEC section 8."""
    parser = argparse.ArgumentParser(description="Run a paired static/dynamic persona experiment.")
    parser.add_argument("--mode", required=True, choices=("mock", "live"))
    parser.add_argument("--limit", type=int, default=None, help="maximum conversations to run")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--source", choices=("fixture", "processed"), default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.source is None:
        args.source = "fixture" if args.mode == "mock" else "processed"
    if args.experiment_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.experiment_id = f"{args.mode}-{stamp}"
    if not args.experiment_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("experiment-id may contain only letters, digits, hyphens, and underscores")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the experiment. Expected input errors print and return 1; artifacts stay on disk."""
    try:
        args = parse_args(argv)
        run_dir = run_experiment(
            mode=args.mode,
            limit=args.limit,
            experiment_id=args.experiment_id,
            source=args.source,
            output_root=args.output_root,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    summary = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    coverage = summary["coverage"]
    print(run_dir)
    print(
        f"completed_pairs={coverage['n_completed_pairs']} excluded={coverage['n_excluded']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
