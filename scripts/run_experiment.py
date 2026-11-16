"""Run the static and dynamic experiments, judge every conversation blind, and compare the
verdicts with the dataset's real outcomes.

Usage:
    python scripts/run_experiment.py --run-name dry-run --mock
    python scripts/run_experiment.py --run-name pilot-20
    python scripts/run_experiment.py --run-name pilot-20 --provider gemini --model <model id>

Settings: configs/data.yaml, configs/personas.yaml, configs/experiment.yaml. A live run reads
its API key from the environment variable named for the provider in configs/experiment.yaml
(OPENAI_API_KEY or GEMINI_API_KEY); --mock needs no key and makes no network calls.

Running again with the same --run-name resumes: finished conversations are kept and failed
ones are retried. Outputs go to outputs/<run-name>/: run.json, simulations.jsonl,
verdicts.jsonl, metrics.json, results.csv, summary.md, and run.log.
"""

import argparse
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path

from trustsim.config import (
    EXPERIMENTS,
    MOCK_PROVIDER,
    ConfigError,
    check_sample_size,
    load_data_settings,
    load_experiment_settings,
    with_overrides,
)
from trustsim.data.loader import load_examples
from trustsim.llm.client import LLMSetupError, UsageMeter
from trustsim.llm.factory import build_clients
from trustsim.personas.cast import load_cast
from trustsim.pipeline import SUMMARY, RunMismatchError, build_roles, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
LOGGER = logging.getLogger("run_experiment")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Run the static and dynamic trust experiments.")
    parser.add_argument("--run-name", required=True, help="folder name under outputs/")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mock", action="store_true", help="use the offline mock model")
    source.add_argument("--provider", help="provider from configs/experiment.yaml, e.g. gemini")
    parser.add_argument("--model", help="model id for every role (agents, critic, judge)")
    parser.add_argument("--sample-size", type=int, help="conversations to use (even with pairs)")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, help="default: both")
    parser.add_argument("--workers", type=int, help="conversations simulated in parallel")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help="also log every generated turn")
    args = parser.parse_args(argv)
    if not RUN_NAME.fullmatch(args.run_name):
        parser.error("--run-name may use letters, digits, '.', '_' and '-' only")
    return args


def configure_logging(log_path: Path, verbose: bool) -> None:
    """Log to the terminal and to the run's log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; returns the process exit code."""
    args = parse_args(argv)
    run_dir = args.output_dir / args.run_name
    configure_logging(run_dir / "run.log", args.verbose)
    try:
        data = load_data_settings(CONFIG_DIR / "data.yaml", ROOT)
        if args.sample_size is not None:
            check_sample_size(args.sample_size, data.keep_pairs)
            data = replace(data, sample_size=args.sample_size)
        cast = load_cast(CONFIG_DIR / "personas.yaml")
        settings = with_overrides(
            load_experiment_settings(CONFIG_DIR / "experiment.yaml", ROOT),
            provider=MOCK_PROVIDER if args.mock else args.provider,
            model=args.model,
            experiments=args.experiments,
            workers=args.workers,
        )
        examples = load_examples(data)
        meter = UsageMeter()
        roles, prompts = build_roles(build_clients(settings.llm, meter), settings)
        run_pipeline(run_dir, examples, cast, settings, roles, prompts, meter)
    except (ConfigError, FileNotFoundError, RunMismatchError, LLMSetupError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print((run_dir / SUMMARY).read_text(encoding="utf-8"))
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
