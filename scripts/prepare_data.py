"""Download CGA-WIKI once and write agent-facing seeds and evaluation-only labels.

Usage:
    python scripts/prepare_data.py [--opening-turns N]

Settings come from configs/data.yaml; --opening-turns overrides opening_turns for this run.
Needs ConvoKit (in requirements.txt). The corpus is cached in data/raw/.

Outputs (gitignored):
    data/processed/seeds.jsonl   openings only: no labels and no comments after the cut
    data/processed/labels.jsonl  real outcome, split, and attack/civil pair of each conversation
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from convokit import Corpus, download

from trustsim.config import load_data_settings
from trustsim.data.cga import build_records
from trustsim.storage import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
CORPUS_NAME = "conversations-gone-awry-corpus"
RAW_DIR = ROOT / "data" / "raw"


def main(argv: list[str] | None = None) -> int:
    """Build seeds.jsonl and labels.jsonl; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Prepare CGA-WIKI seeds and labels.")
    parser.add_argument("--opening-turns", type=int, help="real comments kept per opening")
    args = parser.parse_args(argv)
    settings = load_data_settings(ROOT / "configs" / "data.yaml", ROOT)
    max_turns = settings.opening_turns if args.opening_turns is None else args.opening_turns
    if max_turns < 1:
        parser.error("--opening-turns must be at least 1")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(filename=download(CORPUS_NAME, data_dir=str(RAW_DIR)))
    seeds, labels, skipped = build_records(corpus.iter_conversations(), max_turns)
    write_jsonl(settings.seeds_path, seeds)
    write_jsonl(settings.labels_path, labels)

    print(f"Wrote {len(seeds)} seeds with up to {max_turns} opening comments each.")
    print(f"Skipped {skipped} conversations whose first comment was already an attack.")
    counts = Counter((label["split"], label["real_outcome"]) for label in labels)
    for (split, outcome), count in sorted(counts.items(), key=str):
        print(f"  split={split} real_outcome={outcome}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
