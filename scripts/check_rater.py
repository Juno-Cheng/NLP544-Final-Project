"""One-off check that the trust rater scores real attacks below civil comments (DECISIONS.md F).

Rates N attack comments and N comments from civil conversations in the CGA-WIKI train
split, each with up to 8 preceding comments as context, keeping only two-speaker windows.
Labels are used only here, to validate the rater; nothing from this script feeds the
experiment, which uses the test split.

Usage:
    export OPENAI_API_KEY=... OPENAI_MODEL=...
    python scripts/check_rater.py [--n 10]

Needs convokit, the openai SDK and src/llm_client.py (Arya's branch).
Pass = every attack comment scores below the civil mean; exits 1 on fail.
"""

import argparse
import os
import random
import sys
from pathlib import Path

from convokit import Corpus, download
from openai import OpenAI

from prepare_data import CORPUS_NAME, RAW_DIR, load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import OpenAILLMClient  # noqa: E402
from src.schemas import DialogueTurn  # noqa: E402
from src.state_updater import TrustUpdater  # noqa: E402

SPLIT = "train"
CONTEXT_TURNS = 8


def candidates(corpus, attack: bool) -> list[tuple[str, list[DialogueTurn]]]:
    """(reader_id, history ending in the rated comment) for every eligible train-split comment."""
    found = []
    for convo in corpus.iter_conversations():
        if convo.meta.get("split") != SPLIT:
            continue
        if not attack and convo.meta["conversation_has_personal_attack"]:
            continue
        utts = [utt for utt in convo.get_chronological_utterance_list() if not utt.meta.get("is_section_header")]
        for i, utt in enumerate(utts):
            if bool(utt.meta.get("comment_has_personal_attack")) != attack:
                continue
            window = utts[max(0, i - CONTEXT_TURNS) : i + 1]
            # Match the experiment: two speakers, so the reader is the one the comment answers.
            if len({u.speaker.id for u in window}) != 2:
                continue
            # The reader is the latest earlier speaker who is not the comment's author.
            readers = [u.speaker.id for u in window[:-1] if u.speaker.id != utt.speaker.id]
            if not readers:
                continue
            history = [DialogueTurn(n, u.speaker.id, u.text) for n, u in enumerate(window)]
            found.append((readers[-1], history))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)  # comments rated per class
    args = parser.parse_args()

    corpus = Corpus(filename=download(CORPUS_NAME, data_dir=str(RAW_DIR)))
    rng = random.Random(load_config()["random_seed"])
    updater = TrustUpdater(OpenAILLMClient(OpenAI()), os.environ["OPENAI_MODEL"])

    ratings = {}
    for kind, attack in (("attack", True), ("civil", False)):
        picked = rng.sample(candidates(corpus, attack), args.n)
        ratings[kind] = [updater.rate(reader, history[-1], history) for reader, history in picked]
        print(f"{kind}: {[round(r, 2) for r in ratings[kind]]}")

    civil_mean = sum(ratings["civil"]) / len(ratings["civil"])
    passed = max(ratings["attack"]) < civil_mean
    print(f"civil mean {civil_mean:.2f}, highest attack {max(ratings['attack']):.2f}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
