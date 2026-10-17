"""Download CGA-WIKI and export agent-facing seeds and evaluation-only labels.

Usage:
    python scripts/prepare_data.py [--opening-turns N]

Settings come from configs/data.yaml; --opening-turns overrides opening_turns for one run.

Outputs (both gitignored):
    data/processed/seeds.jsonl   ConversationSeed records; no labels, no later turns
    data/processed/labels.jsonl  conversation_id -> real_outcome, split, pair_id, ...

============================================
A seed (seeds.jsonl) is the start of a real conversation that the simulated agents continue. 
It holds the conversation's ID and its first few comments (3 by default), each with a turn number, 
the speaker's username and the text.  A seed is everything the agents are allowed to see.

A label (labels.jsonl) is the answer key: did the real conversation end in a personal attack? 
real_outcome is 1 for yes and 0 for no. The label record also stores the split (train/val/test), 
pair_id (the matching civil or attack conversation) and a few audit fields.

============================================

prepare_data.py is a one-time step. It downloads the raw corpus and writes seeds.jsonl and labels.jsonl. You only re-run it if you change how openings are cut.

"""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from convokit import Corpus, download

CORPUS_NAME = "conversations-gone-awry-corpus"
SOURCE = "cga-wiki"
LABEL_DEFINITION = (
    "conversation_has_personal_attack: any comment in the conversation was judged "
    "by crowdsourced annotators to contain a personal attack"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "data.yaml"

#================ Helper Functions==================
# Read the shared data settings from configs/data.yaml.
def load_config(path=CONFIG_PATH):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)



# Pick the first few real comments of a conversation to use as the seed opening.
def opening_turns(convo, max_turns):
    """First non-header comments in chronological order, stopping before any attack."""
    turns = []
    # Each utt is one comment, oldest first. Example:
    #   utt.speaker.id -> "Sirex98"
    #   utt.text       -> "I notice you reverted the move to Bill Chen..."
    #   utt.meta       -> {"is_section_header": False, "comment_has_personal_attack": False, ...}
    for utt in convo.get_chronological_utterance_list():
        if utt.meta.get("is_section_header"):
            continue
        if utt.meta.get("comment_has_personal_attack") or len(turns) == max_turns:
            break
        turns.append({"turn_id": len(turns), "speaker_id": utt.speaker.id, "text": utt.text})
    return turns


# Save a list of dicts to a file, one JSON object per line.
def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

#================ Main Function==================
# Download the corpus, build seeds and labels for every conversation, and save both.
def main():
    # 1. Read settings from data.yaml; --opening-turns overrides it for this run only
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening-turns", type=int, default=None)  # max real comments agents see before continuing; enough context without revealing the escalation
    args = parser.parse_args()
    max_turns = args.opening_turns if args.opening_turns is not None else config["opening_turns"]
    print(f"Using opening_turns={max_turns}")

    # 2. Make sure the output folders exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Download the corpus (skipped if already in data/raw) and load it
    corpus = Corpus(filename=download(CORPUS_NAME, data_dir=str(RAW_DIR)))
    corpus.print_summary_stats()

    # 4. Build one seed and one label per conversation
    seeds, labels, skipped = [], [], 0
    for convo in corpus.iter_conversations():
        turns = opening_turns(convo, max_turns)
        if not turns:
            skipped += 1
            continue
        # Seed: only what the agents may see
        seeds.append({"conversation_id": convo.id, "source": SOURCE, "opening_turns": turns})
        # Label: the real outcome, read only by evaluate.py
        labels.append({
            "conversation_id": convo.id,
            "real_outcome": int(bool(convo.meta["conversation_has_personal_attack"])),
            "label_definition": LABEL_DEFINITION,
            "split": convo.meta.get("split"),
            "pair_id": convo.meta.get("pair_id"),
            "verified": convo.meta.get("verified"),
            "annotation_year": convo.meta.get("annotation_year"),
            "page_id": convo.meta.get("page_id"),
        })

    # 5. Save seeds and labels to separate files
    write_jsonl(PROCESSED_DIR / "seeds.jsonl", seeds)
    write_jsonl(PROCESSED_DIR / "labels.jsonl", labels)

    # 6. Print a summary: totals and label balance per split
    counts = Counter((row["split"], row["real_outcome"]) for row in labels)
    print(f"\nWrote {len(seeds)} seeds, skipped {skipped} with no pre-attack opening")
    for (split, outcome), n in sorted(counts.items(), key=str):
        print(f"  split={split} real_outcome={outcome}: {n}")


if __name__ == "__main__":
    main()
