"""Download CGA-WIKI and export agent-facing seeds and evaluation-only labels.

Usage:
    python scripts/prepare_data.py [--opening-turns 3]

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

#================ Helper Functions==================
# Pick the first few real comments of a conversation to use as the seed opening.
def opening_turns(convo, max_turns):
    """First non-header comments in chronological order, stopping before any attack."""
    turns = []
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

if __name__ == "__main__":
    main()
