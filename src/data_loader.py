"""Load prepared CGA-WIKI seeds and labels for an experiment run.

Runs every time the experiment does. Reads data/processed/seeds.jsonl and
labels.jsonl (written once by scripts/prepare_data.py), applies filters
(e.g. "only 2-speaker conversations, test split, 20 examples"), and hands
clean records to the rest of the pipeline.

Usage:
    examples = load_and_prepare()        # LabeledExamples, for run_experiment / evaluate
    seeds = seeds_only(examples)         # ConversationSeeds only, safe to give the simulator
"""

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data.yaml"


#================ Helper Functions==================
# Read the shared data settings from configs/data.yaml.
def load_config(path: Path = CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# Read a JSONL file into a list of dicts.
def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Count distinct speakers in a seed's opening turns.
def count_speakers(seed: dict) -> int:
    return len({turn["speaker_id"] for turn in seed["opening_turns"]}) # count the number of distinct speakers in the seed


# Match every seed to its label by conversation_id; every seed must have one.
def join_seeds_and_labels(seeds: list[dict], labels: list[dict]) -> list[dict]:
    labels_by_id = {row["conversation_id"]: row for row in labels} # Dictionary mapping conversation IDs to their corresponding label rows.
    examples = []
    for seed in seeds:
        label = labels_by_id.get(seed["conversation_id"]) # get the label for the conversation
        if label is None:
            raise ValueError(f"no label for conversation {seed['conversation_id']}")
        examples.append({ # append the seed and label to the examples list
            "seed": seed,
            "real_outcome": label["real_outcome"],
            "label_definition": label["label_definition"],
            "split": label["split"],
            "pair_id": label["pair_id"],
            "verified": bool(label.get("verified")),
        })
    return examples


#================ Filters ==================
# Each filter takes a list of examples and returns the ones that pass.

# Keep openings with between min_speakers and max_speakers distinct speakers.
def filter_by_speakers(examples: list[dict], min_speakers: int, max_speakers: int) -> list[dict]:
    return [ex for ex in examples if min_speakers <= count_speakers(ex["seed"]) <= max_speakers]


# Keep examples from the given source splits (train / val / test).
def filter_by_split(examples: list[dict], splits: list[str]) -> list[dict]:
    return [ex for ex in examples if ex["split"] in splits]


# Keep examples whose label was double-checked by an internal annotator.
def filter_verified(examples: list[dict]) -> list[dict]:
    return [ex for ex in examples if ex["verified"]]


# Drop examples whose paired conversation was filtered out, so pairs stay whole.
def complete_pairs(examples: list[dict]) -> list[dict]:
    present = {ex["seed"]["conversation_id"] for ex in examples}
    return [ex for ex in examples if ex["pair_id"] in present]


#================ Sampling ==================
# Pick n examples reproducibly, as whole attack/civil pairs or individually.
def sample_examples(examples: list[dict], n: int | None, keep_pairs: bool, random_seed: int) -> list[dict]:
    """n=None returns everything; n larger than the pool returns the whole pool."""
    rng = random.Random(random_seed)
    by_id = {ex["seed"]["conversation_id"]: ex for ex in examples}

    if not keep_pairs:
        ids = sorted(by_id)
        if n is not None:
            ids = sorted(rng.sample(ids, min(n, len(ids))))
        return [by_id[cid] for cid in ids]

    # Sample whole pairs (n // 2 of them) so attack and civil counts stay equal.
    pairs = sorted({
        tuple(sorted((cid, ex["pair_id"])))
        for cid, ex in by_id.items()
        if ex["pair_id"] in by_id
    })
    if n is not None:
        pairs = sorted(rng.sample(pairs, min(n // 2, len(pairs))))
    return [by_id[cid] for pair in pairs for cid in pair]


#================ Main Functions==================
# Load seeds + labels, join them, filter and sample according to data.yaml.
def load_and_prepare(source: str = "cga-wiki", config: dict | None = None) -> list[dict]:
    """Return LabeledExample dicts: {seed, real_outcome, label_definition, split, pair_id, verified}."""
    config = config or load_config()
    if config.get("source", source) != source:
        raise ValueError(f"source {source!r} does not match config source {config['source']!r}")

    # 1. Read both files and join each seed with its label
    seeds = read_jsonl(ROOT / config["seeds_path"])
    labels = read_jsonl(ROOT / config["labels_path"])
    examples = join_seeds_and_labels(seeds, labels)

    # 2. Apply the split / verified / speaker filters
    examples = filter_by_split(examples, config["splits"])
    if config["verified_only"]:
        examples = filter_verified(examples)
    examples = filter_by_speakers(examples, config["min_speakers"], config["max_speakers"])

    # 3. Drop conversations whose pair partner was filtered out
    if config["keep_pairs"]:
        examples = complete_pairs(examples)

    # 4. Sample down to sample_size, keeping pairs together if configured
    return sample_examples(examples, config["sample_size"], config["keep_pairs"], config["random_seed"])


# Strip labels: the simulator and judge only ever receive these seeds.
def seeds_only(examples: list[dict]) -> list[dict]:
    return [example["seed"] for example in examples]
