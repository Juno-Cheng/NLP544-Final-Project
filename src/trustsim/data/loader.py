"""Load prepared seeds and labels, filter them, and sample whole attack/civil pairs."""

import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustsim.config import DataSettings
from trustsim.dialogue import Seed, Turn
from trustsim.storage import read_jsonl

LOGGER = logging.getLogger(__name__)
SPEAKERS_PER_CONVERSATION = 2  # each real speaker becomes one agent: persona A or persona B


@dataclass(frozen=True)
class Label:
    """A conversation's real outcome and sampling details. Only evaluation reads real_outcome."""

    conversation_id: str
    real_outcome: int
    split: str
    pair_id: str | None
    verified: bool


@dataclass(frozen=True)
class Example:
    """A seed and its label, kept apart so the label can be held back until evaluation."""

    seed: Seed
    label: Label


def parse_seed(record: dict[str, Any]) -> Seed:
    """A seed from one seeds.jsonl record."""
    turns = tuple(Turn(turn["speaker_id"], turn["text"]) for turn in record["opening_turns"])
    if not turns:
        raise ValueError(f"seed {record['conversation_id']} has no opening turns")
    return Seed(record["conversation_id"], turns)


def parse_label(record: dict[str, Any]) -> Label:
    """A label from one labels.jsonl record."""
    outcome = record["real_outcome"]
    if isinstance(outcome, bool) or outcome not in (0, 1):
        raise ValueError(f"label {record['conversation_id']}: real_outcome must be 0 or 1")
    return Label(
        conversation_id=record["conversation_id"],
        real_outcome=outcome,
        split=record["split"],
        pair_id=record.get("pair_id"),
        verified=bool(record.get("verified")),
    )


def join_examples(
    seed_records: Iterable[dict[str, Any]], label_records: Iterable[dict[str, Any]]
) -> list[Example]:
    """Every seed with its label. A seed without a label is an error."""
    labels = {label.conversation_id: label for label in map(parse_label, label_records)}
    examples = []
    for seed in map(parse_seed, seed_records):
        if seed.conversation_id not in labels:
            raise ValueError(f"no label for conversation {seed.conversation_id}")
        examples.append(Example(seed, labels[seed.conversation_id]))
    return examples


def passes_filters(example: Example, settings: DataSettings) -> bool:
    """Whether a conversation fits the split, verification, speaker, and opening-length rules."""
    return (
        example.label.split in settings.splits
        and (example.label.verified or not settings.verified_only)
        and len(example.seed.speakers) == SPEAKERS_PER_CONVERSATION
        and len(example.seed.opening) >= settings.min_opening_turns
    )


def complete_pairs(examples: list[Example]) -> list[Example]:
    """Examples whose attack/civil partner is also present."""
    present = {example.seed.conversation_id for example in examples}
    return [example for example in examples if example.label.pair_id in present]


def sample_examples(
    examples: list[Example], size: int | None, keep_pairs: bool, random_seed: int
) -> list[Example]:
    """A reproducible sample. With keep_pairs it is size // 2 whole attack/civil pairs;
    a size of None keeps everything."""
    rng = random.Random(random_seed)
    by_id = {example.seed.conversation_id: example for example in examples}
    if not keep_pairs:
        ids = sorted(by_id)
        if size is not None:
            ids = sorted(rng.sample(ids, min(size, len(ids))))
        return [by_id[conversation_id] for conversation_id in ids]
    pairs = sorted(
        {
            tuple(sorted((conversation_id, example.label.pair_id)))
            for conversation_id, example in by_id.items()
            if example.label.pair_id in by_id
        }
    )
    if size is not None:
        pairs = sorted(rng.sample(pairs, min(size // 2, len(pairs))))
    return [by_id[conversation_id] for pair in pairs for conversation_id in pair]


def select_examples(examples: list[Example], settings: DataSettings) -> list[Example]:
    """The conversations a run uses: filtered, then sampled."""
    kept = [example for example in examples if passes_filters(example, settings)]
    if settings.keep_pairs:
        kept = complete_pairs(kept)
    return sample_examples(kept, settings.sample_size, settings.keep_pairs, settings.random_seed)


def load_examples(settings: DataSettings) -> list[Example]:
    """The run's conversations, read from the prepared seeds and labels files."""
    for path in (settings.seeds_path, settings.labels_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} is missing; run python scripts/prepare_data.py first")
    examples = join_examples(read_jsonl(settings.seeds_path), read_jsonl(settings.labels_path))
    selected = select_examples(examples, settings)
    attacks = sum(example.label.real_outcome for example in selected)
    LOGGER.info(
        "selected %d of %d conversations (%d attack, %d civil)",
        len(selected),
        len(examples),
        attacks,
        len(selected) - attacks,
    )
    return selected
