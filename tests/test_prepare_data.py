"""Tests for scripts/prepare_data.py.

Unit tests use fake conversations and run offline. Output tests check the real
exported files and are skipped until `python scripts/prepare_data.py` has been run.
"""

import json
from types import SimpleNamespace

import pytest

from prepare_data import PROCESSED_DIR, opening_turns, write_jsonl

SEEDS_PATH = PROCESSED_DIR / "seeds.jsonl"
LABELS_PATH = PROCESSED_DIR / "labels.jsonl"


def make_utt(speaker, text, header=False, attack=False):
    return SimpleNamespace(
        speaker=SimpleNamespace(id=speaker),
        text=text,
        meta={"is_section_header": header, "comment_has_personal_attack": attack},
    )


def make_convo(*utts):
    return SimpleNamespace(get_chronological_utterance_list=lambda: list(utts))


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_opening_skips_section_headers():
    convo = make_convo(make_utt("A", "Title", header=True), make_utt("A", "hi"), make_utt("B", "hello"))
    turns = opening_turns(convo, max_turns=3)
    assert [t["text"] for t in turns] == ["hi", "hello"]


def test_opening_stops_before_attack():
    convo = make_convo(make_utt("A", "hi"), make_utt("B", "you idiot", attack=True), make_utt("A", "after"))
    turns = opening_turns(convo, max_turns=3)
    assert [t["text"] for t in turns] == ["hi"]


def test_opening_respects_max_turns():
    convo = make_convo(*(make_utt("A" if i % 2 else "B", f"c{i}") for i in range(5)))
    assert len(opening_turns(convo, max_turns=3)) == 3


def test_opening_turn_format():
    turns = opening_turns(make_convo(make_utt("A", "hi"), make_utt("B", "yo")), max_turns=3)
    assert turns == [
        {"turn_id": 0, "speaker_id": "A", "text": "hi"},
        {"turn_id": 1, "speaker_id": "B", "text": "yo"},
    ]


def test_opening_empty_when_first_comment_is_attack():
    assert opening_turns(make_convo(make_utt("A", "rude", attack=True)), max_turns=3) == []


def test_write_jsonl_round_trip(tmp_path):
    records = [{"a": 1}, {"text": "café ▪"}]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path) == records


needs_outputs = pytest.mark.skipif(
    not (SEEDS_PATH.exists() and LABELS_PATH.exists()),
    reason="run scripts/prepare_data.py first",
)


@pytest.fixture(scope="module")
def seeds():
    return read_jsonl(SEEDS_PATH)


@pytest.fixture(scope="module")
def labels():
    return read_jsonl(LABELS_PATH)


@needs_outputs
def test_seeds_contain_no_label_fields(seeds):
    for seed in seeds:
        assert set(seed) == {"conversation_id", "source", "opening_turns"}


@needs_outputs
def test_seeds_and_labels_cover_same_conversations(seeds, labels):
    seed_ids = [s["conversation_id"] for s in seeds]
    label_ids = [l["conversation_id"] for l in labels]
    assert len(seed_ids) == len(set(seed_ids))
    assert set(seed_ids) == set(label_ids)


@needs_outputs
def test_every_seed_has_turns(seeds):
    assert all(seed["opening_turns"] for seed in seeds)


@needs_outputs
def test_labels_are_binary(labels):
    assert {l["real_outcome"] for l in labels} == {0, 1}


@needs_outputs
def test_pairs_stay_in_same_split(labels):
    by_id = {l["conversation_id"]: l for l in labels}
    for label in labels:
        partner = by_id.get(label["pair_id"])
        if partner:
            assert partner["split"] == label["split"]
