"""Tests for src/data_loader.py.

Most tests use a small fake dataset written to a temp folder, so they run offline.
The last test checks the real data with configs/data.yaml and is skipped until
`python scripts/prepare_data.py` has been run.

Fake dataset (pair = one attack + one civil conversation):
    p1-p4  test split, 2 speakers, verified       -> always kept
    p5     train split                            -> dropped by split filter
    p6     test, attack side has 3 speakers       -> attack dropped, so civil side loses its pair
    p7     test, attack side not verified         -> dropped when verified_only is on
"""

import json

import pytest

from src.data_loader import (
    complete_pairs,
    count_speakers,
    filter_by_speakers,
    filter_by_split,
    filter_verified,
    load_and_prepare,
    load_config,
    read_jsonl,
    sample_examples,
    seeds_only,
)

SEED_KEYS = {"conversation_id", "source", "opening_turns"}
EXAMPLE_KEYS = {"seed", "real_outcome", "label_definition", "split", "pair_id", "verified"}


def make_seed(cid, speakers):
    turns = [{"turn_id": i, "speaker_id": s, "text": f"comment {i}"} for i, s in enumerate(speakers)]
    return {"conversation_id": cid, "source": "cga-wiki", "opening_turns": turns}


def make_label(cid, outcome, split, pair_id, verified=True):
    return {
        "conversation_id": cid,
        "real_outcome": outcome,
        "label_definition": "test definition",
        "split": split,
        "pair_id": pair_id,
        "verified": verified,
    }


def make_pair(n, split="test", attack_speakers=("A", "B", "A"), verified=True):
    attack, civil = f"a{n}", f"c{n}"
    seeds = [make_seed(attack, attack_speakers), make_seed(civil, ("C", "D", "C"))]
    labels = [
        make_label(attack, 1, split, civil, verified),
        make_label(civil, 0, split, attack),
    ]
    return seeds, labels


@pytest.fixture
def fake_files(tmp_path):
    seeds, labels = [], []
    for n in range(1, 5):
        s, l = make_pair(n)
        seeds += s
        labels += l
    for s, l in [
        make_pair(5, split="train"),
        make_pair(6, attack_speakers=("A", "B", "E")),
        make_pair(7, verified=False),
    ]:
        seeds += s
        labels += l

    seeds_path, labels_path = tmp_path / "seeds.jsonl", tmp_path / "labels.jsonl"
    for path, rows in [(seeds_path, seeds), (labels_path, labels)]:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return seeds_path, labels_path


@pytest.fixture
def config(fake_files):
    seeds_path, labels_path = fake_files
    return {
        "seeds_path": str(seeds_path),
        "labels_path": str(labels_path),
        "min_speakers": 2,
        "max_speakers": 2,
        "splits": ["test"],
        "verified_only": True,
        "sample_size": None,
        "keep_pairs": True,
        "random_seed": 544,
    }


@pytest.fixture
def examples(config):
    return load_and_prepare("cga-wiki", {**config, "splits": ["test", "train"],
                                         "min_speakers": 1, "max_speakers": 10,
                                         "verified_only": False, "keep_pairs": False})


def ids(examples):
    return {e["seed"]["conversation_id"] for e in examples}


def test_read_jsonl_reads_every_line(fake_files):
    seeds_path, _ = fake_files
    assert len(read_jsonl(seeds_path)) == 14


def test_load_config_reads_real_yaml():
    config = load_config()
    assert config["min_speakers"] <= config["max_speakers"]


def test_count_speakers():
    assert count_speakers(make_seed("x", ("A", "B", "A"))) == 2
    assert count_speakers(make_seed("x", ("A", "B", "C"))) == 3


def test_examples_have_expected_fields(examples):
    assert len(examples) == 14
    for e in examples:
        assert set(e) == EXAMPLE_KEYS
        assert set(e["seed"]) == SEED_KEYS


def test_filter_by_speakers_drops_three_speaker_opening(examples):
    kept = filter_by_speakers(examples, min_speakers=2, max_speakers=2)
    assert "a6" not in ids(kept)
    assert "c6" in ids(kept)


def test_filter_by_split(examples):
    kept = filter_by_split(examples, ["test"])
    assert {"a5", "c5"}.isdisjoint(ids(kept))
    assert all(e["split"] == "test" for e in kept)


def test_filter_verified(examples):
    assert "a7" not in ids(filter_verified(examples))


def test_complete_pairs_drops_orphans(examples):
    kept = complete_pairs(filter_by_speakers(examples, 2, 2))
    assert "c6" not in ids(kept)
    for e in kept:
        assert e["pair_id"] in ids(kept)


def test_sample_keeps_pairs_and_balances_labels(examples):
    pool = complete_pairs(filter_by_split(examples, ["test"]))
    sample = sample_examples(pool, n=4, keep_pairs=True, random_seed=544)
    assert len(sample) == 4
    assert sorted(e["real_outcome"] for e in sample) == [0, 0, 1, 1]
    for e in sample:
        assert e["pair_id"] in ids(sample)


def test_sample_is_reproducible(examples):
    pool = complete_pairs(examples)
    first = sample_examples(pool, n=4, keep_pairs=True, random_seed=544)
    second = sample_examples(pool, n=4, keep_pairs=True, random_seed=544)
    assert ids(first) == ids(second)


def test_sample_none_returns_everything(examples):
    assert len(sample_examples(examples, n=None, keep_pairs=False, random_seed=544)) == len(examples)


def test_sample_larger_than_pool_returns_whole_pool(examples):
    pool = complete_pairs(examples)
    assert ids(sample_examples(pool, n=1000, keep_pairs=True, random_seed=544)) == ids(pool)


def test_load_and_prepare_applies_all_filters(config):
    result = load_and_prepare("cga-wiki", config)
    assert ids(result) == {"a1", "c1", "a2", "c2", "a3", "c3", "a4", "c4"}


def test_load_and_prepare_without_verified_filter_keeps_p7(config):
    result = load_and_prepare("cga-wiki", {**config, "verified_only": False})
    assert {"a7", "c7"} <= ids(result)


def test_load_and_prepare_respects_sample_size(config):
    result = load_and_prepare("cga-wiki", {**config, "sample_size": 4})
    assert len(result) == 4


def test_seeds_only_contains_no_label_information(config):
    seeds = seeds_only(load_and_prepare("cga-wiki", config))
    for seed in seeds:
        assert set(seed) == SEED_KEYS
    dumped = json.dumps(seeds)
    for leaked in ("real_outcome", "split", "pair_id", "verified", "label_definition"):
        assert leaked not in dumped


def test_seed_without_label_raises(config, tmp_path):
    labels = read_jsonl(config["labels_path"])[1:]
    labels_path = tmp_path / "missing.jsonl"
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8")
    with pytest.raises(ValueError):
        load_and_prepare("cga-wiki", {**config, "labels_path": str(labels_path)})


def test_real_data_with_project_config():
    config = load_config()
    try:
        result = load_and_prepare("cga-wiki", config)
    except FileNotFoundError:
        pytest.skip("run scripts/prepare_data.py first")
    if config["sample_size"]:
        assert len(result) == config["sample_size"]
    for e in result:
        assert e["split"] in config["splits"]
        assert config["min_speakers"] <= count_speakers(e["seed"]) <= config["max_speakers"]
    if config["keep_pairs"]:
        outcomes = [e["real_outcome"] for e in result]
        assert outcomes.count(0) == outcomes.count(1)
