from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from trustsim.data.cga import build_records, opening_turns
from trustsim.data.loader import join_examples, load_examples, parse_label, sample_examples


@dataclass
class FakeUtterance:
    speaker_id: str
    text: str
    header: bool = False
    attack: bool = False

    @property
    def speaker(self) -> SimpleNamespace:
        return SimpleNamespace(id=self.speaker_id)

    @property
    def meta(self) -> dict:
        return {"is_section_header": self.header, "comment_has_personal_attack": self.attack}


@dataclass
class FakeConversation:
    id: str
    utterances: list[FakeUtterance]
    meta: dict = field(default_factory=dict)

    def get_chronological_utterance_list(self) -> list[FakeUtterance]:
        return list(self.utterances)

    def iter_utterances(self):
        return iter(self.utterances)


def conversation(conversation_id: str, utterances: list[FakeUtterance], attack: bool):
    meta = {"conversation_has_personal_attack": attack, "split": "test", "pair_id": "p"}
    return FakeConversation(conversation_id, utterances, meta)


def test_opening_stops_before_the_first_attack():
    convo = conversation(
        "c",
        [
            FakeUtterance("a", "Header", header=True),
            FakeUtterance("a", "First."),
            FakeUtterance("b", "You clearly cannot read.", attack=True),
            FakeUtterance("a", "Third."),
        ],
        attack=True,
    )
    assert opening_turns(convo, max_turns=3) == [
        {"turn_id": 0, "speaker_id": "a", "text": "First."}
    ]


def test_opening_stops_at_max_turns():
    utterances = [FakeUtterance(speaker, f"Comment {i}.") for i, speaker in enumerate("ababab")]
    assert len(opening_turns(conversation("c", utterances, attack=False), max_turns=3)) == 3


def test_records_keep_labels_out_of_seeds_and_skip_attacks_in_the_first_comment():
    kept = conversation(
        "kept",
        [
            FakeUtterance("a", "Hi."),
            FakeUtterance("b", "Hello."),
            FakeUtterance("a", "Rude!", attack=True),
        ],
        attack=True,
    )
    skipped = conversation("skipped", [FakeUtterance("a", "Rude!", attack=True)], attack=True)
    seeds, labels, skipped_count = build_records([kept, skipped], max_turns=3)
    assert skipped_count == 1
    assert [seed["conversation_id"] for seed in seeds] == ["kept"]
    assert set(seeds[0]) == {"conversation_id", "source", "opening_turns"}
    assert labels[0]["real_outcome"] == 1
    assert labels[0]["real_turn_count"] == 3


def test_filters_keep_only_complete_two_speaker_test_pairs(examples):
    ids = {example.seed.conversation_id for example in examples}
    assert ids == {"t1-attack", "t1-civil", "t2-attack", "t2-civil", "t3-attack", "t3-civil"}


def test_sample_holds_whole_pairs_with_equal_labels(examples):
    sample = sample_examples(examples, size=4, keep_pairs=True, random_seed=1)
    outcomes = [example.label.real_outcome for example in sample]
    assert len(sample) == 4
    assert outcomes.count(1) == outcomes.count(0) == 2
    ids = {example.seed.conversation_id for example in sample}
    assert all(example.label.pair_id in ids for example in sample)


def test_sampling_is_reproducible(examples):
    first = sample_examples(examples, size=4, keep_pairs=True, random_seed=7)
    second = sample_examples(examples, size=4, keep_pairs=True, random_seed=7)
    assert first == second


def test_verified_only_drops_unverified_pairs(data_settings):
    examples = load_examples(replace(data_settings, verified_only=True))
    assert {example.seed.conversation_id for example in examples} == {
        "t1-attack",
        "t1-civil",
        "t3-attack",
        "t3-civil",
    }


def test_seed_without_a_label_is_an_error():
    seed = {"conversation_id": "x", "opening_turns": [{"speaker_id": "a", "text": "Hi."}]}
    with pytest.raises(ValueError, match="no label"):
        join_examples([seed], [])


def test_real_outcome_must_be_zero_or_one():
    with pytest.raises(ValueError, match="0 or 1"):
        parse_label({"conversation_id": "x", "real_outcome": True, "split": "test"})


def test_missing_prepared_data_points_to_the_prepare_script(data_settings, tmp_path):
    settings = replace(data_settings, seeds_path=tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError, match=r"prepare_data\.py"):
        load_examples(settings)
