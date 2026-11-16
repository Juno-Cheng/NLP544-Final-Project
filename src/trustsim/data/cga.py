"""Cut CGA-WIKI conversations into agent-facing seeds and evaluation-only labels.

Works on ConvoKit conversation objects; scripts/prepare_data.py downloads the corpus.
"""

from collections.abc import Iterable
from typing import Any

SOURCE = "cga-wiki"
LABEL_DEFINITION = (
    "conversation_has_personal_attack: any comment in the conversation was judged "
    "by crowdsourced annotators to contain a personal attack"
)


def opening_turns(conversation: Any, max_turns: int) -> list[dict[str, Any]]:
    """First non-header comments in order, stopping before the first personal attack."""
    turns: list[dict[str, Any]] = []
    for utterance in conversation.get_chronological_utterance_list():
        if utterance.meta.get("is_section_header"):
            continue
        if utterance.meta.get("comment_has_personal_attack") or len(turns) == max_turns:
            break
        turns.append(
            {"turn_id": len(turns), "speaker_id": utterance.speaker.id, "text": utterance.text}
        )
    return turns


def real_turn_count(conversation: Any) -> int:
    """Number of comments in the real conversation, section headers excluded."""
    return sum(
        1
        for utterance in conversation.iter_utterances()
        if not utterance.meta.get("is_section_header")
    )


def build_records(
    conversations: Iterable[Any], max_turns: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Seed and label records for every conversation with a pre-attack opening, plus how many
    conversations had none (their very first comment was an attack)."""
    seeds: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    skipped = 0
    for conversation in conversations:
        turns = opening_turns(conversation, max_turns)
        if not turns:
            skipped += 1
            continue
        seeds.append({"conversation_id": conversation.id, "source": SOURCE, "opening_turns": turns})
        labels.append(
            {
                "conversation_id": conversation.id,
                "real_outcome": int(bool(conversation.meta["conversation_has_personal_attack"])),
                "label_definition": LABEL_DEFINITION,
                "split": conversation.meta.get("split"),
                "pair_id": conversation.meta.get("pair_id"),
                "real_turn_count": real_turn_count(conversation),
                "verified": bool(conversation.meta.get("verified")),
            }
        )
    return seeds, labels, skipped
