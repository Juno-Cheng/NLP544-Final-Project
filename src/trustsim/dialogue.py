"""Conversation turns and the real openings the agents continue."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One comment: who wrote it and what it says."""

    speaker_id: str
    text: str


@dataclass(frozen=True)
class Seed:
    """The opening comments of a real conversation. A seed never carries its outcome label."""

    conversation_id: str
    opening: tuple[Turn, ...]

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speakers in order of first appearance."""
        return tuple(dict.fromkeys(turn.speaker_id for turn in self.opening))


def turns_payload(turns: Iterable[Turn]) -> list[dict[str, str]]:
    """Turns as plain dicts for a model prompt."""
    return [{"speaker": turn.speaker_id, "text": turn.text} for turn in turns]
