"""The LLM critic of the dynamic experiment: it rates the message an agent just received."""

import json
import math
from collections.abc import Sequence
from typing import Any

from trustsim.config import CRITIC
from trustsim.dialogue import Turn, turns_payload
from trustsim.llm.client import AnswerFormatError, LLMClient, LLMRequest, ask, expect_keys

RATING_SCHEMA = {
    "type": "object",
    "properties": {"rating": {"type": "number"}},
    "required": ["rating"],
    "additionalProperties": False,
}


def parse_rating(answer: dict[str, Any]) -> float:
    """The rating from the critic's JSON answer, a number from 0 to 1."""
    expect_keys(answer, {"rating"})
    rating = answer["rating"]
    if isinstance(rating, bool) or not isinstance(rating, int | float) or not math.isfinite(rating):
        raise AnswerFormatError(f"rating must be a number, got {rating!r}")
    if not 0 <= rating <= 1:
        raise AnswerFormatError(f"rating must be between 0 and 1, got {rating}")
    return float(rating)


def rating_payload(reader_id: str, conversation: Sequence[Turn]) -> dict[str, Any]:
    """What the critic sees: who received the last message, and the conversation so far.
    The critic is never shown anyone's trust, so trust cannot feed back into its ratings."""
    if not conversation:
        raise ValueError("the critic needs a message to rate")
    if conversation[-1].speaker_id == reader_id:
        raise ValueError(f"{reader_id} wrote the last message, so cannot be its reader")
    return {"reader": reader_id, "conversation": turns_payload(conversation)}


class Critic:
    """Rates the last message of a conversation for its reader, with the critic model."""

    def __init__(self, client: LLMClient, instructions: str, attempts: int) -> None:
        self._client = client
        self._instructions = instructions
        self._attempts = attempts

    def rate(self, reader_id: str, conversation: Sequence[Turn]) -> float:
        """How the reader would take the last message: 0 hostile, 0.5 neutral, 1 repair.
        Earlier turns are context only; they are not scored again."""
        request = LLMRequest(
            role=CRITIC,
            system=self._instructions,
            user=json.dumps(rating_payload(reader_id, conversation), ensure_ascii=False),
            schema_name="critic_rating",
            schema=RATING_SCHEMA,
        )
        return ask(self._client, request, parse_rating, self._attempts)
