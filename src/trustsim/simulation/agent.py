"""Agent replies: a persona writes its next talk-page comment, or walks away."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trustsim.config import AGENT
from trustsim.dialogue import Turn, turns_payload
from trustsim.llm.client import AnswerFormatError, LLMClient, LLMRequest, ask, expect_keys
from trustsim.personas.cast import Persona

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "leaving": {"type": "boolean"},
    },
    "required": ["message", "leaving"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Reply:
    """An agent's next comment, and whether it leaves the discussion with it."""

    text: str
    leaving: bool


def parse_reply(answer: dict[str, Any]) -> Reply:
    """A reply from the agent's JSON answer."""
    expect_keys(answer, {"message", "leaving"})
    message, leaving = answer["message"], answer["leaving"]
    if not isinstance(message, str) or not message.strip():
        raise AnswerFormatError("message must be a non-empty string")
    if not isinstance(leaving, bool):
        raise AnswerFormatError("leaving must be true or false")
    return Reply(message.strip(), leaving)


def reply_payload(
    speaker_id: str, persona: Persona, mood: str, conversation: Sequence[Turn]
) -> dict[str, Any]:
    """What the agent sees: who it is, how it feels, and the conversation so far.
    It is never told its trust value or which experiment it is in."""
    return {
        "you": speaker_id,
        "persona": persona.profile(),
        "current_mood": mood,
        "conversation": turns_payload(conversation),
    }


class ReplyWriter:
    """Writes agent replies with the agent model."""

    def __init__(self, client: LLMClient, instructions: str, attempts: int) -> None:
        self._client = client
        self._instructions = instructions
        self._attempts = attempts

    def write(
        self, speaker_id: str, persona: Persona, mood: str, conversation: Sequence[Turn]
    ) -> Reply:
        """The speaker's next reply to the conversation so far."""
        request = LLMRequest(
            role=AGENT,
            system=self._instructions,
            user=json.dumps(
                reply_payload(speaker_id, persona, mood, conversation), ensure_ascii=False
            ),
            schema_name="agent_reply",
            schema=REPLY_SCHEMA,
        )
        return ask(self._client, request, parse_reply, self._attempts)
