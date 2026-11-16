"""The blind judge: reads a finished conversation and marks it 1 (personal attack) or 0."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trustsim.config import JUDGE
from trustsim.dialogue import Turn
from trustsim.llm.client import AnswerFormatError, LLMClient, LLMRequest, ask, expect_keys

WALK_AWAYS = ("none", "calm", "frustrated")
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "personal_attack": {"type": "boolean"},
        "walk_away": {"type": "string", "enum": list(WALK_AWAYS)},
        "evidence": {"type": "string"},
    },
    "required": ["personal_attack", "walk_away", "evidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verdict:
    """The judge's reading of one conversation. personal_attack is 1 when any comment attacks
    the other editor as a person, which is what the dataset's label of 1 means. walk_away is
    reported on its own and never turns a 0 into a 1."""

    personal_attack: int
    walk_away: str
    evidence: str


def parse_verdict(answer: dict[str, Any]) -> Verdict:
    """A verdict from the judge's JSON answer."""
    expect_keys(answer, {"personal_attack", "walk_away", "evidence"})
    attack, walk_away, evidence = answer["personal_attack"], answer["walk_away"], answer["evidence"]
    if not isinstance(attack, bool):
        raise AnswerFormatError("personal_attack must be true or false")
    if walk_away not in WALK_AWAYS:
        raise AnswerFormatError(f"walk_away must be one of {list(WALK_AWAYS)}, got {walk_away!r}")
    if not isinstance(evidence, str):
        raise AnswerFormatError("evidence must be a string")
    return Verdict(int(attack), walk_away, evidence.strip())


def judge_payload(transcript: Sequence[Turn]) -> dict[str, Any]:
    """Everything the judge sees: the conversation text, with speakers renamed Editor 1 and
    Editor 2 in order of appearance. No experiment, trust, persona, model, or label goes in."""
    if not transcript:
        raise ValueError("the judge needs a conversation to read")
    names: dict[str, str] = {}
    for turn in transcript:
        names.setdefault(turn.speaker_id, f"Editor {len(names) + 1}")
    return {
        "conversation": [
            {"speaker": names[turn.speaker_id], "text": turn.text} for turn in transcript
        ]
    }


class Judge:
    """Judges whole conversations with the judge model."""

    def __init__(self, client: LLMClient, instructions: str, attempts: int) -> None:
        self._client = client
        self._instructions = instructions
        self._attempts = attempts

    def judge(self, transcript: Sequence[Turn]) -> Verdict:
        """The verdict for a whole conversation, opening lines included."""
        request = LLMRequest(
            role=JUDGE,
            system=self._instructions,
            user=json.dumps(judge_payload(transcript), ensure_ascii=False),
            schema_name="judge_verdict",
            schema=VERDICT_SCHEMA,
        )
        return ask(self._client, request, parse_verdict, self._attempts)
