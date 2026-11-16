"""Offline stand-ins for tests: scripted model answers and fake simulation roles."""

import json
from collections import deque
from collections.abc import Callable, Iterable, Sequence

from trustsim.dialogue import Turn
from trustsim.llm.client import LLMRequest
from trustsim.personas.cast import Persona
from trustsim.simulation.agent import Reply


def reply_answer(message: str = "Could you cite a source for that?", leaving: bool = False) -> str:
    """An agent answer in its JSON format."""
    return json.dumps({"message": message, "leaving": leaving})


def rating_answer(rating: float) -> str:
    """A critic answer in its JSON format."""
    return json.dumps({"rating": rating})


def verdict_answer(attack: bool = False, walk_away: str = "none", evidence: str = "") -> str:
    """A judge answer in its JSON format."""
    return json.dumps({"personal_attack": attack, "walk_away": walk_away, "evidence": evidence})


def default_answer(request: LLMRequest) -> str:
    """A valid, neutral answer for any role."""
    return {
        "agent": reply_answer(),
        "critic": rating_answer(0.5),
        "judge": verdict_answer(),
    }[request.role]


class ScriptedClient:
    """Answers from per-role queues, then from a fallback function; records every request."""

    def __init__(
        self,
        answers: dict[str, Iterable[str]] | None = None,
        fallback: Callable[[LLMRequest], str] = default_answer,
    ) -> None:
        self._queues = {role: deque(items) for role, items in (answers or {}).items()}
        self._fallback = fallback
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.requests.append(request)
        queue = self._queues.get(request.role)
        if queue:
            return queue.popleft()
        return self._fallback(request)

    def calls(self, role: str) -> list[LLMRequest]:
        """Requests made for one role."""
        return [request for request in self.requests if request.role == role]


class FakeWriter:
    """Returns scripted replies, then numbered neutral ones; records what each agent was given."""

    def __init__(self, replies: Iterable[Reply] = ()) -> None:
        self._replies = deque(replies)
        self.calls: list[dict] = []

    def write(
        self, speaker_id: str, persona: Persona, mood: str, conversation: Sequence[Turn]
    ) -> Reply:
        self.calls.append(
            {
                "speaker_id": speaker_id,
                "persona_id": persona.persona_id,
                "mood": mood,
                "conversation": list(conversation),
            }
        )
        if self._replies:
            return self._replies.popleft()
        return Reply(f"Comment number {len(self.calls)}.", False)


class FakeCritic:
    """Returns scripted ratings, then a fixed one; records who rated whose message."""

    def __init__(self, ratings: Iterable[float] = (), then: float = 0.5) -> None:
        self._ratings = deque(ratings)
        self._then = then
        self.calls: list[tuple[str, str]] = []

    def rate(self, reader_id: str, conversation: Sequence[Turn]) -> float:
        self.calls.append((reader_id, conversation[-1].speaker_id))
        return self._ratings.popleft() if self._ratings else self._then


class ForbiddenCritic:
    """A critic that must never be called."""

    def rate(self, reader_id: str, conversation: Sequence[Turn]) -> float:
        raise AssertionError("the static experiment must never call the critic")
