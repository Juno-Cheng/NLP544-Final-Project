"""A deterministic offline model for dry runs: no network and no API key."""

import hashlib
import json

from trustsim.config import AGENT, CRITIC, JUDGE
from trustsim.llm.client import LLMRequest, UsageMeter

MOCK_MESSAGES = (
    "Could you point me to the source for that change?",
    "The sentence did not match the cited reference, so I removed it.",
    "I can restore it if we add a reliable source.",
    "That is not what the policy page says. Read it before you revert again.",
    "Fair point, I reverted too quickly. Sorry about that.",
    "Let us check the policy page and adjust the wording together.",
)
MOCK_RATINGS = (0.1, 0.3, 0.5, 0.5, 0.7, 0.9)
MOCK_WALK_AWAYS = ("none", "none", "none", "calm", "frustrated")
LEAVE_EVERY = 17  # roughly one canned reply in 17 walks away, so dry runs cover early endings


class MockLLMClient:
    """Answers every role in its JSON format, picking canned answers by a hash of the prompt.
    Dry runs are repeatable, and dynamic trust still moves because ratings vary."""

    def __init__(self, meter: UsageMeter | None = None) -> None:
        self._meter = meter

    def complete(self, request: LLMRequest) -> str:
        """A canned answer in the format of the request's role."""
        pick = int.from_bytes(hashlib.sha256(request.user.encode("utf-8")).digest()[:4], "big")
        if self._meter is not None:
            self._meter.record(request.role, 0, 0)
        if request.role == AGENT:
            message = MOCK_MESSAGES[pick % len(MOCK_MESSAGES)]
            return json.dumps({"message": message, "leaving": pick % LEAVE_EVERY == 0})
        if request.role == CRITIC:
            return json.dumps({"rating": MOCK_RATINGS[pick % len(MOCK_RATINGS)]})
        if request.role == JUDGE:
            attack = pick % 3 == 0
            return json.dumps(
                {
                    "personal_attack": attack,
                    "walk_away": MOCK_WALK_AWAYS[pick % len(MOCK_WALK_AWAYS)],
                    "evidence": MOCK_MESSAGES[3] if attack else "",
                }
            )
        raise ValueError(f"the mock has no answers for role {request.role!r}")
