"""One simulated continuation of a real opening, in the static or the dynamic experiment."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from trustsim.config import DYNAMIC, EXPERIMENTS
from trustsim.dialogue import Seed, Turn
from trustsim.personas.cast import Cast, Persona, assign_personas
from trustsim.simulation.agent import Reply
from trustsim.simulation.trust import update_trust

LOGGER = logging.getLogger(__name__)
TURN_LIMIT = "turn_limit"
WALK_AWAY = "walk_away"


class Writer(Protocol):
    """Writes an agent's next reply."""

    def write(
        self, speaker_id: str, persona: Persona, mood: str, conversation: Sequence[Turn]
    ) -> Reply: ...


class Rater(Protocol):
    """Rates the last message of a conversation for its reader."""

    def rate(self, reader_id: str, conversation: Sequence[Turn]) -> float: ...


@dataclass(frozen=True)
class GeneratedTurn:
    """One generated comment, the writer's mood band, and both personas' trust after it.
    rating is the critic's score of this comment for its receiver (dynamic experiment only)."""

    speaker_id: str
    persona_id: str
    text: str
    leaving: bool
    mood: str
    rating: float | None
    trust: Mapping[str, float]


@dataclass(frozen=True)
class Simulation:
    """A finished continuation: the real opening, the generated turns, and how it ended.
    personas maps each real speaker to the persona that played it."""

    conversation_id: str
    experiment: str
    personas: Mapping[str, str]
    starting_trust: float
    opening: tuple[Turn, ...]
    turns: tuple[GeneratedTurn, ...]
    ended_by: str

    def transcript(self) -> list[Turn]:
        """The whole conversation: the real opening, then the generated turns."""
        return [*self.opening, *(Turn(turn.speaker_id, turn.text) for turn in self.turns)]

    def lowest_trust(self) -> float:
        """The lowest trust either persona reached during the conversation."""
        values = [value for turn in self.turns for value in turn.trust.values()]
        return min([self.starting_trust, *values])

    def to_record(self) -> dict[str, Any]:
        """Plain values for a JSONL record."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Simulation":
        """Rebuild a simulation saved with to_record."""
        return cls(
            conversation_id=record["conversation_id"],
            experiment=record["experiment"],
            personas=dict(record["personas"]),
            starting_trust=record["starting_trust"],
            opening=tuple(Turn(**turn) for turn in record["opening"]),
            turns=tuple(
                GeneratedTurn(**{**turn, "trust": dict(turn["trust"])}) for turn in record["turns"]
            ),
            ended_by=record["ended_by"],
        )


def simulate(
    seed: Seed,
    experiment: str,
    cast: Cast,
    writer: Writer,
    critic: Rater | None,
    max_turns: int,
) -> Simulation:
    """Continue a real opening for up to max_turns generated turns.

    Both experiments start every agent at the cast's starting trust. In the static experiment
    trust never moves and the critic is never called. In the dynamic experiment the critic rates
    each message for the agent receiving it, and only the receiver's trust changes. An agent that
    leaves ends the conversation with its leaving message.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}")
    if experiment == DYNAMIC and critic is None:
        raise ValueError("the dynamic experiment needs a critic")
    personas = assign_personas(seed)
    trust = {persona_id: cast.starting_trust for persona_id in personas.values()}
    conversation = list(seed.opening)
    speaker = _partner(personas, seed.opening[-1].speaker_id)
    turns: list[GeneratedTurn] = []
    for _ in range(max_turns):
        persona_id = personas[speaker]
        band, mood = cast.mood(persona_id, trust[persona_id])
        reply = writer.write(speaker, cast.personas[persona_id], mood, conversation)
        conversation.append(Turn(speaker, reply.text))
        receiver = _partner(personas, speaker)
        rating = None
        if experiment == DYNAMIC:
            rating = critic.rate(receiver, conversation)
            receiver_id = personas[receiver]
            trust[receiver_id] = update_trust(
                trust[receiver_id], rating, cast.personas[receiver_id], cast.starting_trust
            )
        turns.append(
            GeneratedTurn(speaker, persona_id, reply.text, reply.leaving, band, rating, dict(trust))
        )
        LOGGER.debug(
            "%s %s turn %d by %s (%s), rating %s, trust %s",
            seed.conversation_id,
            experiment,
            len(turns),
            persona_id,
            band,
            rating,
            trust,
        )
        if reply.leaving:
            break
        speaker = receiver
    return Simulation(
        conversation_id=seed.conversation_id,
        experiment=experiment,
        personas=personas,
        starting_trust=cast.starting_trust,
        opening=seed.opening,
        turns=tuple(turns),
        ended_by=WALK_AWAY if turns and turns[-1].leaving else TURN_LIMIT,
    )


def _partner(personas: Mapping[str, str], speaker_id: str) -> str:
    return next(other for other in personas if other != speaker_id)
