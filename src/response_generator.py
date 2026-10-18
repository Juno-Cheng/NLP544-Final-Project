"""Persona-conditioned reply generation using the shared callable contract."""

import json
import math
import re
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.llm_client import GenerationError, LLMClient
from src.schemas import AgentState, DialogueTurn, GenerationConfig, from_dict, to_dict

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PersonaResponseGenerator:
    """Compose a client with a versioned prompt; never update the supplied actor state."""

    def __init__(
        self,
        client: LLMClient,
        *,
        on_generation: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._client = client
        self._on_generation = on_generation

    def __call__(
        self,
        history: list[DialogueTurn],
        actor: AgentState,
        config: GenerationConfig,
    ) -> DialogueTurn:
        """Allow the instance to be passed directly to run_simulation as its generator."""
        return self.generate_reply(history, actor, config)

    def generate_reply(
        self,
        history: list[DialogueTurn],
        actor: AgentState,
        config: GenerationConfig,
    ) -> DialogueTurn:
        """Return one reply with the next turn ID and the supplied actor's speaker ID."""
        # Rebuild to enforce the shared parser's strict types and isolate mutable fields.
        if not isinstance(actor, AgentState) or not isinstance(
            config, GenerationConfig
        ):
            raise TypeError("actor and config must use the shared schema types")
        actor = from_dict(AgentState, to_dict(actor))
        config = from_dict(GenerationConfig, to_dict(config))
        if not isinstance(history, list) or not history:
            raise ValueError("history must be a non-empty list of DialogueTurn")
        if not all(isinstance(turn, DialogueTurn) for turn in history):
            raise ValueError("history must contain DialogueTurn objects")
        turns = [from_dict(DialogueTurn, to_dict(turn)) for turn in history]
        ids = [turn.turn_id for turn in turns]
        if ids != sorted(set(ids)):
            raise ValueError("history turn IDs must be strictly increasing")
        speakers = {turn.speaker_id for turn in turns}
        if len(speakers) != 2 or actor.speaker_id not in speakers:
            raise ValueError("history must have two speakers including the actor")
        if any(not turn.text.strip() for turn in turns):
            raise ValueError("history cannot contain empty text")
        if not math.isfinite(config.temperature) or not 0 <= config.temperature <= 2:
            raise ValueError("temperature must be finite and between 0 and 2")
        if not re.fullmatch(r"reply_v[1-9][0-9]*", config.prompt_version):
            raise ValueError("prompt_version must have the form reply_vN")
        prompt_path = PROMPTS_DIR / f"{config.prompt_version}.txt"
        instructions = prompt_path.read_text(encoding="utf-8")
        if not instructions.strip():
            raise ValueError("reply prompt is empty")
        payload = json.dumps(
            {
                "actor": to_dict(actor),
                "history": [to_dict(turn) for turn in turns],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        completion = self._client.complete(instructions, payload, config)
        if not isinstance(completion.text, str) or not completion.text.strip():
            raise GenerationError("Model returned an empty or non-text reply")
        reply = DialogueTurn(ids[-1] + 1, actor.speaker_id, completion.text.strip())
        if self._on_generation is not None:
            self._on_generation(
                {
                    **deepcopy(completion.metadata),
                    "turn_id": reply.turn_id,
                    "speaker_id": reply.speaker_id,
                    "prompt_version": config.prompt_version,
                    "prompt_sha256": sha256(instructions.encode("utf-8")).hexdigest(),
                }
            )
        return reply
