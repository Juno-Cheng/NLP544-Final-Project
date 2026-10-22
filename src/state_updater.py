"""Static and dynamic trust updates for an agent that just observed its partner's turn.

Static keeps trust fixed and makes no model call. Dynamic asks a rater model how
cooperative the observed turn is (rating in [0, 1], 0.5 = neutral) and shifts trust by it:
    trust' = clamp(trust + alpha * (rating - 0.5))
with alpha = alpha_down below 0.5 and alpha_up otherwise, so trust falls faster than it
recovers and a neutral message leaves trust unchanged. Each speaker can have its own
alphas (a persona's trust_sensitivity); otherwise configs/updater.yaml supplies them.

Usage:
    updater = TrustUpdater(client, model_id=os.environ["OPENAI_MODEL"],
                           sensitivity=sensitivity_by_speaker(personas), on_rating=log)
    start = updater.replay_opening(initialize_personas(personas), seed.opening_turns)
    result = updater(state, observed_turn, visible_history, UpdaterConfig("dynamic", updater.version))

The client is anything with complete(instructions, payload, config) returning an
object with .text and .metadata, i.e. the LLMClient protocol from src/llm_client.py.
"""

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from src.schemas import (
    AgentState,
    DialogueTurn,
    GenerationConfig,
    PersonaConfig,
    StateUpdateResult,
    UpdaterConfig,
    to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "updater.yaml"
PROMPTS_DIR = ROOT / "prompts"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the trust-update settings from configs/updater.yaml."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_rating(text: str) -> float:
    """Return the rater's reply as a float in [0, 1]; anything else is an error, never a default."""
    try:
        rating = float(text.strip())
    except ValueError:
        raise ValueError(f"rater returned a non-numeric reply: {text!r}") from None
    if not math.isfinite(rating) or not 0.0 <= rating <= 1.0:
        raise ValueError(f"rater rating must be in [0, 1], got {text!r}")
    return rating


def _check_alphas(alphas: dict[str, float], where: str) -> tuple[float, float]:
    """Return (alpha_up, alpha_down), each required to be in (0, 1]."""
    checked = []
    for name in ("alpha_up", "alpha_down"):
        alpha = float(alphas[name])
        if not 0.0 < alpha <= 1.0:  # also rejects nan, which would silently zero every trust
            raise ValueError(f"{where} {name} must be in (0, 1], got {alpha}")
        checked.append(alpha)
    return checked[0], checked[1]


def sensitivity_by_speaker(personas: dict[str, PersonaConfig]) -> dict[str, dict[str, float] | None]:
    """Each speaker's trust_sensitivity, keyed like the speaker-keyed personas from assign_personas."""
    return {speaker_id: persona.trust_sensitivity for speaker_id, persona in personas.items()}


class TrustUpdater:
    """Callable StateUpdater for run_simulation; the client is only called in the dynamic condition."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        settings: dict | None = None,
        *,
        sensitivity: dict[str, dict[str, float] | None] | None = None,
        on_rating: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """sensitivity maps speaker_id to its alphas; None for a speaker means the config defaults."""
        settings = load_config() if settings is None else settings
        self.version = settings["updater_version"]
        self.replay_opening_enabled = bool(settings.get("replay_opening", False))
        self.alpha_up, self.alpha_down = _check_alphas(settings, "default")
        self.sensitivity = None
        if sensitivity is not None:
            self.sensitivity = {
                speaker_id: (self.alpha_up, self.alpha_down) if alphas is None else _check_alphas(alphas, speaker_id)
                for speaker_id, alphas in sensitivity.items()
            }
        rater = settings["rater"]
        self.rater_config = GenerationConfig(
            model_id, rater["temperature"], rater["max_output_tokens"], rater["prompt_version"], rater["random_seed"]
        )
        self.instructions = (PROMPTS_DIR / f"{rater['prompt_version']}.txt").read_text(encoding="utf-8")
        self.client = client
        self.on_rating = on_rating

    def update_state(
        self,
        current_state: AgentState,
        observed_turn: DialogueTurn,
        visible_history: list[DialogueTurn],
        config: UpdaterConfig,
    ) -> StateUpdateResult:
        """Return the receiver's state after reading observed_turn."""
        if config.updater_version != self.version:
            raise ValueError(f"config asks for {config.updater_version!r} but configs/updater.yaml is {self.version!r}")
        # Trust belongs to the receiver and may only use turns it has already seen.
        if observed_turn.speaker_id == current_state.speaker_id:
            raise ValueError(f"{current_state.speaker_id} cannot update trust from its own turn")
        if not visible_history or visible_history[-1] != observed_turn:
            raise ValueError("observed_turn must be the latest turn of visible_history")

        if config.condition == "static":
            return StateUpdateResult(current_state, current_state.trust)
        alpha_up, alpha_down = self.alphas_for(current_state.speaker_id)
        rating = self.rate(current_state.speaker_id, observed_turn, visible_history)
        alpha = alpha_down if rating < 0.5 else alpha_up
        # Rounding keeps float error from dropping trust just below a band edge (0.3999... vs 0.4).
        trust = round(min(1.0, max(0.0, current_state.trust + alpha * (rating - 0.5))), 6)
        return StateUpdateResult(replace(current_state, trust=trust), current_state.trust)

    def alphas_for(self, speaker_id: str) -> tuple[float, float]:
        """(alpha_up, alpha_down) for this speaker; an unknown speaker is an error once sensitivity is set."""
        if self.sensitivity is None:
            return self.alpha_up, self.alpha_down
        if speaker_id not in self.sensitivity:
            raise ValueError(f"no trust sensitivity for speaker {speaker_id!r}")
        return self.sensitivity[speaker_id]

    def replay_opening(self, states: dict[str, AgentState], opening_turns: list[DialogueTurn]) -> dict[str, AgentState]:
        """Rate each real opening turn as its receiver would and return the post-opening states.

        Every turn is read by the other speaker, using only the turns up to and including it.
        Callers give the result to both arms, so static and dynamic still start identical.
        """
        if len(states) != 2:
            raise ValueError("opening replay needs exactly two speakers")
        states = dict(states)
        config = UpdaterConfig("dynamic", self.version)
        for index, turn in enumerate(opening_turns):
            if turn.speaker_id not in states:
                raise ValueError(f"opening speaker {turn.speaker_id!r} has no state")
            receiver_id = next(speaker for speaker in states if speaker != turn.speaker_id)
            visible = list(opening_turns[: index + 1])
            states[receiver_id] = self.update_state(states[receiver_id], turn, visible, config).state
        return states

    def rate(self, reader_id: str, observed_turn: DialogueTurn, visible_history: list[DialogueTurn]) -> float:
        """Rater's score for observed_turn (the last visible turn); it never sees trust, so ratings don't feed back."""
        payload = json.dumps(
            {"reader_id": reader_id, "history": [to_dict(turn) for turn in visible_history]},
            ensure_ascii=False,
        )
        completion = self.client.complete(self.instructions, payload, self.rater_config)
        # Report before parsing so failed ratings still count toward cost and keep their raw text.
        if self.on_rating is not None:
            self.on_rating(
                {
                    **completion.metadata,
                    "turn_id": observed_turn.turn_id,
                    "reader_id": reader_id,
                    "raw_text": completion.text,
                    "prompt_version": self.rater_config.prompt_version,
                }
            )
        return parse_rating(completion.text)

    __call__ = update_state
