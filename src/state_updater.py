"""Static and dynamic trust updates for an agent that just observed its partner's turn.

Static keeps trust fixed and makes no model call. Dynamic asks a rater model how
cooperative the observed turn is (rating in [0, 1], 0.5 = neutral) and shifts trust by it:
    trust' = clamp(trust + alpha * (rating - 0.5))
so a neutral message leaves trust unchanged.

Usage:
    updater = TrustUpdater(client, model_id=os.environ["OPENAI_MODEL"], on_rating=log)
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

from src.schemas import AgentState, DialogueTurn, GenerationConfig, StateUpdateResult, UpdaterConfig, to_dict

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


class TrustUpdater:
    """Callable StateUpdater for run_simulation; the client is only called in the dynamic condition."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        settings: dict | None = None,
        *,
        on_rating: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        settings = settings or load_config()
        self.version = settings["updater_version"]
        self.alpha = float(settings["alpha"])
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
        rating = self.rate(current_state.speaker_id, observed_turn, visible_history)
        trust = min(1.0, max(0.0, current_state.trust + self.alpha * (rating - 0.5)))
        return StateUpdateResult(replace(current_state, trust=trust), current_state.trust)

    def rate(self, reader_id: str, observed_turn: DialogueTurn, visible_history: list[DialogueTurn]) -> float:
        """Rater's cooperativeness score for observed_turn; it never sees trust, so ratings don't feed back."""
        payload = json.dumps(
            {
                "reader_id": reader_id,
                "history": [to_dict(turn) for turn in visible_history],
                "observed_turn": to_dict(observed_turn),
            },
            ensure_ascii=False,
        )
        completion = self.client.complete(self.instructions, payload, self.rater_config)
        rating = parse_rating(completion.text)
        if self.on_rating is not None:
            self.on_rating(
                {
                    **completion.metadata,
                    "turn_id": observed_turn.turn_id,
                    "reader_id": reader_id,
                    "rating": rating,
                    "prompt_version": self.rater_config.prompt_version,
                }
            )
        return rating

    __call__ = update_state
