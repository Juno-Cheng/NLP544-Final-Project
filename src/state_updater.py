"""Static and dynamic trust updates for an agent that just observed its partner's turn.

Static keeps trust fixed and makes no model call. Dynamic asks a rater model how
cooperative the observed turn is (rating in [0, 1]) and moves trust toward it:
    trust' = (1 - alpha) * trust + alpha * rating
which stays in [0, 1] without clamping.

Usage:
    updater = TrustUpdater(client, model_id=os.environ["OPENAI_MODEL"])
    result = updater(state, observed_turn, visible_history, UpdaterConfig("dynamic", "trust_v1"))

The client is anything with complete(instructions, payload, config) returning an
object with .text, i.e. the LLMClient protocol from src/llm_client.py.
"""

import json
import math
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

    def __init__(self, client: Any, model_id: str, settings: dict | None = None) -> None:
        settings = settings or load_config()
        self.version = settings["updater_version"]
        self.alpha = float(settings["alpha"])
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        rater = settings["rater"]
        self.rater_config = GenerationConfig(
            model_id, rater["temperature"], rater["max_output_tokens"], rater["prompt_version"], rater["random_seed"]
        )
        self.instructions = (PROMPTS_DIR / f"{rater['prompt_version']}.txt").read_text(encoding="utf-8")
        self.client = client

    def __call__(
        self,
        current_state: AgentState,
        observed_turn: DialogueTurn,
        visible_history: list[DialogueTurn],
        config: UpdaterConfig,
    ) -> StateUpdateResult:
        return self.update_state(current_state, observed_turn, visible_history, config)

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
        trust = (1 - self.alpha) * current_state.trust + self.alpha * rating
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
        return parse_rating(self.client.complete(self.instructions, payload, self.rater_config).text)
