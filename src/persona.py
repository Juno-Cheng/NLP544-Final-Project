"""Persona configs and fresh per-run agent states for the two simulated speakers.

Usage:
    by_role = load_persona_configs()             # {"A": PersonaConfig, "B": PersonaConfig}
    personas = assign_personas(seed, by_role)    # keyed by the seed's real speaker_ids
    states = initialize_personas(personas)       # call once per arm; never reuse across arms
"""

from copy import deepcopy
from pathlib import Path

import yaml

from src.schemas import AgentState, ConversationSeed, PersonaConfig, from_dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "personas.yaml"
ROLES = ("A", "B")


def load_persona_configs(path: Path = CONFIG_PATH) -> dict[str, PersonaConfig]:
    """Return the A and B persona configs from configs/personas.yaml."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or set(data) != set(ROLES):
        raise ValueError(f"{path} must define exactly the roles {ROLES}")
    return {role: from_dict(PersonaConfig, {"persona_id": role, **data[role]}) for role in ROLES}


def assign_personas(seed: ConversationSeed, by_role: dict[str, PersonaConfig]) -> dict[str, PersonaConfig]:
    """Map the opening's first speaker to role A and the other to role B."""
    speakers = list(dict.fromkeys(turn.speaker_id for turn in seed.opening_turns))
    if len(speakers) != len(ROLES):
        raise ValueError(f"{seed.conversation_id} has {len(speakers)} speakers; personas need exactly 2")
    return dict(zip(speakers, (by_role[role] for role in ROLES)))


def initialize_personas(configs: dict[str, PersonaConfig]) -> dict[str, AgentState]:
    """Fresh AgentState per speaker at initial trust; traits are copied so arms never share state."""
    return {
        speaker_id: AgentState(speaker_id, config.initial_trust, deepcopy(config.stable_traits))
        for speaker_id, config in configs.items()
    }
