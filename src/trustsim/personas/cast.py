"""Persona A and Persona B: fixed attributes, a mood for every trust band, and who plays whom."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from trustsim.config import ConfigError, check_keys, load_yaml, number_in, text
from trustsim.dialogue import Seed

PERSONA_IDS = ("A", "B")
PERSONA_FIELDS = {"role", "background", "temperament", "alpha_up", "alpha_down", "forgiveness"}


@dataclass(frozen=True)
class Persona:
    """One persona. Nothing here changes during a conversation."""

    persona_id: str
    role: str
    background: str
    temperament: str
    alpha_up: float
    alpha_down: float
    forgiveness: float
    moods: Mapping[str, str]

    def profile(self) -> dict[str, str]:
        """What the agent is told about itself."""
        return {"role": self.role, "background": self.background, "temperament": self.temperament}


@dataclass(frozen=True)
class MoodBand:
    """A trust range, from min_trust up to the next band's minimum."""

    name: str
    min_trust: float


@dataclass(frozen=True)
class Cast:
    """Both personas and the settings they share."""

    starting_trust: float
    bands: tuple[MoodBand, ...]
    personas: Mapping[str, Persona]

    def band(self, trust: float) -> str:
        """Name of the band that a trust value falls in."""
        for band in self.bands:
            if trust >= band.min_trust:
                return band.name
        raise ValueError(f"trust {trust} is below every mood band")

    def mood(self, persona_id: str, trust: float) -> tuple[str, str]:
        """The band name and the persona's mood for that band."""
        band = self.band(trust)
        return band, self.personas[persona_id].moods[band]

    def summary(self) -> dict[str, Any]:
        """Plain values for the run manifest."""
        return {
            "starting_trust": self.starting_trust,
            "bands": [[band.name, band.min_trust] for band in self.bands],
            "personas": {
                persona_id: {
                    **persona.profile(),
                    "alpha_up": persona.alpha_up,
                    "alpha_down": persona.alpha_down,
                    "forgiveness": persona.forgiveness,
                    "moods": dict(persona.moods),
                }
                for persona_id, persona in self.personas.items()
            },
        }


def load_cast(path: Path) -> Cast:
    """Both personas from configs/personas.yaml, checked for the values the experiments need."""
    where = str(path)
    raw = check_keys(load_yaml(path), where, {"starting_trust", "mood_bands", "personas"})
    bands = _bands(raw["mood_bands"], f"{where}: mood_bands")
    personas_raw = check_keys(raw["personas"], f"{where}: personas", set(PERSONA_IDS))
    personas = {
        persona_id: _persona(persona_id, personas_raw[persona_id], bands, f"{where}: {persona_id}")
        for persona_id in PERSONA_IDS
    }
    return Cast(
        starting_trust=number_in(raw["starting_trust"], f"{where}: starting_trust", 0, 1),
        bands=bands,
        personas=personas,
    )


def _bands(raw: Any, where: str) -> tuple[MoodBand, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{where} must be a non-empty list")
    bands = []
    for index, item in enumerate(raw):
        item = check_keys(item, f"{where}[{index}]", {"name", "min"})
        bands.append(
            MoodBand(
                text(item["name"], f"{where}[{index}].name"),
                number_in(item["min"], f"{where}[{index}].min", 0, 1),
            )
        )
    minimums = [band.min_trust for band in bands]
    if any(higher <= lower for higher, lower in pairwise(minimums)):
        raise ConfigError(f"{where} must be listed from the highest min to the lowest")
    if minimums[-1] != 0:
        raise ConfigError(f"{where}: the last band must start at 0 so every trust value has a mood")
    if len({band.name for band in bands}) != len(bands):
        raise ConfigError(f"{where}: band names must be unique")
    return tuple(bands)


def _persona(persona_id: str, raw: Any, bands: tuple[MoodBand, ...], where: str) -> Persona:
    raw = check_keys(raw, where, PERSONA_FIELDS | {"moods"})
    moods = check_keys(raw["moods"], f"{where}.moods", {band.name for band in bands})
    alphas = {}
    for name in ("alpha_up", "alpha_down"):
        alphas[name] = number_in(raw[name], f"{where}.{name}", 0, 1)
        if alphas[name] == 0:
            raise ConfigError(f"{where}.{name} must be above 0, or trust could never move")
    forgiveness = number_in(raw["forgiveness"], f"{where}.forgiveness", 0, 1)
    if forgiveness == 1:
        raise ConfigError(
            f"{where}.forgiveness must be below 1, or every message would be forgotten"
        )
    return Persona(
        persona_id=persona_id,
        role=text(raw["role"], f"{where}.role"),
        background=text(raw["background"], f"{where}.background"),
        temperament=text(raw["temperament"], f"{where}.temperament"),
        alpha_up=alphas["alpha_up"],
        alpha_down=alphas["alpha_down"],
        forgiveness=forgiveness,
        moods={band.name: text(moods[band.name], f"{where}.moods.{band.name}") for band in bands},
    )


def assign_personas(seed: Seed) -> dict[str, str]:
    """Map each real speaker to a persona: the first speaker plays A, the other plays B."""
    speakers = seed.speakers
    if len(speakers) != len(PERSONA_IDS):
        raise ValueError(
            f"conversation {seed.conversation_id} has {len(speakers)} speakers; "
            f"the experiments need exactly {len(PERSONA_IDS)}"
        )
    return dict(zip(speakers, PERSONA_IDS, strict=True))
