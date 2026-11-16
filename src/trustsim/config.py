"""Typed settings read from configs/data.yaml and configs/experiment.yaml."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

STATIC = "static"
DYNAMIC = "dynamic"
EXPERIMENTS = (STATIC, DYNAMIC)
AGENT = "agent"
CRITIC = "critic"
JUDGE = "judge"
ROLES = (AGENT, CRITIC, JUDGE)
MOCK_PROVIDER = "mock"
TOKEN_LIMIT_PARAMS = ("max_completion_tokens", "max_tokens")


class ConfigError(ValueError):
    """A settings file is missing a value, has an unknown one, or has one that cannot work."""


@dataclass(frozen=True)
class DataSettings:
    """Which prepared conversations a run uses."""

    seeds_path: Path
    labels_path: Path
    opening_turns: int
    min_opening_turns: int
    splits: tuple[str, ...]
    verified_only: bool
    sample_size: int | None
    keep_pairs: bool
    random_seed: int


@dataclass(frozen=True)
class ProviderSettings:
    """How to reach one model provider through the OpenAI SDK."""

    base_url: str | None
    api_key_env: str
    token_limit_param: str


@dataclass(frozen=True)
class RoleSettings:
    """Model settings for one role: the agents, the critic, or the judge."""

    model: str
    prompt_path: Path
    max_output_tokens: int
    temperature: float | None
    reasoning_effort: str | None


@dataclass(frozen=True)
class LLMSettings:
    """The provider in use and the settings of every role."""

    provider: str
    providers: Mapping[str, ProviderSettings]
    max_retries: int
    timeout_seconds: float
    roles: Mapping[str, RoleSettings]


@dataclass(frozen=True)
class ExperimentSettings:
    """Settings shared by both experiments."""

    experiments: tuple[str, ...]
    max_turns: int
    workers: int
    attempts_per_answer: int
    llm: LLMSettings


def load_yaml(path: Path) -> dict[str, Any]:
    """The mapping at the top level of a YAML file."""
    if not path.is_file():
        raise ConfigError(f"{path} does not exist")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def check_keys(
    raw: Any, where: str, required: set[str], optional: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """The mapping itself, after checking it has every required key and nothing unexpected."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    missing = sorted(required - raw.keys())
    unknown = sorted(raw.keys() - required - optional)
    if missing:
        raise ConfigError(f"{where} is missing {missing}")
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown}")
    return raw


def text(value: Any, where: str) -> str:
    """A non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def whole_number(value: Any, where: str, minimum: int) -> int:
    """An integer of at least minimum."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}, got {value!r}")
    return value


def number_in(value: Any, where: str, low: float, high: float) -> float:
    """A number between low and high, inclusive."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not low <= value <= high:
        raise ConfigError(f"{where} must be a number from {low} to {high}, got {value!r}")
    return float(value)


def flag(value: Any, where: str) -> bool:
    """A true/false value."""
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false, got {value!r}")
    return value


def load_data_settings(path: Path, root: Path) -> DataSettings:
    """Data settings, with file paths resolved against the repository root."""
    where = str(path)
    raw = check_keys(
        load_yaml(path),
        where,
        {
            "seeds_path",
            "labels_path",
            "opening_turns",
            "min_opening_turns",
            "splits",
            "verified_only",
            "sample_size",
            "keep_pairs",
            "random_seed",
        },
    )
    splits = raw["splits"]
    if not isinstance(splits, list) or not splits:
        raise ConfigError(f"{where}: splits must be a non-empty list")
    settings = DataSettings(
        seeds_path=root / text(raw["seeds_path"], f"{where}: seeds_path"),
        labels_path=root / text(raw["labels_path"], f"{where}: labels_path"),
        opening_turns=whole_number(raw["opening_turns"], f"{where}: opening_turns", 1),
        min_opening_turns=whole_number(raw["min_opening_turns"], f"{where}: min_opening_turns", 1),
        splits=tuple(text(split, f"{where}: splits") for split in splits),
        verified_only=flag(raw["verified_only"], f"{where}: verified_only"),
        sample_size=None
        if raw["sample_size"] is None
        else whole_number(raw["sample_size"], f"{where}: sample_size", 1),
        keep_pairs=flag(raw["keep_pairs"], f"{where}: keep_pairs"),
        random_seed=whole_number(raw["random_seed"], f"{where}: random_seed", 0),
    )
    if settings.min_opening_turns > settings.opening_turns:
        raise ConfigError(f"{where}: min_opening_turns cannot exceed opening_turns")
    check_sample_size(settings.sample_size, settings.keep_pairs)
    return settings


def check_sample_size(sample_size: int | None, keep_pairs: bool) -> None:
    """Whole attack/civil pairs need an even sample size."""
    if keep_pairs and sample_size is not None and sample_size % 2:
        raise ConfigError(f"sample_size must be even when keep_pairs is true, got {sample_size}")


def load_experiment_settings(path: Path, root: Path) -> ExperimentSettings:
    """Experiment settings, with prompt paths resolved against the repository root."""
    where = str(path)
    raw = check_keys(
        load_yaml(path),
        where,
        {"experiments", "max_turns", "workers", "attempts_per_answer", "llm"},
    )
    experiments = raw["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ConfigError(f"{where}: experiments must be a non-empty list")
    return ExperimentSettings(
        experiments=check_experiments(experiments),
        max_turns=whole_number(raw["max_turns"], f"{where}: max_turns", 1),
        workers=whole_number(raw["workers"], f"{where}: workers", 1),
        attempts_per_answer=whole_number(
            raw["attempts_per_answer"], f"{where}: attempts_per_answer", 1
        ),
        llm=_llm_settings(raw["llm"], f"{where}: llm", root),
    )


def check_experiments(names: Sequence[str]) -> tuple[str, ...]:
    """Experiment names in the standard order, each known and listed once."""
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        raise ConfigError(f"unknown experiments {unknown}; choose from {list(EXPERIMENTS)}")
    if len(set(names)) != len(names):
        raise ConfigError(f"experiments are listed more than once: {list(names)}")
    return tuple(name for name in EXPERIMENTS if name in names)


def _llm_settings(raw: Any, where: str, root: Path) -> LLMSettings:
    raw = check_keys(
        raw, where, {"provider", "providers", "max_retries", "timeout_seconds", "roles"}
    )
    providers_raw = raw["providers"]
    if not isinstance(providers_raw, dict) or not providers_raw:
        raise ConfigError(f"{where}: providers must be a non-empty mapping")
    providers = {
        name: _provider_settings(value, f"{where}: providers.{name}")
        for name, value in providers_raw.items()
    }
    roles_raw = check_keys(raw["roles"], f"{where}: roles", set(ROLES))
    settings = LLMSettings(
        provider=text(raw["provider"], f"{where}: provider"),
        providers=providers,
        max_retries=whole_number(raw["max_retries"], f"{where}: max_retries", 0),
        timeout_seconds=number_in(raw["timeout_seconds"], f"{where}: timeout_seconds", 1, 3600),
        roles={
            role: _role_settings(roles_raw[role], f"{where}: roles.{role}", root) for role in ROLES
        },
    )
    check_provider(settings.provider, settings.providers)
    return settings


def _provider_settings(raw: Any, where: str) -> ProviderSettings:
    raw = check_keys(raw, where, {"base_url", "api_key_env", "token_limit_param"})
    token_limit_param = raw["token_limit_param"]
    if token_limit_param not in TOKEN_LIMIT_PARAMS:
        raise ConfigError(f"{where}: token_limit_param must be one of {list(TOKEN_LIMIT_PARAMS)}")
    return ProviderSettings(
        base_url=None if raw["base_url"] is None else text(raw["base_url"], f"{where}: base_url"),
        api_key_env=text(raw["api_key_env"], f"{where}: api_key_env"),
        token_limit_param=token_limit_param,
    )


def _role_settings(raw: Any, where: str, root: Path) -> RoleSettings:
    raw = check_keys(
        raw,
        where,
        {"model", "prompt", "max_output_tokens", "temperature", "reasoning_effort"},
    )
    prompt_path = root / text(raw["prompt"], f"{where}: prompt")
    if not prompt_path.is_file():
        raise ConfigError(f"{where}: prompt file {prompt_path} does not exist")
    return RoleSettings(
        model=text(raw["model"], f"{where}: model"),
        prompt_path=prompt_path,
        max_output_tokens=whole_number(raw["max_output_tokens"], f"{where}: max_output_tokens", 16),
        temperature=None
        if raw["temperature"] is None
        else number_in(raw["temperature"], f"{where}: temperature", 0, 2),
        reasoning_effort=None
        if raw["reasoning_effort"] is None
        else text(raw["reasoning_effort"], f"{where}: reasoning_effort"),
    )


def check_provider(provider: str, providers: Mapping[str, ProviderSettings]) -> None:
    """The provider must be configured, or be the offline mock."""
    if provider != MOCK_PROVIDER and provider not in providers:
        known = [*providers, MOCK_PROVIDER]
        raise ConfigError(f"unknown provider {provider!r}; choose from {known}")


def with_overrides(
    settings: ExperimentSettings,
    *,
    provider: str | None = None,
    model: str | None = None,
    experiments: Sequence[str] | None = None,
    workers: int | None = None,
) -> ExperimentSettings:
    """Settings with command-line overrides applied; a model override applies to every role."""
    llm = settings.llm
    if provider is not None:
        check_provider(provider, llm.providers)
        llm = replace(llm, provider=provider)
    if model is not None:
        model = text(model, "--model")
        llm = replace(
            llm, roles={name: replace(role, model=model) for name, role in llm.roles.items()}
        )
    return replace(
        settings,
        llm=llm,
        experiments=settings.experiments if experiments is None else check_experiments(experiments),
        workers=settings.workers if workers is None else whole_number(workers, "--workers", 1),
    )
