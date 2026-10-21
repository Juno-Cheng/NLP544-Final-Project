"""Shared data contracts for the static vs. dynamic persona experiment.

Pipeline modules import these types instead of redefining them. Serialize with
`to_dict` and rebuild with `from_dict`:

    payload = json.dumps(to_dict(seed), ensure_ascii=False)
    seed = from_dict(ConversationSeed, json.loads(payload))

Invalid records raise ValueError. Direct construction and `from_dict` use the
same checks.
"""

from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from types import NoneType, UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")

CONDITIONS = ("static", "dynamic")
Condition = Literal["static", "dynamic"]
DEFAULT_RUBRIC_PATH = "prompts/judge_rubric_v1.txt"
JudgeCallable = Callable[[str], Any]


def to_dict(obj: Any) -> dict[str, Any]:
    """Return a plain dict suitable for json.dumps(..., ensure_ascii=False)."""
    if not is_dataclass(obj):
        raise TypeError(f"expected a schema dataclass, got {type(obj).__name__}")
    return asdict(obj)


def from_dict(cls: type[T], data: dict[str, Any], path: str | None = None) -> T:
    """Build a schema object from a dict, including nested schema objects."""
    label = path or cls.__name__
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a dict")
    known = {field.name: field for field in fields(cls)}
    unexpected = sorted(set(data) - set(known))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, field in known.items():
        if name not in data:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"{label} missing required field {name!r}")
            continue
        kwargs[name] = _convert(hints[name], data[name], f"{label}.{name}")
    return cls(**kwargs)


def judge_payload(transcript: list["DialogueTurn"]) -> list[dict[str, Any]]:
    """Ordered dialogue only. Do not add condition, trust, model ids, or labels."""
    if not isinstance(transcript, list) or not all(isinstance(turn, DialogueTurn) for turn in transcript):
        raise ValueError("judge payload must be a list of DialogueTurn")
    return [to_dict(turn) for turn in transcript]


def _convert(typ: Any, value: Any, path: str) -> Any:
    if typ is Any:
        return value
    inner = _optional_inner(typ)
    if inner is not None:
        if value is None:
            return None
        return _convert(inner, value, path)
    origin = get_origin(typ)
    if origin is Literal:
        allowed = get_args(typ)
        if value not in allowed:
            raise ValueError(f"{path} must be one of {allowed}, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        (item_type,) = get_args(typ)
        return [_convert(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a dict")
        key_type, value_type = get_args(typ)
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            converted_key = _dict_key(key_type, key, path)
            converted[converted_key] = _convert(value_type, item, f"{path}.{converted_key}")
        return converted
    if origin is Callable:
        if not callable(value):
            raise ValueError(f"{path} must be callable")
        return value
    if isinstance(typ, type) and is_dataclass(typ):
        return from_dict(typ, value, path)
    if typ is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a bool")
        return value
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an int")
        return value
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a number")
        return float(value)
    if typ is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        return value
    raise TypeError(f"unsupported schema type {typ!r} at {path}")


def _dict_key(key_type: Any, key: Any, path: str) -> Any:
    if key_type is int:
        if isinstance(key, bool):
            raise ValueError(f"{path} keys must be ints")
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.isdecimal():
            return int(key)
        raise ValueError(f"{path} keys must be ints")
    if not isinstance(key, key_type):
        raise ValueError(f"{path} keys must be {getattr(key_type, '__name__', key_type)}")
    return key


def _optional_inner(typ: Any) -> Any:
    origin = get_origin(typ)
    if origin not in (Union, UnionType):
        return None
    args = get_args(typ)
    non_none = [arg for arg in args if arg is not NoneType]
    if len(args) == 2 and len(non_none) == 1:
        return non_none[0]
    return None


def _nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _unit_interval(value: float, field: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1, got {value}")


def _signed_unit(value: float, field: str) -> None:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between -1 and 1, got {value}")


def _outcome_counts(counts: dict[int, int], field: str) -> None:
    for key, count in counts.items():
        if isinstance(key, bool) or key not in (0, 1):
            raise ValueError(f"{field} keys must be 0 or 1")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field} counts must be nonnegative ints")


def _string_keyed(value: dict, field: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a dict with string keys")


def _unique_turn_ids(turns: list["DialogueTurn"], where: str) -> None:
    seen: set[int] = set()
    for turn in turns:
        if turn.turn_id in seen:
            raise ValueError(f"{where} repeats turn_id {turn.turn_id}")
        seen.add(turn.turn_id)


@dataclass(frozen=True)
class DialogueTurn:
    """One utterance: a real opening comment or a generated reply."""

    turn_id: int
    speaker_id: str
    text: str

    def __post_init__(self) -> None:
        if self.turn_id < 0:
            raise ValueError(f"turn_id must be >= 0, got {self.turn_id}")
        _nonempty(self.speaker_id, "speaker_id")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")


@dataclass(frozen=True)
class ConversationSeed:
    """Opening the agents may see. Outcome labels are not a field of this type."""

    conversation_id: str
    source: str
    opening_turns: list[DialogueTurn]

    def __post_init__(self) -> None:
        _nonempty(self.conversation_id, "conversation_id")
        _nonempty(self.source, "source")
        if not self.opening_turns:
            raise ValueError("opening_turns must contain at least one turn")
        _unique_turn_ids(self.opening_turns, "opening_turns")


@dataclass(frozen=True)
class LabeledExample:
    """Seed plus the real outcome. Only evaluation code should receive this object."""

    seed: ConversationSeed
    real_outcome: int
    label_definition: str
    split: str
    pair_id: str | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if self.real_outcome not in (0, 1):
            raise ValueError(f"real_outcome must be 0 or 1, got {self.real_outcome}")
        _nonempty(self.label_definition, "label_definition")
        _nonempty(self.split, "split")
        if self.pair_id is not None:
            _nonempty(self.pair_id, "pair_id")


TRUST_SENSITIVITY_KEYS = ("alpha_up", "alpha_down")


@dataclass(frozen=True)
class PersonaConfig:
    """Stable traits and the trust value both arms start from. Trust is not a measured human score.

    trust_sensitivity holds this persona's alpha_up/alpha_down step sizes; None uses the updater defaults.
    """

    persona_id: str
    stable_traits: dict[str, Any]
    initial_trust: float
    trust_sensitivity: dict[str, float] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.persona_id, "persona_id")
        _string_keyed(self.stable_traits, "stable_traits")
        _unit_interval(self.initial_trust, "initial_trust")
        if self.trust_sensitivity is not None:
            if set(self.trust_sensitivity) != set(TRUST_SENSITIVITY_KEYS):
                raise ValueError(f"trust_sensitivity must have exactly the keys {TRUST_SENSITIVITY_KEYS}")
            for key, alpha in self.trust_sensitivity.items():
                if not 0.0 < alpha <= 1.0:  # also rejects nan
                    raise ValueError(f"trust_sensitivity.{key} must be in (0, 1], got {alpha}")


@dataclass(frozen=True)
class AgentState:
    """One speaker's trust and traits at a single moment in a run."""

    speaker_id: str
    trust: float
    stable_traits: dict[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.speaker_id, "speaker_id")
        _string_keyed(self.stable_traits, "stable_traits")
        _unit_interval(self.trust, "trust")


@dataclass(frozen=True)
class StateSnapshot:
    """Trust of each speaker after a visible turn."""

    turn_id: int
    speaker_states: dict[str, AgentState]

    def __post_init__(self) -> None:
        if self.turn_id < 0:
            raise ValueError(f"turn_id must be >= 0, got {self.turn_id}")
        for speaker_id, state in self.speaker_states.items():
            if state.speaker_id != speaker_id:
                raise ValueError(
                    f"speaker_states key {speaker_id!r} does not match state.speaker_id {state.speaker_id!r}"
                )


@dataclass(frozen=True)
class SimulationConfig:
    """Settings for one simulated continuation. Both arms must be given the same values except condition."""

    condition: Condition
    model_id: str
    max_generated_turns: int
    temperature: float
    max_output_tokens: int
    random_seed: int

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        _nonempty(self.model_id, "model_id")
        if self.max_generated_turns <= 0:
            raise ValueError("max_generated_turns must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    """Per-reply model settings. There is no condition field: the reply sees trust only via AgentState."""

    model_id: str
    temperature: float
    max_output_tokens: int
    prompt_version: str
    random_seed: int

    def __post_init__(self) -> None:
        _nonempty(self.model_id, "model_id")
        _nonempty(self.prompt_version, "prompt_version")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True)
class UpdaterConfig:
    """Trust-update policy. This is the only place a reply's state update learns static vs. dynamic."""

    condition: Condition
    updater_version: str

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        _nonempty(self.updater_version, "updater_version")


@dataclass(frozen=True)
class StateUpdateResult:
    """Agent state after observing one partner turn."""

    state: AgentState
    previous_trust: float

    def __post_init__(self) -> None:
        _unit_interval(self.previous_trust, "previous_trust")


@dataclass(frozen=True)
class RunMetadata:
    """Versions and settings needed to trace one run. error is set when status is a failure."""

    experiment_id: str
    paired_run_id: str
    model_id: str
    prompt_version: str
    updater_version: str
    max_generated_turns: int
    max_output_tokens: int
    temperature: float
    random_seed: int
    error: str | None = None
    token_usage: dict[str, Any] | None = None
    latency_ms: float | None = None
    cost: float | None = None

    def __post_init__(self) -> None:
        for field in ("experiment_id", "paired_run_id", "model_id", "prompt_version", "updater_version"):
            _nonempty(getattr(self, field), field)
        if self.max_generated_turns <= 0:
            raise ValueError("max_generated_turns must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.error is not None:
            _nonempty(self.error, "error")
        if self.token_usage is not None:
            _string_keyed(self.token_usage, "token_usage")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.cost is not None and self.cost < 0:
            raise ValueError("cost must be >= 0")


@dataclass(frozen=True)
class SimulationResult:
    """One arm of a paired run. transcript is the opening plus generated turns, in order."""

    run_id: str
    conversation_id: str
    condition: Condition
    transcript: list[DialogueTurn]
    generated_turns: list[DialogueTurn]
    state_history: list[StateSnapshot]
    status: str
    run_metadata: RunMetadata

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _nonempty(self.conversation_id, "conversation_id")
        _nonempty(self.status, "status")
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        if not self.transcript:
            raise ValueError("transcript must contain the opening")
        _unique_turn_ids(self.transcript, "transcript")
        _unique_turn_ids(self.generated_turns, "generated_turns")
        if len(self.generated_turns) > len(self.transcript):
            raise ValueError("generated_turns is longer than transcript")
        tail = self.transcript[-len(self.generated_turns) :] if self.generated_turns else []
        if list(tail) != list(self.generated_turns):
            raise ValueError("generated_turns must be the ordered tail of transcript")


@dataclass(frozen=True)
class JudgeConfig:
    """Rubric and judge model. Condition, trust, and labels do not belong here.

    judge_callable is injected at runtime and is absent from configs/judge.yaml.
    """

    judge_model_id: str
    rubric_version: str
    rubric_path: str = DEFAULT_RUBRIC_PATH
    judge_callable: JudgeCallable | None = None

    def __post_init__(self) -> None:
        _nonempty(self.judge_model_id, "judge_model_id")
        _nonempty(self.rubric_version, "rubric_version")
        _nonempty(self.rubric_path, "rubric_path")


@dataclass(frozen=True)
class EscalationScore:
    """Blind judge score in [0, 1] for one run's transcript."""

    run_id: str
    score: float
    judge_model_id: str
    rubric_version: str

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _nonempty(self.judge_model_id, "judge_model_id")
        _nonempty(self.rubric_version, "rubric_version")
        _unit_interval(self.score, "score")


@dataclass(frozen=True)
class DataConfig:
    """Fields of configs/data.yaml. sample_size null means keep every example that passes the filters."""

    source: str
    seeds_path: str
    labels_path: str
    opening_turns: int
    min_opening_turns: int
    min_speakers: int
    max_speakers: int
    splits: list[str]
    verified_only: bool
    sample_size: int | None
    keep_pairs: bool
    random_seed: int

    def __post_init__(self) -> None:
        _nonempty(self.source, "source")
        _nonempty(self.seeds_path, "seeds_path")
        _nonempty(self.labels_path, "labels_path")
        if self.opening_turns <= 0:
            raise ValueError("opening_turns must be positive")
        if self.min_opening_turns <= 0:
            raise ValueError("min_opening_turns must be positive")
        if self.min_opening_turns > self.opening_turns:
            raise ValueError("min_opening_turns cannot exceed opening_turns")
        if self.min_speakers < 1 or self.max_speakers < self.min_speakers:
            raise ValueError("speaker bounds must satisfy 1 <= min_speakers <= max_speakers")
        if not self.splits or any(not split.strip() for split in self.splits):
            raise ValueError("splits must be a non-empty list of names")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be positive or null")


@dataclass(frozen=True)
class PairedComparison:
    """One source conversation whose static and dynamic arms both scored."""

    conversation_id: str
    real_outcome: int
    static_run_id: str
    dynamic_run_id: str
    static_score: float
    dynamic_score: float
    score_delta: float
    status: str
    unverified_pair_settings: list[str]
    pair_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.conversation_id, "conversation_id")
        _nonempty(self.static_run_id, "static_run_id")
        _nonempty(self.dynamic_run_id, "dynamic_run_id")
        _nonempty(self.status, "status")
        if self.real_outcome not in (0, 1):
            raise ValueError(f"real_outcome must be 0 or 1, got {self.real_outcome}")
        if self.pair_id is not None:
            _nonempty(self.pair_id, "pair_id")
        _unit_interval(self.static_score, "static_score")
        _unit_interval(self.dynamic_score, "dynamic_score")
        _signed_unit(self.score_delta, "score_delta")
        if abs(self.score_delta - (self.dynamic_score - self.static_score)) > 1e-9:
            raise ValueError("score_delta must equal dynamic_score - static_score")
        if any(not item.strip() for item in self.unverified_pair_settings):
            raise ValueError("unverified_pair_settings entries must be non-empty")


@dataclass(frozen=True)
class ExcludedPair:
    """A source conversation left out of the metrics, with the reason for each missing piece."""

    conversation_id: str
    real_outcome: int
    static_run_id: str | None
    dynamic_run_id: str | None
    reasons: list[str]

    def __post_init__(self) -> None:
        _nonempty(self.conversation_id, "conversation_id")
        if self.real_outcome not in (0, 1):
            raise ValueError(f"real_outcome must be 0 or 1, got {self.real_outcome}")
        for field in ("static_run_id", "dynamic_run_id"):
            value = getattr(self, field)
            if value is not None:
                _nonempty(value, field)
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must be a non-empty list of messages")


@dataclass(frozen=True)
class CoverageReport:
    """How many source conversations were attempted, scored, and left out."""

    n_examples: int
    n_attempted: int
    n_completed_pairs: int
    n_excluded: int
    completion_rate: float | None
    label_counts_selected: dict[int, int]
    label_counts_evaluated: dict[int, int]

    def __post_init__(self) -> None:
        counts = (self.n_examples, self.n_attempted, self.n_completed_pairs, self.n_excluded)
        if any(count < 0 for count in counts):
            raise ValueError("coverage counts must be >= 0")
        if self.n_attempted > self.n_examples or self.n_completed_pairs > self.n_examples:
            raise ValueError("coverage counts cannot exceed n_examples")
        if self.n_examples == 0:
            if self.completion_rate is not None:
                raise ValueError("completion_rate must be null when there are no examples")
        elif self.completion_rate is None:
            raise ValueError("completion_rate is required when there are examples")
        elif abs(self.completion_rate - (self.n_completed_pairs / self.n_examples)) > 1e-9:
            raise ValueError("completion_rate must equal n_completed_pairs / n_examples")
        _outcome_counts(self.label_counts_selected, "label_counts_selected")
        _outcome_counts(self.label_counts_evaluated, "label_counts_evaluated")


@dataclass(frozen=True)
class EvaluationMetrics:
    """Descriptive scores on completed pairs. None means the statistic is undefined."""

    n_pairs: int
    label_counts: dict[int, int]
    roc_auc_static: float | None
    roc_auc_dynamic: float | None
    roc_auc_delta: float | None
    spearman_static: float | None
    spearman_dynamic: float | None
    spearman_delta: float | None
    mean_score_delta: float | None

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be >= 0")
        _outcome_counts(self.label_counts, "label_counts")
        for field in ("roc_auc_static", "roc_auc_dynamic"):
            value = getattr(self, field)
            if value is not None:
                _unit_interval(value, field)
        for field in ("roc_auc_delta", "spearman_static", "spearman_dynamic", "mean_score_delta"):
            value = getattr(self, field)
            if value is not None:
                _signed_unit(value, field)
        if self.spearman_delta is not None and not -2.0 <= self.spearman_delta <= 2.0:
            raise ValueError(f"spearman_delta must be between -2 and 2, got {self.spearman_delta}")


@dataclass(frozen=True)
class EvaluationSummary:
    """Joined comparison of static and dynamic scores against held-out labels."""

    pairs: list[PairedComparison]
    excluded: list[ExcludedPair]
    coverage: CoverageReport
    metrics: EvaluationMetrics
    unverified_pair_settings: list[str]
    notes: list[str]

    def __post_init__(self) -> None:
        if self.coverage.n_completed_pairs != len(self.pairs) or self.metrics.n_pairs != len(self.pairs):
            raise ValueError("completed-pair counts must equal len(pairs)")
        if self.coverage.n_excluded != len(self.excluded):
            raise ValueError("coverage.n_excluded must equal len(excluded)")
        if self.coverage.n_completed_pairs + self.coverage.n_excluded != self.coverage.n_examples:
            raise ValueError("completed and excluded pairs must account for every example")
        expected = sorted({key for row in self.pairs for key in row.unverified_pair_settings})
        if list(self.unverified_pair_settings) != expected:
            raise ValueError("unverified_pair_settings must be the sorted union of the pair lists")


ResponseGenerator = Callable[[list[DialogueTurn], AgentState, GenerationConfig], DialogueTurn]
StateUpdater = Callable[[AgentState, DialogueTurn, list[DialogueTurn], UpdaterConfig], StateUpdateResult]
