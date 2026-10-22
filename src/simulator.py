"""Run one two-speaker continuation with injected reply and trust-update callbacks.

The experiment runner supplies the seed, speaker-keyed persona configs, and
RunMetadata (including prompt/updater versions and experiment/pair IDs):
    result = run_simulation(seed, personas, config, generator, updater,
                            run_metadata=metadata)

The next speaker is the partner of the opening's last speaker. Opening turns
never update trust. State history starts with the post-opening initial states,
then records both agents after each generated turn and its receiver's update.
Callback failures return a failed result with the dialogue completed so far;
invalid setup raises TypeError or ValueError. Saving belongs to the experiment runner.
"""

import logging
from copy import deepcopy
from dataclasses import replace
from itertools import pairwise
from time import perf_counter
from uuid import uuid4

from src.persona import initialize_personas

LOGGER = logging.getLogger(__name__)
from src.schemas import (
    AgentState,
    ConversationSeed,
    DialogueTurn,
    GenerationConfig,
    PersonaConfig,
    ResponseGenerator,
    RunMetadata,
    SimulationConfig,
    SimulationResult,
    StateSnapshot,
    StateUpdater,
    StateUpdateResult,
    UpdaterConfig,
)


def _validate_setup(
    seed: ConversationSeed,
    personas: dict[str, PersonaConfig],
    config: SimulationConfig,
    generator: ResponseGenerator,
    updater: StateUpdater,
    run_metadata: RunMetadata,
    run_id: str | None,
) -> list[str]:
    """Validate one run's inputs and return the speakers in first-appearance order."""
    if not isinstance(seed, ConversationSeed):
        raise TypeError("seed must be a ConversationSeed without outcome labels")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if not isinstance(run_metadata, RunMetadata):
        raise TypeError("run_metadata must be a RunMetadata")
    if not callable(generator) or not callable(updater):
        raise TypeError("generator and updater must be callable")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a non-empty string")

    speakers = list(dict.fromkeys(turn.speaker_id for turn in seed.opening_turns))
    if len(speakers) != 2:
        raise ValueError("simulation requires exactly 2 speakers in the opening")
    if not isinstance(personas, dict) or set(personas) != set(speakers):
        raise ValueError("personas must be keyed by the opening's two speaker_ids")
    if not all(isinstance(persona, PersonaConfig) for persona in personas.values()):
        raise ValueError("personas must contain PersonaConfig values")
    turn_ids = [turn.turn_id for turn in seed.opening_turns]
    if any(current <= previous for previous, current in pairwise(turn_ids)):
        raise ValueError("opening turn_ids must be strictly increasing")

    for field in ("model_id", "max_generated_turns", "max_output_tokens", "temperature", "random_seed"):
        if getattr(run_metadata, field) != getattr(config, field):
            raise ValueError(f"run_metadata.{field} must match config.{field}")
    if run_metadata.error is not None:
        raise ValueError("run_metadata.error must be unset before a run")
    return speakers


def _validate_reply(turn: DialogueTurn, speaker_id: str, turn_id: int) -> None:
    """Require one nonempty reply from the requested speaker at the next turn ID."""
    if not isinstance(turn, DialogueTurn):
        raise TypeError("generator must return a DialogueTurn")
    if turn.speaker_id != speaker_id:
        raise ValueError(f"generator returned speaker {turn.speaker_id!r}; expected {speaker_id!r}")
    if turn.turn_id != turn_id:
        raise ValueError(f"generator returned turn_id {turn.turn_id}; expected {turn_id}")
    if not turn.text.strip():
        raise ValueError("generator returned an empty reply")


def _validate_update(
    result: StateUpdateResult, current_state: AgentState, config: UpdaterConfig
) -> None:
    """Require an update of only the receiver's trust, with static trust unchanged."""
    if not isinstance(result, StateUpdateResult) or not isinstance(result.state, AgentState):
        raise TypeError("updater must return a StateUpdateResult containing an AgentState")
    if result.state.speaker_id != current_state.speaker_id:
        raise ValueError("updater changed the receiver's speaker_id")
    if result.state.stable_traits != current_state.stable_traits:
        raise ValueError("updater changed stable_traits")
    if result.previous_trust != current_state.trust:
        raise ValueError("updater previous_trust does not match the receiver's current trust")
    if not 0.0 <= result.state.trust <= 1.0:
        raise ValueError("updater trust must be between 0 and 1")
    if config.condition == "static" and result.state.trust != current_state.trust:
        raise ValueError("updater changed trust in the static condition")


def run_simulation(
    seed: ConversationSeed,
    personas: dict[str, PersonaConfig],
    config: SimulationConfig,
    generator: ResponseGenerator,
    updater: StateUpdater,
    *,
    run_metadata: RunMetadata,
    run_id: str | None = None,
) -> SimulationResult:
    """Return one continuation, preserving partial dialogue and an error on callback failure."""
    speakers = _validate_setup(seed, personas, config, generator, updater, run_metadata, run_id)
    started = perf_counter()
    states = initialize_personas(personas)
    transcript = deepcopy(seed.opening_turns)
    generated_turns: list[DialogueTurn] = []
    state_history = [StateSnapshot(transcript[-1].turn_id, deepcopy(states))]
    generation_config = GenerationConfig(
        model_id=config.model_id,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        prompt_version=run_metadata.prompt_version,
        random_seed=config.random_seed,
    )
    updater_config = UpdaterConfig(config.condition, run_metadata.updater_version)
    speaker_id = next(speaker for speaker in speakers if speaker != transcript[-1].speaker_id)
    status = "ok"
    error = None
    LOGGER.info(
        "simulate %s condition=%s generate %s turns starting with %s",
        seed.conversation_id,
        config.condition,
        config.max_generated_turns,
        speaker_id,
    )

    for _ in range(config.max_generated_turns):
        turn_id = transcript[-1].turn_id + 1
        stage = "generation"
        try:
            # Frozen schemas still contain mutable lists/dicts; isolate callback inputs.
            turn = generator(deepcopy(transcript), deepcopy(states[speaker_id]), generation_config)
            _validate_reply(turn, speaker_id, turn_id)
            turn = deepcopy(turn)
            transcript.append(turn)
            generated_turns.append(turn)

            stage = "state update"
            receiver_id = next(speaker for speaker in speakers if speaker != speaker_id)
            visible_history = deepcopy(transcript)
            update = updater(
                deepcopy(states[receiver_id]), visible_history[-1], visible_history, updater_config
            )
            _validate_update(update, states[receiver_id], updater_config)
            states[receiver_id] = deepcopy(update.state)
            state_history.append(StateSnapshot(turn_id, deepcopy(states)))
            speaker_id = receiver_id
        except Exception as exc:  # noqa: BLE001 -- retain callback failures as inspectable run results.
            status = "failed"
            error = f"{stage} failed at turn {turn_id} after requesting {speaker_id!r}: {type(exc).__name__}: {exc}"
            LOGGER.error("%s", error)
            break

    LOGGER.info("simulate %s %s: %s generated turns", seed.conversation_id, config.condition, len(generated_turns))
    return SimulationResult(
        run_id=str(uuid4()) if run_id is None else run_id,
        conversation_id=seed.conversation_id,
        condition=config.condition,
        transcript=transcript,
        generated_turns=generated_turns,
        state_history=state_history,
        status=status,
        run_metadata=replace(
            deepcopy(run_metadata), error=error, latency_ms=(perf_counter() - started) * 1000
        ),
    )
