# Static vs. Dynamic Personas — Technical Specification

**CSCI 544** | v0.2 | Proposed for team approval | October 13, 2026

---

## 1. Research goal and current scope

Compare whether explicitly updated trust in simulated conversations helps predict which real human conversations eventually contain personal attacks, versus keeping that trust fixed.

- **Static:** Responses use the evolving chat history, but explicit trust stays fixed.
- **Dynamic:** Same model, initial personas, and chat opening, but explicit trust updates as new messages arrive and the updated state affects subsequent responses.
- **Core deliverable:** Paired static/dynamic rollouts, a blind escalation judge, and comparison to real outcome labels.
- **Deferred:** Mathematical trust formula, full-scale mechanism comparisons, and advanced memory systems until the core pipeline works and the TA reviews the extension.

## 2. End-to-end flow and module handoffs

```text
Cornell CGA-WIKI corpus
  -> data_loader.py: ConversationSeed + separately stored real label
  -> persona.py: initial AgentState for speaker A and speaker B
  -> simulator.py + response_generator.py + state_updater.py:
       run static and dynamic from SAME seed and initial states
       save complete transcript, generated turns, per-turn states, run metadata
  -> escalation_judge.py: receive transcript text ONLY, output blind score
  -> evaluate.py: join scores with real labels by conversation_id/run_id
       produce paired comparison and reportable metrics
```

> **Rule:** The generator and state updater never receive real outcome labels or future human turns. The judge never receives the condition, trust history, or real label.

## 3. Ownership and files (proposed; confirm tonight)

| Owner | Module | Must hand off |
|---|---|---|
| Aditya | `simulator.py` | Dialogue loop, speaker turns, static/dynamic run orchestration, saved `SimulationResult` |
| Arya | `response_generator.py` | Consistent LLM client and persona-conditioned reply generation used by both modes |
| Magnus | `persona.py`, `state_updater.py` | Separate per-speaker state, static/no-update and dynamic/update policies, state history |
| Jonathan Cheng | `data_loader.py` | Validated `LabeledExample` records; clean `ConversationSeed`; label stored separately |
| Jon or Prakar | `schemas.py`, `run_experiment.py`, tests | Shared types, mock integration, one command to run paired experiment |
| Shadi | `escalation_judge.py`, `evaluate.py` | Blind score, judge validation checks, matched results, ROC-AUC when sample permits |

Everyone helps review the two-slide presentation and four-page report; name one report coordinator during the meeting. The exact Jon/Prakar assignment and report coordinator remain **OPEN**.

## 4. Common data definitions (freeze before independent coding)

Implement these with Pydantic or dataclasses in one `schemas.py`. Names and field meanings are the interface contract; implementation details may be agreed by the integration owner.

```python
DialogueTurn = {"turn_id": int, "speaker_id": str, "text": str}

ConversationSeed = {"conversation_id": str, "source": str,
                    "opening_turns": list[DialogueTurn]}

LabeledExample = {"seed": ConversationSeed, "real_outcome": int,
                  "label_definition": str, "split": str,
                  "pair_id": str | None}

PersonaConfig = {"persona_id": str, "stable_traits": dict,
                 "initial_trust": float}  # 0..1, not actual measured human trust

AgentState = {"speaker_id": str, "trust": float, "stable_traits": dict}

SimulationConfig = {"condition": str, "model_id": str,
                    "max_generated_turns": int, "temperature": float,
                    "max_output_tokens": int, "random_seed": int}

SimulationResult = {"run_id": str, "conversation_id": str,
                    "condition": str, "transcript": list[DialogueTurn],
                    "generated_turns": list[DialogueTurn],
                    "state_history": list, "status": str,
                    "run_metadata": dict}

EscalationScore = {"run_id": str, "score": float,
                   "judge_model_id": str, "rubric_version": str}
```

- `transcript` contains the original opening plus generated turns, in order.
- `generated_turns` contains only newly generated text.
- All run IDs are unique.
- `run_metadata` records model/prompt/updater versions, parameters, seed, errors, token usage, and available latency/cost information.

## 5. Required module interfaces

```python
load_and_prepare(source: str, config: DataConfig) -> list[LabeledExample]

initialize_personas(configs: dict[str, PersonaConfig]) -> dict[str, AgentState]

# Model adapter / reply generator: MUST receive current trust in the persona state.
generate_reply(history: list[DialogueTurn], actor: AgentState,
               config: GenerationConfig) -> DialogueTurn

# Called when an agent observes a partner's message; only prior/current turns allowed.
update_state(current_state: AgentState, observed_turn: DialogueTurn,
             visible_history: list[DialogueTurn], config: UpdaterConfig) -> StateUpdateResult

run_simulation(seed: ConversationSeed, personas: dict[str, PersonaConfig],
               config: SimulationConfig, generator: ResponseGenerator,
               updater: StateUpdater) -> SimulationResult

score_escalation(run_id: str, transcript: list[DialogueTurn],
                 config: JudgeConfig) -> EscalationScore

evaluate_conditions(examples: list[LabeledExample],
                    runs: list[SimulationResult],
                    scores: list[EscalationScore]) -> EvaluationSummary
```

**Who calls whom:**

- `run_experiment.py` calls the data loader, creates paired configurations, calls the simulator twice, calls the blind judge on each transcript, then calls evaluation.
- `simulator.py` calls `generate_reply()` and `update_state()`; it does **not** call the judge.
- The judge does not look up real labels; `evaluate.py` does that final join.

**Trust timing decision:** A trust update belongs to the agent *receiving* a partner's message, not automatically the agent that just wrote it. Confirm whether both speakers update, and how the existing seed turns initialize state. Each agent has independent state. Both conditions always see the same amount of chat history; do not add a separate long-term memory system to only one condition.

## 6. Experimental safeguards (non-negotiable)

1. Match seed, speaker roles, starting personas, model/version, turn budget, and generation settings between static/dynamic; only the explicit state-update policy differs.
2. Dynamic trust must affect later response generation, not just be logged. Static trust stays fixed; both agents can still respond to history.
3. Never use future real human turns or final attack labels in generation, state updates, or judge inputs.
4. Give the judge only ordered dialogue text and its fixed rubric. Strip condition names, trust values, model provenance, and reference labels from judge-facing payloads.
5. Score the same phenomenon as the reference label. For CGA-WIKI, the recorded outcome is whether a conversation contains a personal attack. Decide whether the judge predicts personal-attack presence in the generated continuation or uses an escalation-severity proxy. Do not silently equate the two.
6. Keep related or paired source conversations within the same evaluation split; prefer source-provided split information when appropriate.
7. Log failures; do not drop failed runs without reporting them. Do not treat repeated rollouts on the same original conversation as independent human examples.
8. AUC needs multiple labeled positive and negative source examples; one or five paired runs are software smoke tests, not evidence of predictive improvement.
9. Validate judge behavior on independently reviewed transcripts; scores in 0..1 are not automatically calibrated real-world probabilities.
10. Freeze prompts, evaluation thresholds, and state-update policies before held-out testing. Do not expose secrets in the repo/logs.

## 7. CGA-WIKI dataset checks before declaring it ready

- ConvoKit download name: `conversations-gone-awry-corpus`.
- Record the exact meaning of `conversation_has_personal_attack` and per-turn `comment_has_personal_attack`.
- Ignore comments marked `is_section_header`; inspect chronological order, speaker identities, and `reply_to` structure.
- Inspect existing `split` and `pair_id` fields; prevent paired/related examples from leaking across splits.
- Check whether selected openings are genuinely pre-attack and contain the required speaker structure. Do not assume every conversation is a clean two-person alternating exchange.
- Preserve raw data and source identifiers. Export a small valid sample in the common schema for everyone else.

Reference: <https://convokit.cornell.edu/documentation/awry.html>

## 8. Development/integration approach

**Before parallel coding:** Approve primary decisions; publish `schemas.py`, `sample_seed.json`, `sample_result.json`, and a mock LLM client.

**Parallel after contracts freeze:** Data parser; reply generator; trust updater; simulator (with mock clients); blind judge (with sample transcripts); metric code (with fabricated labeled rows).

**Integration (depends on those outputs):** Connect the modules, run one static/dynamic pair with mocks, then live models on five real openings. Audit logs and judge blindness. Run a larger pilot only if these checks pass.

**Minimum integration success:** From one real conversation opening, both conditions produce valid saved transcripts and state histories; the blind judge scores both; analysis joins their scores to a separately stored real outcome. Provide a single command and a reproducible result artifact.

## 9. Open decisions for October 13 team meeting

- [ ] Source dataset version, label semantics, seed cut-off, and how to handle >2 speakers/branches.
- [ ] Model/API access, budget, fixed model parameters, and retry policy.
- [ ] Which agent(s) update trust, the trust meaning, and first simple update method.
- [ ] Exact scoring target, judge rubric, and human-validation plan.
- [ ] Number of turns/repeats, run termination rule, and splits.
- [ ] Owners of data/integration, repository branch/PR process, and report coordinator.

The optional mathematical trust relationship is not an implementation prerequisite. Keep state logs now to support it later after TA approval.
