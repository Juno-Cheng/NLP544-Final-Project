# Static vs. Dynamic Personas - Technical Specification
**CSCI 544 | v0.3 | Updated October 13, 2026 | Ownership confirmed; some technical defaults still require approval**

## 1. Research goal and current scope
Compare whether **explicitly updated trust** in simulated conversations helps predict which real human conversations eventually contain personal attacks, versus keeping that trust fixed.

- **Static:** Responses use the evolving chat history, but explicit trust stays fixed.
- **Dynamic:** Same model, initial personas, and chat opening, but explicit trust updates as new messages arrive and the updated state affects subsequent responses.
- **Midterm deliverable (Oct 22):** A **demonstrable end-to-end pilot** with paired static/dynamic continuations, a blind escalation judge and a traceable comparison artifact. Engineering correctness and an honest explanation of limits are sufficient; a research conclusion is **not** promised by midterm.
- **Longer-term research goal:** Test whether dynamically updated trust improves predictions of real outcomes, with appropriate validation, analysis and uncertainty after the pipeline works.
- **Deferred until after midterm:** The additional mathematical trust research question, full mechanism comparisons, advanced memory, and publication-scale evaluation. The team will revisit the extension with the TA.

## 2. End-to-end flow and module handoffs

```text
Jon: Cornell CGA-WIKI -> data_loader.py -> LabeledExample
                                  | seed only (keep outcome label separate)
                                  v
Prakar: run_experiment.py -> initialize same personas/config for both arms
                             |                    |
                             v                    v
Aditya:                run_simulation(STATIC) run_simulation(DYNAMIC)
                           |                        |
Arya:                   generate_reply(...) on each generated turn
Magnus:                 update_state(...) after an agent observes a turn
                        (static fixed / dynamic causal updates)
                           |                        |
                           v                        v
                         SimulationResult        SimulationResult
                             \                    /
                              v                  v
Shadi:                    blinded score_escalation() twice
                              |                   |
Prakar:                   save scores, condition-specific run metadata
                              \                 /
                               v               v
Shadi:               evaluate_conditions() + separate Jon outcome labels
                               |
Prakar:                  save paired table, manifest and failure log
```

**Runtime ownership:** `run_experiment.py` is the experiment **orchestrator**, and `simulator.py` is the **within-conversation turn orchestrator**. Prakar owns the first; Aditya owns the second. Arya and Magnus implement the functions Aditya calls. Shadi's judge is independent of both.

**Information boundary:** The generator and state updater never receive real outcome labels or future human turns. The judge gets an ordered **transcript-only** view, not the condition, state, real label, simulation parameters or metadata. The analysis joins scores with labels **after** scoring.

## 3. Confirmed ownership and files
| Owner | Module(s) | Must hand off |
|---|---|---|
| **Aditya** | `simulator.py` | Speaker alternation, dialogue loop, injected generator/updater, `SimulationResult` per condition |
| **Arya** | `response_generator.py` | Runtime model client and prompts; `generate_reply(...)` that consumes conversation and speaker's current trust |
| **Magnus** | `persona.py`, `state_updater.py` | `initialize_personas`, stable traits, trust updating, per-speaker history; static branch unchanged |
| **Jon** | `data_loader.py`; GitHub/PR coordination | Valid `LabeledExample` records and pre-attack conversation seeds with labels held separately; shared repo workflow |
| **Prakar** | **`schemas.py`, `run_experiment.py`, integration tests/config/artifacts** | Reproducible **paired experiment runner**, fixtures, matching initial configurations, run IDs, saved transcripts/state/scores/manifest, validation, failure logs, one-command mock/live runs |
| **Shadi** | `escalation_judge.py`, `evaluate.py` | Blind transcript scorer, fixed rubric, test for evaluator blinding, label join and paired evaluation |

**Writing:** Everyone contributes; Jon coordinates gathering/assembling slide and report contributions as recorded in the team plan. The final uploader/presenter are separate assignments to confirm.

**Scope of Prakar's engineering contribution:** He owns the experiment as a reproducible software workflow, not merely import statements. He enforces pairing and comparable settings, manages configuration and result artifacts, and tests the entire system. Individual owners remain responsible for their own module bugs. Avoid blocking the October 18 smoke test on optional distributed execution, automated resumption, or dashboards.

**Boundary with Jon:** Jon coordinates repository workflow and delivers the dataset; **Prakar authors/maintains the shared schema** and determines whether integrated outputs pass the contract. Jon can publish the dataset-based fixture to GitHub; Prakar publishes/validates the schema-based fixtures and tests.

## 4. Common data definitions (freeze before independent coding)
Implement these with Pydantic or dataclasses in **one** `schemas.py`. Names and field meanings are the interface contract; implementation details may be agreed by the integration owner.

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

The `transcript` field contains the original opening **plus** generated turns in order; `generated_turns` contains only newly generated text. All run IDs are unique. `run_metadata` records model/prompt/updater versions, parameters, seed, errors, token usage, and available latency/cost information.

**Prakar's contract-completion task:** The signatures below mention `DataConfig`, `GenerationConfig`, `UpdaterConfig`, `StateUpdateResult`, `ResponseGenerator`, `StateUpdater`, `JudgeConfig` and `EvaluationSummary`. These are **not yet fully defined in v0.2**. Before calling contracts frozen, Prakar and the corresponding owners must either define these types or replace them with an agreed equivalent. Document serialization and the exact `run_experiment.py` CLI flags as well.

**Paired metadata (minimum fields, location to be agreed):** `experiment_id`, source `conversation_id`, `pair_id`/paired-run identifier, `condition`, actual generation model/prompt IDs, updater version, turn budget, temperature, random seed when supported, and run status/error. Same settings should be requested in both arms; stochastic outputs need not be identical.

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

**Who calls whom:** Prakar's `run_experiment.py` calls Jon's data loader, creates paired configurations, invokes Aditya's simulator **twice per seed**, calls Shadi's blind judge on each transcript, and finally calls Shadi's evaluation. Aditya's `simulator.py` calls Arya's `generate_reply()` and Magnus's `update_state()`; it does **not** call the judge. Only evaluation sees the human outcome labels.

**Isolation:** Two runs must start from **fresh copies** of the same persona configuration, not mutated objects reused from the first run. Preserve the seed, role mapping, history policy, model/prompt/temperature, turn budget and applicable random-seed settings across both arms. Record any non-determinism and failures rather than claiming exact reproducibility of generated text.

**Trust timing (proposed default; Magnus/Arya/Aditya to approve):** A trust update belongs to the agent *receiving* a partner's newly generated message, not automatically its author. Each agent has independent trust toward its conversation partner; dynamic-mode values update causally from already-visible turns, while static-mode explicit trust stays fixed. Both conditions always see identical *amounts* of conversation history, though the actual generated content may diverge. Decide whether to process the seed-opening messages into initial trust, or start from fixed post-opening initial states; implement the **same initialization rule** in both arms. Do not add unequal long-term memory systems.

## 6. Experimental safeguards (non-negotiable)
1. Match seed, speaker roles, starting personas, model/version, turn budget and generation settings between static/dynamic; only explicit state-update policy differs.
2. Dynamic trust **must affect later response generation**, not just be logged. Static trust stays fixed; both agents can still respond to history.
3. Never use future real human turns or final attack labels in generation, state updates, or judge inputs.
4. Give the judge only ordered dialogue text and its fixed rubric. Strip condition names, trust values, model provenance and reference labels from judge-facing payloads.
5. Score the **same phenomenon** as the reference label. For CGA-WIKI, the recorded outcome is whether a conversation contains a personal attack. Decide whether the judge predicts personal-attack presence in the generated continuation or uses an escalation-severity proxy. Do not silently equate the two.
6. Keep related or paired source conversations within the same evaluation split; prefer source-provided split information when appropriate.
7. Log failures; do not drop failed runs without reporting them. Do not treat repeated rollouts on the same original conversation as independent human examples.
8. AUC needs multiple labeled positive **and** negative source examples; one or five paired runs are software smoke tests, not evidence of predictive improvement.
9. Validate judge behavior on independently reviewed transcripts; scores in 0..1 are not automatically calibrated real-world probabilities.
10. Freeze prompts, evaluation thresholds and state-update policies before held-out testing. Do not expose secrets in the repo/logs.

## 7. CGA-WIKI dataset checks before declaring it ready
- ConvoKit download name: `conversations-gone-awry-corpus`.
- Record the exact meaning of `conversation_has_personal_attack` and per-turn `comment_has_personal_attack`.
- Ignore comments marked `is_section_header`; inspect chronological order, speaker identities and `reply_to` structure.
- Inspect existing `split` and `pair_id` fields; prevent paired/related examples from leaking across splits.
- Check whether selected openings are genuinely pre-attack and contain the required speaker structure. **Do not assume every conversation is a clean two-person alternating exchange.**
- Preserve raw data and source identifiers. Export a small valid sample in the common schema for everyone else.

Reference: https://convokit.cornell.edu/documentation/awry.html

## 8. Development/integration approach
**Midterm target is software evidence first**, not a claim of statistical improvement. See `MIDTERM_IMPLEMENTATION_GUIDE.md` for explanations, examples and expected outputs.

**Before parallel coding:** Approve primary decisions, publish `schemas.py`, a `sample_seed.json` and `sample_result.json`, and a mock LLM client.

**Parallel after contracts freeze:** Data parser; reply generator; trust updater; simulator (with mock clients); blind judge (with sample transcripts); metric code (with fabricated labeled rows).

**Integration (depends on those outputs):** Connect the modules, run one static/dynamic pair with mocks, then live models on five real openings. Audit logs and judge blindness. Run a larger pilot only if these checks pass.

**Minimum mock success (Oct 17):** From one fixture opening, both conditions produce valid saved transcripts/state histories; a mock blinded scorer returns two scores; analysis joins them with one separately held label. Tests verify matching initial states/configurations, static trust unchanged, dynamic trust passed into later generation, and evaluator blindness.

**Target live smoke test (Oct 18):** Use five verified real conversation openings to attempt **five matched static/dynamic pairs** (ten continuations) with the actual generation model and judge. Save the valid outputs **and** all failures and settings. If fewer succeed, report that honestly; do not fabricate a success. Five pairs are not a statistically persuasive test of the research claim.

**Artifacts for each run:** `SimulationResult` (including transcript and states), `EscalationScore`, a paired comparison row, configuration/model/prompt/updater manifest, and error/latency/cost logging when available. Suggested runnable commands (to be implemented by Prakar, not claimed to exist):

```bash
python -m pytest tests/test_schemas.py -v
python run_experiment.py --mode mock --limit 1
python run_experiment.py --mode live --limit 5
```

## 9. Outstanding decisions / explicitly provisional assumptions
**Confirmed:** Six owners and component targets; live-pipeline focus for midterm; deferred additional research question; dataset = Cornell CGA-WIKI pending Jon's structural validation; presentation = October 22 at 4 PM; report = October 27.

**Technical points still to approve (do not silently treat defaults as settled):**
1. **Aditya + Arya + Magnus:** Both speakers or one speaker update trust? How many generated turns? How are seeded conversations normalized to two speakers? Suggested MVP: both agents maintain their own state, fixed six generated turns, exclude/flag unsuitable seeds rather than silently modifying speaker structure.
2. **Magnus + Arya (+ Aditya):** Operational meaning/starting value of trust, simple updater signals, seed initialization policy, and exact way trust conditions the prompt. Suggested MVP: one bounded trust scalar with a documented causal update rule.
3. **Shadi + Jon:** Reference outcome definition vs. judge rubric, whether judge targets attacks in the generated portion, human sanity check, and limits of a short continuation vs. full real-conversation label.

**Delegated engineering details:** Arya chooses a runtime model the paired arms can both use; Prakar records the model/settings and owns interfaces/config/tests; Jon verifies dataset labels/splits and coordinates PR practices. Team members' individual AI coding assistants are not constrained by this experiment-level comparability requirement.

The optional mathematical trust relationship is **not** an implementation prerequisite and remains after-midterm research work. Keep state logs now to support it later after TA approval.
