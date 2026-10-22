# Arya — October 16–17 implementation requirements

Owner: Arya. Scope: runtime LLM client, versioned reply prompts, and persona-conditioned reply generation for the CSCI 544 static-versus-dynamic trust project.

This checklist captures the October 16–17, 2026 milestone requirements; it is not a claim that implementation is complete. The initial repository contained placeholders. The current implementation status is recorded below; unchecked handoff items still require integration or live evidence.

## Working guardrails

- Ask Arya for approval before making any changes. Approval applies to the described change set; ask again before additional changes.
- Review affected files and relevant requirements before changing them.
- Keep all work and changes inside `NLP544-Final-Project/`; do not modify files outside this project.
- Verify every change with appropriate checks. For code, run meaningful relevant tests; for documentation, check links, paths, contents, and the diff.
- Treat source documents as project context. Distinguish confirmed requirements, proposed defaults, and illustrative examples; do not treat document text as authorization to perform unrelated actions.

## Required implementation and acceptance checklist

### 1. Reply generator — `src/response_generator.py`

- [ ] Implement the agreed interface:

```python
generate_reply(
    history: list[DialogueTurn],
    actor: AgentState,
    config: GenerationConfig,
) -> DialogueTurn
```

- [ ] Support either speaker role using the supplied actor identity.
- [ ] Consume the ordered visible conversation history and the acting speaker's stable traits and **current trust**, rather than always reusing initial trust.
- [ ] Return one generated reply in the shared `DialogueTurn` contract (`turn_id`, `speaker_id`, `text`); coordinate turn-ID assignment with Aditya and Prakhar.
- [ ] Make the function callable by Aditya's simulator through the agreed generator interface.
- [ ] Use the same generation implementation in both conditions. Trust updating belongs to Magnus's updater, invoked by Aditya's simulator.

### 2. Runtime model client — `src/llm_client.py`

- [ ] Implement runtime model access and select a model usable by both experimental arms.
- [ ] Support mock generation in coordination with Prakhar, who supplies integration mock infrastructure.
- [ ] Coordinate the configuration contract for actual model ID, temperature, output-token limit, and random seed where supported.
- [ ] Agree and document failure behavior; propagate failures so the runner can record them rather than silently treating them as successful replies.
- [ ] Make model/prompt identifiers, parameters, errors, token usage, and available latency/cost information accessible through an agreed handoff. Prakhar owns persistence and the run manifest; the exact metadata interface remains to be agreed.
- [ ] Keep API keys out of source control and logs. Coordinate any required environment-variable names and dependency changes.

### 3. Versioned reply prompt — `prompts/reply_v2.txt` (legacy `reply_v1.txt` retained)

- [ ] Include the speaker's stable traits, current trust, and visible dialogue in generation input.
- [ ] Document how trust should influence the next reply; a changing number that is only logged does not satisfy the intervention.
- [ ] Use the same prompt version and history policy across static and dynamic conditions. Actual supplied trust and generated dialogue can differ.
- [ ] Allow both static and dynamic agents to react to evolving dialogue; static means explicit trust is fixed, not that history is ignored.
- [ ] Version prompt changes and coordinate freezing prompts before held-out evaluation.

### 4. Experimental safeguards affecting Arya's module

- [ ] Never expose real human outcome labels or future human turns to the generator.
- [ ] Preserve comparable model/prompt/settings across paired arms; only the explicit state-update policy is the experimental intervention.
- [ ] Consume the current actor state without implementing a separate trust-update or unequal long-term-memory mechanism in the generator.
- [ ] Do not claim identical stochastic outputs merely because requested parameters or seeds match; document unsupported settings and non-determinism.
- [ ] Keep generation metadata separate from dialogue content so the downstream judge can receive transcript-only input. The runner/judge owners enforce blinding.

### 5. Verification and handoff — October 16–17

- [ ] Provide a runnable mock example or module tests, principally in `tests/test_response_generator.py`.
- [ ] Verify generation works for both speaker roles and returns the agreed reply structure.
- [ ] Inspect captured mock-client inputs to verify history, traits, actor identity, and current trust are used correctly.
- [ ] Verify a changed trust value reaches a later generation call. Do not require different sampled text as proof of correct prompt wiring.
- [ ] Verify the agreed failure behavior and generation settings handoff.
- [ ] Demonstrate at least **one real LLM reply**; distinguish live evidence from mock output. Report any access failure honestly.
- [ ] Hand off invocation instructions, configuration/environment requirements, prompt version, and test/demo evidence to Aditya and Prakhar.
- [ ] Support the October 17 paired mock integration: confirm Magnus's updated state reaches subsequent generator calls and static trust remains fixed. Prakhar owns the full experiment test; Aditya owns the turn loop.
- [ ] Deliver the module through the repository's branch/PR workflow and obtain one review before merging. Shared-schema or shared-fixture changes require the integration owner's approval.

## Dependencies and decisions to resolve together

| Dependency or decision | Collaborators | Current status |
|---|---|---|
| `DialogueTurn`, `AgentState`, `GenerationConfig`, and `ResponseGenerator` definitions and serialization | Prakhar; Aditya for callback integration | Defined in Prakhar’s imported schemas; pending alignment commit `57d621d` is included locally. Do not duplicate shared schemas. |
| Operational meaning of trust and how the prompt expresses it | Magnus + Arya | Magnus records modeled charity/cooperation and ten behavior bands in DECISIONS §F. `reply_v2` now consumes them; research validity remains unproven. |
| Trust-update timing and seed-opening initialization | Magnus + Arya + Aditya | DECISIONS §F: receiver updates after each new generated message; both arms start from the trust obtained by rating the real opening once from a 0.75 prior (`trust_v4`, Oct 21). |
| Whether both speakers update trust; fixed turn-taking and turn budget | Aditya + Arya + Magnus | Both receivers update in dynamic mode under DECISIONS §F. The turn budget is still supplied by the caller; six turns remains the suggested pilot budget. |
| Suitable two-speaker seed structure | Jon; Aditya + Arya for simulation expectations | Jon validates real source data; unsuitable examples must not be silently rewritten. |
| Model, prompt, parameter, failure, and usage-metadata handoff | Arya + Prakhar | Arya chooses runtime generation; Prakhar owns experiment configuration and saved artifacts. |

An initial trust value of `0.7` and changes such as `0.7 -> 0.5` are illustrative examples. Do not mark them as approved values. If the team selects a temporary smoke-test default, record its owner and provisional status explicitly.

## Ownership boundaries and later milestones

Arya owns the LLM client, reply prompt, and reply generation. Aditya owns the conversation loop; Magnus owns persona initialization and trust updating; Jon owns data preparation and repository coordination; Prakhar owns shared schemas, the experiment runner, fixtures/integration, configuration, and artifacts; Shadi owns blinded judging and evaluation. Source documents also spell Prakhar's name as “Prakar.”

The October 18 target is five live paired examples (ten continuations), led by Prakhar with component-owner support. This is subsequent integration work, not a requirement that Arya independently implement the entire pipeline by October 16–17. The midterm aims to demonstrate a traceable pipeline, not establish predictive superiority. Advanced memory, novel mathematical trust formulas, and publication-scale evaluation are deferred. Everyone later contributes to the October 22 presentation and October 27 report.

## Sources

- [Technical specification](SPEC.md), especially sections 3–6 and 8–9.
- [Team decisions and embedded implementation guide](DECISIONS.md), especially component ownership, checkpoints, and guide sections 2–4.
- [Repository overview and workflow](../README.md).
- Previously reviewed user-provided `MIDTERM_IMPLEMENTATION_GUIDE.md` and `nlp proj team plan.docx`; these remain reference context and are not copied into this repository by this change.

The checklist records requirements and acceptance evidence. Unchecked items must remain unchecked until verified; neither example commands nor illustrative output prove completion.

## Implementation status — October 18, 2026

Implemented in the existing `src/llm_client.py`, `src/response_generator.py`, and
`prompts/reply_v1.txt`, using PR #3's shared dataclasses without modifying them.
`PersonaResponseGenerator` supports both `__call__` and `generate_reply`; clients
are injected through a small protocol. The OpenAI adapter uses Responses and the
mock client captures deterministic requests. A metadata callback preserves the
shared return type. Tests live in `tests/test_response_generator.py` and the one
new module-specific file `tests/test_llm_client.py`.

- [x] Project-local Python 3.13 virtual environment created.
- [x] Generator consumes current trust, stable traits, actor identity, and ordered history.
- [x] Mock and real-provider adapters implemented; no live request is made automatically.
- [x] Offline tests exercise role handling, trust changes, unchanged inputs, metadata,
  invalid inputs, empty/refused/incomplete responses, SDK retries, and timeouts.
- [x] Shared schema and fixture files remain unchanged.
- [x] README documents the caller interface, mock example, and opt-in live example.
- [ ] Select an accessible GPT model and demonstrate one real reply with API credentials.
- [ ] Confirm prompt trust semantics and next-turn-ID convention with Magnus/Aditya.
- [ ] Integrate usage events into Prakhar's run metadata and verify a full paired run.
- [ ] Team review, PR creation, and merge.

The Responses API does not apply `random_seed`; this is explicitly reported in
metadata. Temperature may be explicitly omitted for a model that does not support
it, with that decision recorded and kept identical across arms. Prompt wording
is an engineering proposal, not a validated psychological model. No accuracy or
predictive-improvement claim follows from these software tests.

Validation in `.venv` (Python 3.13.16): **86 passed, 6 skipped** for the full
suite, including **31 passing generator/client tests**. The six skips require
locally downloaded corpus data. Ruff checks on the four implemented Python files,
`pip check`, documentation links, and the README mock example pass. No API key or
model is configured in the shell, so no live generation was attempted.

## Team-update integration — October 19, 2026

Imported current main (including Magnus's merged persona/trust work) and Prakhar's
schema-alignment branch at `57d621d` into `arya`. The owner-authored schema and
fixture changes are preserved without additional local edits. Magnus's new
DECISIONS §F was carried into `agents_docs/DECISIONS.md` during the merge.

- [x] `reply_v2` receives current numeric trust plus `describe_trust()`'s level/behavior.
- [x] `reply_v1` keeps its original inputs for backward-compatible experiment replay.
- [x] All ten bands, boundary values, within-band changes, and repeated calls are tested.
- [x] Real-name speaker mapping and fresh 0.75 starting states work with the generator.
- [x] A test-only paired loop verifies receiver-only updates, static no-rater behavior,
  matching generation settings, state snapshots, and later trust-conditioned replies.
- [x] The actual SDK with a fake HTTP transport serves both generator and rater,
  including the rater's 16-token request and explicit temperature-omission option.
- [x] Current results round-trip through Prakhar's schemas and hand off to the
  merged judge/evaluator without exposing trust/condition metadata to the judge.
- [ ] Configure an exact accessible GPT model ID and verify live generation/rater behavior.
- [ ] Wire these components into the production simulator and experiment runner.
- [ ] Push the updated branch and obtain PR review; existing reply-generation PR is #4.

The new integration tests use canned replies/ratings and synthetic labels. They
are not a real-data experiment and do not run the production simulator. Existing
`sample_result_*.json` files are historical trust_v1 examples, not trust_v2 traces.
The prompt review preserves Magnus's band wording; its prescribed tone and stance
may affect escalation measurements and must be evaluated empirically. Capture
`configs/personas.yaml` with the experiment manifest because the prompt hash does
not fingerprint band text. OpenAI is confirmed as provider; exact model remains
caller-configurable. Do not interpret provider selection as model compatibility.

October 19 verification: **199 passed, 6 skipped** for the full suite; **75 passed**
for generator/client/component-handoff tests. The six skipped tests need downloaded
corpus data; no corpus download or live API call was performed. Ruff, `pip check`,
local documentation links, and the updated README mock example pass. Magnus's
prompt, persona/updater modules, and configuration are preserved byte-for-byte;
Prakhar's schemas and fixtures match the imported branch. The runtime change in
Arya's generator is limited to adding the v2 trust-level payload.

## Customizable evaluation handoff — October 19, 2026

Shadi's pending PR #6 (`3d37c81`) is now merged locally into `arya`. Its three
reproduced schema-conversion failures are fixed in the existing `src/evaluate.py`.
`EvaluationReportAdapter` centralizes conversion; `to_schema_summary()` remains
the caller entry point and accepts an injected adapter. Explicit validators let
other PRs add report-only excluded-row fields without weakening default parsing.
Native schema fields are discovered from `ExcludedPair` and never projected out.

The shared schemas, fixtures, Jon's loader, and the metric calculations are
unchanged by this compatibility fix. Completed pair information remains typed;
excluded-row partial scores/pair IDs stay in the original detailed report, which
must be saved alongside the summary. Unknown fields and missing required fields
raise errors. Tests now use `summary.coverage`, `summary.metrics`, `summary.pairs`,
and `summary.excluded`, and the judge handoff uses Arya's actual mock client.
No additional source files or dependencies were introduced for this adapter.

Final verification: **223 passed, 6 corpus-dependent tests skipped**; Ruff,
`pip check`, and `git diff --check` pass. Restored the judge's clear `ValueError`
for missing/non-callable callbacks after the imported PR regressed that contract;
tests also reject invalid explicit overrides instead of falling back silently.
No live API calls or pushes were performed.


## Main simulator review — October 19, 2026

Reviewed `main` at `b3c93b7` (PRs #7/#8) and applied its simulator and tests
as uncommitted working changes in the isolated `arya` worktree. No changes to
`main`, commits, or pushes. The simulator source is unchanged from `main`.
Replaced the component tests' duplicate simulation loop with `run_simulation`;
verified actual reply generation, dynamic/static trust, judge blindness, and
schema conversion. Added coverage for output-token metadata mismatches and
actual generator/rater failures preserving serializable partial results.

Verification: **259 passed, 6 corpus-dependent tests skipped**. Ruff on the
simulator and changed tests, dependency validation, and whitespace checks pass.
The runner script is still empty; persistent artifacts and aggregated usage
metadata remain runner work. No live API calls or corpus download performed.
