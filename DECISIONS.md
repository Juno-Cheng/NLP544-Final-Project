# Team Decisions and Action Plan
**CSCI 544 | v0.3 | Updated October 13, 2026 | Confirmed ownership; proposed technical defaults**

## Purpose and confirmed scope
This is the **team's action tracker**, not an individual teammate's presentation script. Each owner is responsible for their component, and technical choices are agreed collaboratively.

**Confirmed midterm objective:** Demonstrate **one working end-to-end static/dynamic paired experiment**: real conversation opening -> two simulated continuations -> blinded judge scores -> comparison artifact. The midterm does **not** require proving the hypothesis or deriving a novel trust formula. Postpone the additional research question, formula discovery, and advanced memory/mechanism comparisons until after the midterm.

**Confirmed dates:** October 14 (shared contracts and usable example), October 16-17 (individual components and mock integration), October 18 (target live paired smoke test), **October 22 at 4:00 PM** (presentation), **October 27** (report). The team already accepted the component ownership and first implementation targets. The October 20 pilot is conditional on pipeline readiness.

**Writing:** Everyone contributes. Jon is listed as the slides/report coordinator; the assembly plan and final submission owner should be confirmed without making Jon responsible for writing everyone's sections. Course format: two slides, two-minute talk + one-minute Q&A; four-page, single-spaced, 10-point report.

**Related reference:** `SPEC_v0.3.md` defines technical contracts. `MIDTERM_IMPLEMENTATION_GUIDE.md` explains the decisions and what completed code should visibly produce.

## A. Remaining technical decisions (owners lead; team confirms)
The ownership and dates are **not being reopened**. Only three experiment-design decisions need a short group agreement before live implementation. Suggested defaults are **proposals**, not decisions already approved.

| Decision | Owners who lead | What to agree on (suggested default) | What to say in the call |
|---|---|---|---|
| **A1. Static vs. dynamic** | Aditya + Arya + Magnus | Two speakers continue the same opening, same initial personas/model/settings; only explicit trust updating differs. Both speakers can update their own trust in the dynamic arm. | "Can we keep both runs identical except that trust changes in dynamic mode and affects later replies?" |
| **A2. Trust state and timing** | Magnus + Arya, with Aditya | Start with one trust score per speaker in [0,1]; use a simple, documented causal update rule; receiving agent updates after reading the other's new message. Same conversation history in both arms. | "What does trust mean, when is it updated, and exactly how does Arya use it in the next prompt?" |
| **A3. Evaluation target** | Shadi + Jon | Define whether the judge rates attacks **in generated turns** or an escalation-risk proxy. Prefer a personal-attack-related score aligned with Jon's reference label; keep model judge blind. Agree on limitations of comparing short rollouts to original full-conversation labels. | "What does a 0-to-1 judge score mean, and are we comparing it to the same phenomenon as the real label?" |

**Implementation choices to resolve within the responsible pair, and document in the spec:**
- **Jon (data):** Verify CGA-WIKI split, pair/branch/speaker structure, label definitions, and a valid pre-attack two-speaker seed-selection rule. Source data must not leak future turns or outcomes.
- **Aditya + Arya (simulation):** Decide fixed turn-taking, how to handle unsuitable source conversations, and a provisional **six generated turns** per condition.
- **Arya + Prakar (experiment model/config):** Record actual runtime LLM, model ID, prompt version, temperature/token limit, costs, and failure behavior. Individual teammates may use whatever **AI coding assistants** they prefer; that is distinct from the model the experiment runs.
- **Prakar (pipeline):** Finalize shared data contracts, matched-run IDs/configuration, files/output layout, validation and tests.
- **Shadi (evaluation):** Freeze the judge rubric and judge-facing input format. Human-check a few transcripts when feasible; report limitations honestly.

**Deferred until after midterm:** New mathematical trust relation/research extension, learned state updaters, advanced memory systems, large mechanism comparisons, and publication-scale conclusions. These must not block the pilot.

## B. Confirmed component ownership and verifiable deliverables
| Person | Responsibility | What the person must hand off | Confirmed target |
|---|---|---|---|
| **Aditya** | A2A simulation orchestration | `simulator.py`: `run_simulation(...)` alternates speakers, invokes generator/updater and saves ordered `SimulationResult` with mock dependencies | **Oct 16-17** |
| **Arya** | LLM client, prompts, reply generation | `response_generator.py`: `generate_reply(...)` works for both speaker roles, uses current trust and history; live one-reply test | **Oct 16-17** |
| **Magnus** | Persona/state/memory | `persona.py`, `state_updater.py`: `initialize_personas`, static/no-update and dynamic/update state, per-speaker history; simple initial memory = shared visible chat history | **Oct 16-17** |
| **Shadi** | Blind escalation judge + evaluation | `escalation_judge.py`, `evaluate.py`: `score_escalation`, `evaluate_conditions`, blinding test, mock paired comparison | **Oct 16-17** |
| **Jon** | Dataset, repository coordination, writing coordination | `data_loader.py`: verified `LabeledExample` seeds/label audit; coordinate GitHub branch/PR workflow and collecting slide/report contributions | **Oct 14** dataset sample; **Oct 16** loader |
| **Prakar** | **Experimental Pipeline, Integration & Reproducibility** | `schemas.py`, `run_experiment.py`, integration tests: typed contracts/fixtures; paired static/dynamic runner; matched settings; output files + run metadata; failure checks; mock and live execution commands | **Oct 14** schemas; **Oct 17** mock pipeline; **Oct 18** live pipeline |

### What Prakar's role actually includes
1. **Contracts and fixtures:** Define, test, and maintain common schemas. Own contract changes, while Jon coordinates how they are pushed/reviewed in GitHub.
2. **Paired experiment engine:** For each real conversation seed, initialize identical starting personas; invoke Aditya's simulator twice (static and dynamic); preserve pairing, seeds, model/prompt/updater configuration and run IDs.
3. **Blind scoring handoff:** Pass **only transcript turns** to Shadi's judge; keep condition, trust history and real labels outside the judge input. Join them only for the final analysis.
4. **Reproducible artifacts:** Save transcripts, state histories, scores, summary rows, configuration/version manifest, and failures to trace a run.
5. **Integration tests:** Catch schema mismatch, missing partners, leaked labels, mismatched initial conditions, broken imports, and failures without silently dropping experiments.

**Boundary:** Prakar builds the runner and validates handoffs; **individual component owners fix their own bugs**. He is not the default person to implement other teammates' modules. Full restart/resume, parallel jobs and elaborate dashboards are optional later, not requirements for October 18.

## C. Dependencies and parallelism
**[FIRST]** = agree on contracts so modules match. **[PARALLEL]** = owners can work at the same time using mocks. **[INTEGRATE]** = requires multiple components to be compatible; owner of the joined pipeline is Prakar.

| Work | Tag | Owner | Prerequisite | Acceptance evidence |
|---|---|---|---|---|
| Select MVP design and freeze interfaces | [FIRST] | Aditya/Arya/Magnus + Shadi/Jon + Prakar | Three short technical decisions | Documented protocol and stable signatures |
| `schemas.py`, fixture examples, contract tests | [FIRST] | **Prakar** | Agreed core fields | Imports succeed; example JSON validates |
| Dataset parser and validated labeled seeds | [PARALLEL] | **Jon** | `ConversationSeed` contract (can stub before freeze) | Usable CGA-WIKI seeds; real labels stored separately |
| Simulation loop | [PARALLEL] | **Aditya** | `SimulationResult` contract; mock generator/updater | Two-speaker turn loop returns valid result |
| Persona-conditioned reply generation | [PARALLEL] | **Arya** | `DialogueTurn` and `AgentState` | One mock reply and one live LLM reply |
| Persona initialization and trust updater | [PARALLEL] | **Magnus** | `AgentState` and updater contract | Trust updates causal, bounded and logged; static unchanged |
| Blinded judge and metric functions | [PARALLEL] | **Shadi** | Transcript/score contracts | Mock transcript scores; condition and state not sent to judge |
| Paired config/runner + result storage | [PARALLEL] initially, then [INTEGRATE] | **Prakar** | Schemas first; stubs initially, real modules later | Same seed, initial personas/model and generation settings; paired artifacts and manifest |
| Complete paired **mock** run | [INTEGRATE] | **Prakar**, owners fix own modules | Compatible components or functioning stubs | One seed -> two transcripts -> two judge scores -> one comparison row |
| Paired **live** smoke test | [INTEGRATE] | **Prakar**, supported by all owners | Data, model access, agents, state, judge, real calls | Target five real seeds, ten LLM-generated transcripts and blinded scores, logged failures |
| Optional pilot analysis | [INTEGRATE] | **Shadi + Jon + Prakar** | Enough real positive/negative examples and valid scores | Preliminary table/plots or documented blockers (not a claim of significance) |
| Midterm slides/report | [PARALLEL], then [INTEGRATE] | **Everyone**; Jon coordinates assembly | Methods can be drafted now; results follow runs | Two slides by Oct 21; four-page report by Oct 27 |

**Data flow:** Jon's `ConversationSeed` -> Prakar's `run_experiment.py` -> Aditya's `run_simulation()` -> Arya's `generate_reply()` and Magnus's `update_state()` -> Aditya's `SimulationResult` -> Shadi's `score_escalation()` -> Shadi's `evaluate_conditions()` with Jon's **separately stored** real label. The runner enforces pairing and saves the result set.

## D. Confirmed checkpoints and what "done" looks like
| Date | Observable checkpoint | Accountable owner |
|---|---|---|
| **Oct 13** | Confirm the three remaining technical decisions; record assumptions in `SPEC_v0.3.md` | Owners of each decision |
| **Oct 14** | `schemas.py` imports; JSON fixtures validate; contract tests pass; Jon demonstrates verified real seed(s) and label meaning. Repo workflow is communicated. | **Prakar** (schema/test), **Jon** (data/repo) |
| **Oct 16** | Each module has a runnable example/unit test with mock dependencies and is pushed; Arya confirms one real generation works | **All component owners** |
| **Oct 17** | One complete **paired mock** pipeline run saves **two** transcripts, states, **two** blind scores and **one** labeled comparison row | **Prakar**, with module owners |
| **Oct 18** | Aim for **five live paired examples** (ten generated continuations) with model calls, run manifest, state logs, blinded scores and comparison table | **Prakar** leads integration; each owner supports their part |
| **Oct 20** | If feasible, a small pilot with positives and negatives; otherwise honest engineering results and documented issues | **Shadi + Jon + Prakar** |
| **Oct 21** | Two slides assembled and presenter rehearsed; everyone reviews their contributions | **Jon** coordinates; **everyone** contributes |
| **Oct 22, 4 PM** | **Midterm presentation** | Assigned presenter(s) |
| **Oct 26** | Four-page report assembled/frozen internally | **Jon** coordinates; **everyone** contributes |
| **Oct 27** | **Midterm report submitted** | Confirm final uploader |

**Pass/fail principle:** An assertion that a module is "done" means working code **plus a runnable example/test**, not merely a design or a promise. Five live pairs are a **pipeline smoke test**, not proof dynamic personas predict better.

**Prakar's proposed demonstration commands** (not existing repo commands until implemented):

```bash
python -m pytest tests/test_schemas.py -v
python run_experiment.py --mode mock --limit 1
python run_experiment.py --mode live --limit 5
```

The specific CLI flags may be adjusted by Prakar; what matters is one reproducible command and inspectable outputs at each stage.

## E. Short final coordination checklist
- [x] Six component owners and initial targets assigned.
- [x] **Jon** on dataset; **Prakar** on schemas/pipeline/integration.
- [x] Aditya/Arya split the simulation engine and LLM generator.
- [x] The team chose a **working midterm pipeline** over the deferred research extension.
- [ ] Agree on the three technical points in **Section A** (or mark each as provisional).
- [ ] Confirm Magnus's interface and any details he missed while absent.
- [ ] **Prakar** publishes schema/fixture/tests by Oct 14; **Jon** coordinates repo process and provides the dataset sample.
- [ ] Define who gives final approval for interface changes after Oct 14.
- [ ] Jon coordinates slide/report assembly; confirm the speaker and final uploader.

## F. Trust design (owner: Magnus, decided Oct 18; resolves A2 for persona/state)
Implemented in `src/persona.py`, `src/state_updater.py`, `configs/personas.yaml`, `configs/updater.yaml` (`trust_v2`).

| Topic | Decision | Why |
|---|---|---|
| Meaning | Trust = modeled willingness to read the partner charitably and cooperate, 0..1 (matches Arya's `reply_v1` wording). | One definition across updater and prompt; not a measured human score. |
| Who updates | Both agents; the **receiver** of a new message updates, using only visible turns. | Symmetric; speaker order cannot bias the result. |
| Initialization | Both arms start at **0.75** after the opening; opening turns never update trust. | Cleanest contrast for the midterm. Replaying the opening through the updater is the first post-midterm experiment. |
| Personas | A and B share identical stable traits; the opening's first speaker is A. | Only the trust policy differs between agents and arms. |
| Signal | An LLM rater (same model as the generator, temperature 0) scores the observed message's cooperativeness 0..1 from the full visible conversation. It never sees trust. | Reads context; no feedback loop from trust into ratings. |
| Rule | `trust' = clamp(trust + 0.5 * (rating - 0.5))`, symmetric. | A neutral message (0.5 = polite, businesslike disagreement) leaves trust unchanged, so dynamic differs from static only because of what was said. |
| Prompt | Reply prompt gets the number plus one of **10 equal bands** (word + behavior sentence covering interpretation, justification, cooperation and tone); `prompts/reply_v2.txt`. | Words make trust changes visible to the model; the number keeps within-band changes. |
| Floor | The lowest bands are sharp and accusatory about actions but never instruct insults or personal attacks. | Escalation must be measured, not prompted, or the judge scores our band wording. |
| Rater failure | A non-numeric or out-of-range rating raises; the run is logged as failed. | Silently holding trust would turn dynamic into static. |
| Snapshots | `StateSnapshot(turn_id=k)` = trust after turn k **and** the receiver's update. | Matches the schema docstring; records each update causally. |
| Cost | `TrustUpdater(on_rating=...)` reports rater usage like the generator's `on_generation`. | Rater calls count toward run cost. |
| Validation | `scripts/check_rater.py` rates 10 attack + 10 civil train-split comments once before the live run; pass = every attack scores below the civil mean. | Labels used only to validate the rater, never in the experiment. |

All values are frozen once held-out testing starts; any change bumps `updater_version`.

# Midterm Implementation Guide — What We Are Actually Building
**CSCI 544 | October 13, 2026 | Informational companion, not a new specification**

> We now need a few concrete technical decisions and working examples, not another round of abstract project planning.

**Read this for understanding; use `TEAM_PLAN_v0.3.md` for owners/dates and `SPEC_v0.3.md` for the actual interface contract.** The examples here are explanatory/mock examples, not observed results or final code. None of the proposed defaults become team decisions until the responsible owners agree.

## 1. What we're trying to show at the midterm

A **single real conversation opening** is continued by two simulated speakers under **two conditions**:

- **Static:** the agents react to ongoing dialogue, but their explicit trust values do not change.
- **Dynamic:** the agents start identically, but their explicit trust values update from observed dialogue and influence later generated replies.

A **separate blinded judge** scores the generated conversations for personal attack/escalation. We save both scores and compare them to a **separately stored real human outcome label**. The midterm success criterion is a working traceable pipeline, **not evidence yet that dynamic is superior**.

**Deferred:** Additional publishable research question, mathematical trust relationship, sophisticated learned update mechanisms, advanced memory, and full-scale statistics. A small post-integration pilot may generate interesting preliminary observations, but we should call them preliminary.

## 2. The three decisions owners need to settle

### Decision A — What is the experimental contrast?
**Who:** Aditya (simulator), Arya (generation), Magnus (state). **Tag:** `[FIRST]` then each codes `[PARALLEL]`.

**Suggested starting point:** Two alternating AI speakers, both with separate trust variables, one shared human opening, same initial personas/model/settings, six generated turns in each condition. Only the **explicit trust-update policy** differs.

**Discuss:**
1. Are both speakers stateful in dynamic mode, or only one? **Recommended: both.**
2. When the original data has extra speakers, branches or turns, should Jon exclude that sample? **Recommended: verify a clean two-speaker usable sample first, without inventing data.**
3. What is held constant? **Same opening, initial traits/trust, generator version, generation parameters, turn budget and history policy.** LLM sampling may produce different messages even with matching seeds/settings.

**Natural way to ask:** "Can we make the first comparison really controlled? Same opening and personas, and the only extra mechanism in dynamic is the trust update that actually feeds into the next reply."

### Decision B — How does trust change, and how does generation use it?
**Who:** Magnus (owns trust), Arya (must use it in prompts), Aditya (calls both modules). **Tag:** `[FIRST]` policy then `[PARALLEL]` implementations.

**Suggested starting point:** A bounded trust value `0.0..1.0` for each agent, plus stable traits that never change. Initial trust might be `0.7` for a mock test (**illustrative**, not evidence of a human trust score). Use a simple, documented rule for the first implementation; it is a baseline engineering mechanism, not a validated psychological model.

**One generated-turn sequence:**
1. Aditya asks Arya's generator for B's next message using **B's current trust + dialogue so far**.
2. Aditya appends B's message to the transcript.
3. Magnus's updater processes B's message for **A**, the receiver, using only already-visible dialogue. A's trust changes **only in dynamic mode**.
4. When A speaks, Arya gets A's **updated** trust and current transcript. The prompt tells the model how to use that value.
5. Aditya saves both the dialogue and per-agent state history.

**Example only:** B says something dismissive, A's dynamic trust changes `0.7 -> 0.5`, and A's next reply is generated with `trust=0.5`. In static mode A stays at `0.7` but still sees and reacts to what B said. The team must agree how to initialize trust relative to the opening turns.

**Critical pass/fail check:** An updater that merely logs a changed number, without the generator **consuming** it, does not implement the dynamic intervention.

**Natural way to ask:** "Magnus, what's the simplest trust update we can test? Arya, what will your prompt do with that number? Aditya, when exactly will you call the updater?"

### Decision C — What does the judge's number mean?
**Who:** Shadi (judge/evaluation), Jon (human labels). **Tag:** `[FIRST]` definition then `[PARALLEL]` code.

**Suggested starting point:** A blinded LLM judge produces a consistent `0..1` **score indicating evidence or risk of a personal attack in the generated continuation**. Do not call it a calibrated probability without calibration. Keep the definition fixed across both conditions.

**Example judge input:**

```text
A: Why did you remove my edit?
B: It wasn't relevant.
A: Could you at least explain why?
B: You don't understand the basics.
```

The judge sees **only the ordered dialogue and the scoring rubric**, not whether it was static/dynamic, the trust series, or the real outcome label. A hypothetical output would be `score=0.72`.

**What the evaluator does next:** For each source conversation, combine `real_outcome`, static score and dynamic score into one paired comparison row. Compare across examples when both label classes are available. ROC-AUC on five examples, if computable at all, is **not strong evidence** of improvement.

**Important label-alignment caveat:** Cornell's real label marks a personal attack in the original human conversation, often across a longer dialogue than the generated six-turn continuation. Jon and Shadi must define that difference and decide whether the judge scores generated **attack presence**, **attack risk**, or another explicitly justified proxy. Do not silently treat them as identical.

**Natural way to ask:** "Jon, what's the exact reference-label definition and what part of the conversation does it cover? I'll align the judge rubric with that, and we'll write down the remaining mismatch as a limitation."

## 3. Who passes what to whom?

```text
Jon:      1 real labeled source conversation
            |-- seed opening --------------------------+
            |-- real label (held back)                |
            v                                        v
Prakar:  run_experiment.py -> create matched configurations
            |                    |
            v                    v
Aditya:  static simulation      dynamic simulation
            |                    |
Arya:     generate_reply()      generate_reply()
Magnus:   trust stays fixed     trust updates causally
            |                    |
            v                    v
          1 static transcript   1 dynamic transcript
            |                    |
Shadi:   score_escalation()     score_escalation()  [blind]
            |                    |
            +----------+---------+
                       v
Shadi:     evaluate_conditions() + Jon's real label
                       |
Prakar:    save comparison row, manifest, logs, failures
```

**Two orchestrators, two distinct jobs:** Aditya runs the **inside of a simulated conversation** (the repeated speaker-turn loop). Prakar runs the **whole experiment across seeds and conditions** (pairing, calling simulation, judging, saving, tests). Prakar does not take over everyone's code.

**Jon vs. Prakar:** Jon handles the dataset and **GitHub process/coordination**. Prakar **authors the shared schemas** and the experimental pipeline; Jon can ensure the files/PRs are visible in the repo. This resolves the older notes that put "By Jon" next to the schema checkpoint while assigning `schemas.py` to Prakar in the ownership table. If the team truly intends Jon to author the schema instead, agree on that explicitly and change both documents together.

**AI model distinction:** Teammates may use their own coding assistants. For the **experiment**, both static and dynamic agents need comparable model/prompt/settings; Arya owns the runtime generator, and Prakar records the exact settings.

## 4. Concrete acceptance tests by date

### October 14 — Shared foundations
**Owners:** Prakar (schema/fixtures/contracts); Jon (verified seed sample and repo workflow). **Tag:** `[FIRST]`.

**What files should exist:** `schemas.py`, sample fixtures such as `fixtures/sample_seed.json` and `fixtures/sample_result.json`, and a small schema test. Jon should have a real CGA-WIKI-derived sample with ID, opening turns, correctly explained label and split/pair fields where available. The real label is *not* in the object handed to an agent.

**Suggested command** (Prakar to implement, not currently guaranteed):

```bash
python -m pytest tests/test_schemas.py -v
```

**Pass =** all sample objects validate/import; Jon can show actual usable source records; each teammate can start their component without inventing an incompatible field name. **Not required:** actual full conversations or real LLM generations.

### October 16–17 — Individual modules + paired mock pipeline
**Owners:** Each owner produces a module test; Prakar stitches together a mock pipeline. **Tag:** `[PARALLEL]` for implementation -> `[INTEGRATE]` for whole mock run.

**Mock** means canned/fabricated responses replace unfinished or paid LLM calls, so you test **data flow, not realism**.

**Suggested command** (not yet implemented):

```bash
python run_experiment.py --mode mock --limit 1
```

**Example successful output (illustrative):**

```text
Loaded 1 labeled example [fixture]
STATIC: 6 generated turns, saved
DYNAMIC: 6 generated turns, saved
State history validated: static unchanged; dynamic updates recorded
Blind judge: 2 mock scores saved
Paired evaluation: 1 joined comparison row saved
Manifest + errors: written
RESULT: PASS
```

**Files to inspect:** A saved static `SimulationResult`, a saved dynamic `SimulationResult`, two `EscalationScore` records, one paired comparison record, and configuration/run metadata. A smoke test should prove trust updates **reach** later generated calls, not that the canned dialogue becomes realistic.

**Pass =** the entry command terminates successfully, test checks pass, both run IDs can be joined to the correct original conversation, and no label/condition/trust metadata appears in judge-facing payloads.

### October 18 — Live end-to-end smoke test
**Owners:** Prakar leads the pipeline; Jon supplies real seeds; Aditya/Arya/Magnus own the simulator behavior; Shadi owns judging/evaluation. **Tag:** `[INTEGRATE]`.

**Suggested command** (not yet implemented):

```bash
python run_experiment.py --mode live --limit 5
```

**Expected process:** Five **verified real** opening dialogues -> five static continuations and five dynamic continuations using **actual LLM generation** -> ten blinded judge scores -> paired comparison table with five rows and the original held-out labels. Save model/prompt/updater versions, matching initial states/settings, per-turn trust history and failures.

**Illustrative result row (NOT real data):**

```json
{
  "conversation_id": "example_001",
  "real_outcome": 1,
  "static_score": 0.35,
  "dynamic_score": 0.82,
  "status": "paired_complete"
}
```

**Pass =** evidence that all modules work together on actual examples, with inspectable transcripts, actual LLM replies, condition-blinded judging and matching results. If an API/model is unavailable, log the blocker and preserve the successful mock demonstration. Do **not** claim an accuracy improvement based on a smoke test.

### October 20–22 — Optional pilot and presentation
**Owners:** Shadi/Jon/Prakar for any pilot evidence; **everyone** contributes slides/report, Jon coordinates assembly.

If possible, run a small balanced set of real positive/negative outcomes and report preliminary scores plus limitations. If not, present the verified pipeline, protocol, demo artifacts, and realistic next steps. Two slides are due for the team presentation **October 22 at 4 PM**. Report due **October 27**.

## 5. Work that can happen at the same time

| Task | Owner | Tag | Needed first |
|---|---|---|---|
| Validate data/source labels | Jon | `[PARALLEL]` immediately | Cornell data access |
| Define schema and test fixtures | Prakar | `[FIRST]` | Owner consensus on fields |
| Build conversation loop with mocked callbacks | Aditya | `[PARALLEL]` | Provisional signatures |
| Generate a reply from a persona-conditioned model | Arya | `[PARALLEL]` | History/state fields and LLM access |
| Build bounded trust updater and test it | Magnus | `[PARALLEL]` | Trust semantics/update timing agreement |
| Build judge/rubric and analysis on fake samples | Shadi | `[PARALLEL]` | Transcript schema and judge target |
| Build experiment runner using mock modules | Prakar | `[PARALLEL]` initially | Shared schema, mock adapters |
| Execute complete mock pair | Prakar + owners | `[INTEGRATE]` | Compatible interfaces |
| Execute live pair(s) | Prakar + owners | `[INTEGRATE]` | Real dataset, generator, state, judge, settings |
| Draft background and methodology | Everyone; Jon coordinates | `[PARALLEL]` | Proposal/spec |

**Don't block:** Shadi doesn't need Aditya's finished agents to code metrics; Aditya doesn't need Arya's model to test a turn loop; Prakar doesn't need a live API to test experiment pairing.

## 6. Five-minute sanity check before calling the pilot ready

- [ ] A real Cornell opening and its label are verified by Jon.
- [ ] Static and dynamic start with the same persona values and generator settings.
- [ ] The **dynamic** state changes through causal updates and gets passed to the next agent reply; **static** explicit trust remains fixed.
- [ ] All outputs are labeled with the correct conversation/run IDs, while labels are hidden from both agents and judge.
- [ ] The judge gets dialogue only, and the same rubric is used in both conditions.
- [ ] Prakar's runner saves an inspectable comparison row, logs configuration, and records failures.
- [ ] Everyone can run the example from the repository instructions without guessing missing inputs.

**Bottom line:** Every person owns a distinct, testable engineering/research deliverable. The purpose of the midterm is to show the pieces work together, not to put one teammate in charge of knowing or explaining everything.
