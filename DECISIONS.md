# Team Decisions and Action Plan

**CSCI 544** | October 13, 2026

Meeting goals:

- An approved minimal experiment design
- One owner + deliverable + date for each component
- Shared code interfaces
- A named integration owner

Record each decision with the date and who agreed, and move items from proposed to [Decision log](#decision-log) once approved.

---

## A. Decisions to make together (proposed defaults, not yet approved)

| Topic | Proposed default |
|---|---|
| **Dataset** | Start with Cornell CGA-WIKI. Verify labels, original split, matched pairs, speaker/branch structure, and pre-attack openings. |
| **Opening** | Same selected early turns in both arms; no future human turns or outcomes passed to agents. |
| **Simulation** | Two LLM speakers, initial identical personas in static/dynamic. Define turn-taking and what to do with >2 original speakers; controlled partner later if needed. |
| **Trust** | Stable personality traits fixed in both; dynamic trust updates only using observed prior/current turns. Decide who trusts whom, whether both agents update, and how trust enters generation. |
| **Memory** | Same visible conversation history in both conditions; no extra long-term memory infrastructure needed for the initial pilot. |
| **LLM** | Choose provider/model, prompt versions, budget, temperature, output cap, retry behavior. |
| **Runs** | Provisional six generated turns; five paired seeds for smoke test; choose pilot size and whether repeated rollouts are feasible. |
| **Judge** | One separate, blinded evaluator. Agree whether the score targets personal-attack occurrence in generated continuation or a separately justified escalation proxy. |
| **Analysis** | ROC-AUC against original real binary labels when sufficient positive/negative examples exist; judge checks, paired comparisons, and caveats. |
| **Collaboration** | Confirm `schemas.py` owner, repo, PR workflow, mock examples, integration command, and report coordinator. |

## B. Component ownership

| Person | Responsibility | Handoff / acceptance condition | First target |
|---|---|---|---|
| Aditya | A2A simulation orchestration | `run_simulation(...)` creates/saves ordered turns and metadata using mock model + updater | Oct 16/17 |
| Arya | Reply generation, persona prompts, model client | `generate_reply(...)` works for either speaker and consumes current trust | Oct 16/17 |
| Magnus | Persona/state/memory (volunteered; **confirm deliverable asynchronously**) | `initialize_personas`, static/dynamic `update_state` + per-speaker state history | Oct 16/17 |
| Shadi | Blind escalation judge + evaluation | `score_escalation`, `evaluate_conditions`, blinded-input test, mock comparison | Oct 16/17 |
| Jon | Dataset **or** infrastructure (**choose tonight**) | If data: validated `LabeledExample`s, first real seeds, label audit | Oct 14 sample; Oct 16 module |
| Prakar | Shared schemas/integration **or** dataset (**choose tonight**) | If integration: `schemas.py`, sample fixture/mock client, executable paired pipeline and tests | Oct 14 schemas; Oct 18 end-to-end (TBC) |

Research writing is shared. Name one **report coordinator** to assemble the two slides and four-page report (not a seventh full-time coding role). Anyone absent may confirm or request reassignment in WhatsApp after the call.

## C. Dependencies and parallelism

**Legend:** `[FIRST]` needed before other components agree on formats; `[PARALLEL]` can be developed simultaneously with mocks; `[INTEGRATE]` needs real outputs from multiple modules.

| Work | Tag | Needs before starting | Deliverable |
|---|---|---|---|
| Agree experiment rules and freeze data contracts | `[FIRST]` | Team decisions | Accepted `SPEC.md`; shared signatures |
| Publish `schemas.py` + mock inputs/outputs | `[FIRST]` | Agreed core fields | Everyone can code to same contract |
| Dataset parsing and seeding | `[PARALLEL]` | Shared `ConversationSeed` type | Labeled seeds with labels separated |
| Persona and trust updater | `[PARALLEL]` | Shared `AgentState` / update interface | State updater tested with fabricated turns |
| LLM reply generation and prompting | `[PARALLEL]` | Shared `DialogueTurn` / state interface | One reply from mocked and live client |
| A2A simulation loop | `[PARALLEL]` | Shared interfaces; mock reply/update implementations | Saved `SimulationResult` |
| Blind judge + metric code | `[PARALLEL]` | Shared transcript and score schemas | Mock transcripts → judge score; mock rows → metrics |
| Mock end-to-end experiment | `[INTEGRATE]` | All required module interfaces work | One completed paired result |
| Live run on real examples | `[INTEGRATE]` | Verified data, model access, simulator, updater and judge | Five paired live runs plus logs |
| Pilot metrics and plots | `[INTEGRATE]` | Several real examples of both labels and judge outputs | Preliminary results or failure analysis |
| Slides/report | `[PARALLEL]` then `[INTEGRATE]` | Can draft background/method now; observations require runs | Midterm deliverables |

**Example handoff:** Jon's `ConversationSeed` → Aditya's `run_simulation()` → Arya's `generate_reply()` (and Magnus's `update_state()`) → Aditya's saved `SimulationResult` → Shadi's `score_escalation()` → Shadi's `evaluate_conditions()` using labels held separately. The integration owner connects the complete sequence in `run_experiment.py` and maintains common schemas/tests.

## D. Internal targets

| Date | Checkpoint | Accountable owner |
|---|---|---|
| Oct 13 (tonight) | Decide owners and interfaces, choose model/API, record unanswered questions | Whole team |
| Oct 14 | Frozen schemas, one mock example, dataset access verified and first usable sample | Integration + data |
| Oct 16 | Each individual module demonstrates tests with mocks; changes pushed | All module owners |
| Oct 17 | Complete paired mock integration run | Integration + A2A + state + evaluation |
| Oct 18 | First live static/dynamic paired runs; inspect logs and issues | Whole team, integration leads |
| Oct 20 | If feasible, small balanced pilot and preliminary scores; otherwise documented issues | Evaluation + data + integration |
| Oct 21 | Two slides ready, speaker rehearsed | Report coordinator + everyone |
| Oct 22, 4 PM | **Midterm presentation** | Assigned presenter(s) |
| Oct 26 | Four-page report frozen internally | Report coordinator + everyone |
| Oct 27 | **Midterm report submitted** | Submission owner (TBC) |

No one needs to work at the same time, but the shared handoffs must land by these dates. For an absent teammate, record their proposed ownership as pending confirmation until they accept the output and date.

## E. Meeting exit checklist

- [ ] Owner and backup/contact confirmed for every required component.
- [ ] Jon and Prakar have selected dataset and integration responsibilities (in whichever order they prefer).
- [ ] Aditya and Arya divided simulation loop vs. LLM reply/client work.
- [ ] Magnus confirmed exact trust update interface/target and delivery asynchronously.
- [ ] Model/API access and minimum experiment protocol agreed.
- [ ] Shared schema owner commits a mock fixture and tests by Oct 14.
- [ ] Integration test scheduled; named person owns running it.
- [ ] Someone coordinates the slides and report; submission owner confirmed later.

---

## Decision log

| Date | Decision | Agreed by |
|---|---|---|
| | | |

## Still open (from SPEC §9)

- [ ] Dataset version, label semantics, seed cut-off, handling of >2 speakers/branches
- [ ] Model/API, budget, fixed parameters, retry policy
- [ ] Which agent(s) update trust, trust meaning, first update method
- [ ] Judge scoring target, rubric, human-validation plan
- [ ] Turns/repeats, termination rule, splits
- [ ] Data vs. integration owner (Jon / Prakar), PR process, report coordinator
