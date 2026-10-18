# Static vs. Dynamic Personas

CSCI 544 Final Project. We simulate two-agent continuations of real Wikipedia talk-page openings (Cornell CGA-WIKI) and test whether letting each agent's **trust** update during the conversation (dynamic) predicts real personal attacks better than keeping trust fixed (static).

See [`SPEC.md`](agents_docs/SPEC.md) for the full design, [`DECISIONS.md`](agents_docs/DECISIONS.md) for team decisions and open questions, and [Arya’s implementation checklist](agents_docs/Arya.md) for the October 16–17 requirements.

## Pipeline

```text
data_loader -> persona -> simulator (response_generator + state_updater)
            -> escalation_judge (blind) -> evaluate (joins real labels)
```

## Repository layout

```text
NLP544-Final-Project/
├── README.md                    # setup + the one integration command
├── agents_docs/                 # project requirements and planning
│   ├── SPEC.md                  # technical specification; provisional choices noted
│   ├── DECISIONS.md             # team decisions, open questions, implementation guide
│   └── Arya.md                  # Arya’s October 16–17 requirements and guardrails
├── requirements.txt
├── .env.example                 # API key variable names only; real .env is gitignored
├── .gitignore                   # ignores .env, data/raw, data/processed, outputs/
│
├── configs/                     # all frozen parameters live here, not in code
│   ├── data.yaml                # cutoff rule, split, sample size, speaker filtering
│   ├── personas.yaml            # stable traits + initial trust for speaker A/B
│   ├── simulation.yaml          # model_id, temperature, max turns, output cap, seeds
│   ├── updater.yaml             # static vs. dynamic policy settings
│   └── judge.yaml               # judge model, rubric_version
│
├── prompts/                     # versioned text, so prompt changes show up in diffs
│   ├── reply_v1.txt             # persona + trust-conditioned reply prompt (Arya)
│   ├── trust_update_v1.txt      # if the dynamic updater uses the LLM (Magnus)
│   └── judge_rubric_v1.txt      # blind rubric (Shadi)
│
├── src/                         # one package, one file per owner
│   ├── schemas.py               # shared types (integration owner)
│   ├── llm_client.py            # real + mock model client (Arya; mock from integration)
│   ├── data_loader.py           # CGA-WIKI → LabeledExample (Jon)
│   ├── persona.py               # initialize_personas (Magnus)
│   ├── state_updater.py         # static / dynamic update_state (Magnus)
│   ├── response_generator.py    # generate_reply (Arya)
│   ├── simulator.py             # run_simulation (Aditya)
│   ├── escalation_judge.py      # score_escalation (Shadi)
│   └── evaluate.py              # evaluate_conditions, ROC-AUC, paired comparisons (Shadi)
│
├── scripts/
│   ├── run_experiment.py        # the single end-to-end command (integration owner)
│   ├── prepare_data.py          # one-off: download corpus, export seeds + labels (Jon)
│   └── audit_labels.py          # dataset checks from SPEC §7 (Jon)
│
├── tests/
│   ├── fixtures/                # committed, small, fake or hand-picked
│   │   ├── sample_seed.json
│   │   ├── sample_labeled_examples.json
│   │   ├── sample_result_static.json
│   │   ├── sample_result_dynamic.json
│   │   └── sample_scores.json
│   ├── test_schemas.py
│   ├── test_data_loader.py
│   ├── test_state_updater.py
│   ├── test_response_generator.py
│   ├── test_simulator.py
│   ├── test_judge_blindness.py  # judge payload has no condition, trust, or label
│   ├── test_evaluate.py
│   └── test_integration_mock.py # full paired run with mock client
│
├── data/                        # gitignored except README describing the contents
│   ├── raw/                     # untouched ConvoKit download
│   └── processed/
│       ├── seeds.jsonl          # ConversationSeed only: what agents may see
│       └── labels.jsonl         # conversation_id → real_outcome, split, pair_id
│
├── outputs/                     # gitignored; one folder per experiment run
│   └── <experiment_id>/
│       ├── manifest.json        # config snapshot, git commit, prompt versions
│       ├── simulations/         # SimulationResult per run_id (static + dynamic)
│       ├── judge_inputs/        # the exact text payloads sent to the judge
│       ├── scores.jsonl         # EscalationScore records
│       ├── evaluation.json      # EvaluationSummary
│       └── logs/                # errors, retries, token usage, cost
│
├── notebooks/                   # exploration only; no pipeline logic
│   └── cga_wiki_exploration.ipynb
│
└── report/
    ├── slides/
    └── midterm_report/
```

## Owners

| Module | Owner |
|---|---|
| `simulator.py` | Aditya |
| `response_generator.py`, `llm_client.py` | Arya |
| `persona.py`, `state_updater.py` | Magnus |
| `data_loader.py`, data scripts | Jon |
| `schemas.py`, `run_experiment.py`, tests | Jon or Prakar (TBD) |
| `escalation_judge.py`, `evaluate.py` | Shadi |

## Setup

```bash
python3.13 -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API key(s); never commit .env
```

Requires Python 3.13. Data is gitignored, so everyone builds their own copy (first run downloads ~45 MB and takes a few minutes):

```bash
python scripts/prepare_data.py                    # build seeds + labels using configs/data.yaml
python scripts/prepare_data.py --opening-turns 5  # optional: override opening_turns for one run (overwrites the files)
pytest tests/test_prepare_data.py                 # should print "12 passed"
pytest tests/test_data_loader.py                  # should print "18 passed"
```

The run should end with 4,188 seeds and balanced label counts per split.

## Usage

To be filled in once the integration command exists:

```bash
python scripts/run_experiment.py   # paired static/dynamic run, judge, evaluate
pytest                             # run all tests
```

## Arya's reply generator

Activate the project-local environment with `source .venv/bin/activate`.
The current local environment includes Python 3.13 under `.venv/runtime`;
`.venv` is ignored by Git and is not portable between machines.

The generator is a callable object compatible with the `ResponseGenerator` type
from `src/schemas.py`. It also exposes the same operation as `generate_reply()`.
No API calls occur on import. A small offline example:

```python
import json
from pathlib import Path
from src.llm_client import MockLLMClient
from src.response_generator import PersonaResponseGenerator
from src.schemas import AgentState, ConversationSeed, GenerationConfig, from_dict

seed = from_dict(ConversationSeed, json.loads(
    Path("tests/fixtures/sample_seed.json").read_text(encoding="utf-8")))
config = GenerationConfig("mock", 0.3, 256, "reply_v1", 544)
events = []  # In the runner, bind this callback to the current run's metadata collector.
generator = PersonaResponseGenerator(
    MockLLMClient(["Could you explain what made it irrelevant?"]),
    on_generation=events.append,
)
reply = generator(seed.opening_turns, AgentState("A", 0.7, {"role": "editor"}), config)
print(reply)
```

For a live reply, export `OPENAI_API_KEY` and `OPENAI_MODEL` in your shell.
`.env` is not automatically loaded. With the same `seed` and imports above:

```python
import os
from openai import OpenAI
from src.llm_client import OpenAILLMClient

config = GenerationConfig(os.environ["OPENAI_MODEL"], 0.3, 256, "reply_v1", 544)
with OpenAI(timeout=30.0, max_retries=2) as sdk:
    generator = PersonaResponseGenerator(OpenAILLMClient(sdk), on_generation=events.append)
    reply = generator(seed.opening_turns, AgentState("A", 0.7, {"role": "editor"}), config)
    print(reply.text)
```

The model ID is intentionally caller-selected. The client uses the
[OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
Use a model that accepts temperature, or explicitly construct
`OpenAILLMClient(sdk, send_temperature=False)` for models that do not; use the
same choice in both arms. Unsupported settings otherwise fail visibly. No automatic
model replacement or parameter-dropping fallback occurs.

`random_seed` remains required by the shared schema, but Responses has no seed
parameter: metadata records `random_seed_requested` and `random_seed_applied=None`.
It also records requested/returned model IDs, requested/applied temperature, token
usage when available, latency, prompt version/hash, and the output turn/speaker IDs.
Cost is `None`, not an invented estimate. Bind one metadata collector per run;
the callback receives no transcript or trust values and callback failures propagate.
The experiment owner still needs to wire these events into `RunMetadata`.

The generator requires an ordered, nonempty two-speaker history and returns
`last_turn_id + 1`. It never appends the reply to the caller's history or updates
trust; the simulator does that. Turn-ID allocation and the proposed trust wording
in `reply_v1.txt` should be confirmed with Aditya and Magnus before the live pilot.
Incomplete, refused, blank, and failed responses raise `GenerationError`; the
runner must record failed runs. No raw provider error body is included in our
exception message. SDK retries are bounded by the caller's `max_retries`.

Run the offline tests (temporary test files stay inside the project):

```bash
python -m pytest tests/test_response_generator.py tests/test_llm_client.py -q --basetemp=.venv/test-tmp
python -m pytest -q --basetemp=.venv/test-tmp
```

Tests use canned responses and a fake HTTP transport with a fake API key. They
make no paid API calls. Live one-reply verification and end-to-end paired integration
remain separate acceptance steps; mock success is not evidence of research efficacy.

## Workflow

- Branch per module, open a PR, get one review before merging to `main`.
- Changes to `src/schemas.py` or `tests/fixtures/` need the integration owner's approval.
- Never commit API keys, raw data, or `outputs/`.
