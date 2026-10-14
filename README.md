# Static vs. Dynamic Personas

CSCI 544 Final Project. We simulate two-agent continuations of real Wikipedia talk-page openings (Cornell CGA-WIKI) and test whether letting each agent's **trust** update during the conversation (dynamic) predicts real personal attacks better than keeping trust fixed (static).

See [`SPEC.md`](SPEC.md) for the full design and [`DECISIONS.md`](DECISIONS.md) for team decisions and open questions.

## Pipeline

```text
data_loader -> persona -> simulator (response_generator + state_updater)
            -> escalation_judge (blind) -> evaluate (joins real labels)
```

## Repository layout

```text
NLP544-Final-Project/
├── README.md                    # setup + the one integration command
├── SPEC.md                      # the frozen spec (v0.2 → v1.0 once approved)
├── DECISIONS.md                 # running log of team decisions + open questions
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
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API key(s); never commit .env
```

Requires Python 3.13. Data is gitignored, so everyone builds their own copy (first run downloads ~45 MB and takes a few minutes):

```bash
python scripts/prepare_data.py                    # build seeds + labels (default: 3 opening turns)
python scripts/prepare_data.py --opening-turns 5  # optional: longer openings (overwrites the files)
pytest tests/test_prepare_data.py                 # should print "11 passed"
```

The run should end with 4,188 seeds and balanced label counts per split.

## Usage

To be filled in once the integration command exists:

```bash
python scripts/run_experiment.py   # paired static/dynamic run, judge, evaluate
pytest                             # run all tests
```

## Workflow

- Branch per module, open a PR, get one review before merging to `main`.
- Changes to `src/schemas.py` or `tests/fixtures/` need the integration owner's approval.
- Never commit API keys, raw data, or `outputs/`.
