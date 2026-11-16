# Static vs. Dynamic Trust Personas

CSCI 544 Final Project (Team 17: Aditya, Arya, Jon, Magnus, Prakhar, Shadi).

Two LLM agents continue the opening of a real Wikipedia talk-page conversation from the
Conversations Gone Awry corpus (CGA-WIKI). We run two separate experiments on the same openings:

- **Static:** each agent's trust in the other stays fixed for the whole conversation.
- **Dynamic:** after every message, an LLM critic rates it for the agent receiving it, and that
  agent's trust moves up or down.

A blind judge reads each finished conversation and marks it **1** (a personal attack happened) or
**0**. Each experiment's marks are compared with the real outcome in the dataset. The agreed design
is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md), section G.

## Pipeline

```text
CGA-WIKI ──► opening lines ──┬──► Experiment 1: static  ──┐
    │                        └──► Experiment 2: dynamic ──┤
    │                                                     ▼
    │                                         blind judge: 1 or 0
    │                                                     │
    └──► real outcome, held back ───────────► compare ◄───┘ ──► metrics and summary
```

| Stage | Code | Settings |
|---|---|---|
| Dataset | `src/trustsim/data/` | `configs/data.yaml` |
| Persona A and B | `src/trustsim/personas/` | `configs/personas.yaml` |
| Experiments: agents, critic, trust rule, turn loop | `src/trustsim/simulation/` | `configs/experiment.yaml`, `prompts/agent_reply_v1.txt`, `prompts/critic_v1.txt` |
| Blind judge | `src/trustsim/judge/` | `prompts/judge_v1.txt` |
| Comparison | `src/trustsim/evaluation/` | |
| Model access (OpenAI, Gemini, offline mock) | `src/trustsim/llm/` | `configs/experiment.yaml` (`llm`) |
| The whole run | `src/trustsim/pipeline.py`, `scripts/run_experiment.py` | |

**Same in both experiments:** openings, personas, starting trust (0.75 for both agents), models,
prompts, the limit of 10 generated turns, and the judge. **Different:** only whether trust moves.

**Trust update (dynamic only).** After each message, only the receiver updates:

```text
new trust = trust + α · (rating − 0.5) + λ · (0.75 − trust)      kept within [0, 1]
```

The rating is the critic's score for the message just received: 0 hostile, 0.5 ordinary
discussion, 0.8 to 1.0 repair. Earlier turns are context only. α is the persona's α_down below 0.5
and α_up above it, so trust falls faster than it recovers. λ is the persona's forgiveness: it pulls
trust back toward 0.75, so trust can settle and recover instead of spiralling down.

**Mood.** Each turn, an agent's mood is looked up from its trust band (high, middle, low, very
low) and its persona. Personas never change during a conversation.

**Walking away.** An agent may leave; its leaving message ends the conversation. Because the
dataset's 1 means a personal attack, leaving alone is a 0. The judge reports separately whether
someone left calmly or frustrated.

## Repository layout

```text
configs/          data selection, personas A and B, experiment and model settings
prompts/          agent_reply_v1.txt, critic_v1.txt, judge_v1.txt
scripts/          prepare_data.py (once), run_experiment.py (every run)
src/trustsim/
  data/           cga.py cuts CGA openings; loader.py filters and samples attack/civil pairs
  personas/       cast.py: personas A and B, mood by trust band, who plays whom
  simulation/     conversation.py turn loop, agent.py replies, critic.py ratings, trust.py rule
  judge/          judge.py: the blind 0/1 judge
  evaluation/     metrics.py match rates and McNemar test, report.py summary
  llm/            client.py OpenAI-compatible client, mock.py offline model, factory.py
  pipeline.py     runs both experiments, judges, evaluates, and resumes stopped runs
  config.py       typed settings from configs/
  dialogue.py     Turn and Seed
  storage.py      JSON and JSONL files
tests/            offline tests and a small fixture dataset
data/             prepared data (gitignored); see data/README.md
outputs/          one folder per run (gitignored)
docs/             DECISIONS.md: the team's decision log; section G is the current design
report/slides/    midterm slides
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # also installs src/trustsim in editable mode
python scripts/prepare_data.py      # once: downloads CGA-WIKI, writes data/processed/
```

## Running

Dry run with the offline mock model. It needs no API key and costs nothing:

```bash
python scripts/run_experiment.py --run-name dry-run --mock
```

Live run with OpenAI (models are set per role in `configs/experiment.yaml`, gpt-5.5 by default):

```bash
export OPENAI_API_KEY=...
python scripts/run_experiment.py --run-name pilot-20
```

Live run with Gemini, through its OpenAI-compatible endpoint:

```bash
export GEMINI_API_KEY=...
python scripts/run_experiment.py --run-name pilot-20-gemini --provider gemini --model <gemini model id>
```

Options: `--sample-size N` (even, so attack/civil pairs stay whole), `--experiments static` or
`--experiments dynamic` to run one experiment at a time into the same run folder, `--workers N`
to simulate conversations in parallel, and `--verbose` to log every generated turn.

Running the same command again resumes the run: finished conversations are kept and failed ones
are retried. A run folder refuses to resume if the conversations, personas, models, prompts, or
turn limit have changed; use a new `--run-name` instead.

The default 20 conversations need at most 640 model calls: 400 agent replies, 200 critic ratings,
and 40 judge verdicts. The calls a run made are counted in its `run.json`.

## Outputs

Each run writes to `outputs/<run-name>/`:

| File | Contents |
|---|---|
| `summary.md` | Static vs dynamic table, paired comparison, and anything left out |
| `metrics.json` | The same numbers in full |
| `results.csv` | One row per conversation: real outcome and each experiment's verdict |
| `simulations.jsonl` | Every generated turn with the writer's mood, the critic's rating, and both agents' trust |
| `verdicts.jsonl` | The judge's verdict for every conversation and experiment |
| `run.json` | Settings, prompt hashes, git commit, status, and model calls and tokens |
| `run.log` | The run's log |

## Tests

```bash
python -m pytest
ruff check src tests scripts && ruff format --check src tests scripts
```

The tests run offline. They cover the trust rule, mood bands, the turn loop of both
experiments, the judge's blindness, retries and resuming, and the metrics.
