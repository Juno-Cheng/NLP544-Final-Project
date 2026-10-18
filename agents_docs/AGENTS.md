# Project agent guide

Read this guide before working on Arya's contribution. It lives in `agents_docs/`
at Arya's request; agents working in sibling directories should be explicitly
pointed here rather than relying on automatic repository-wide discovery.

## Read first

- [Arya's requirements and implementation status](Arya.md)
- [Technical specification](SPEC.md)
- [Team decisions and implementation guide](DECISIONS.md)
- [Setup and caller examples](../README.md)

Review the actual affected source, prompts, configuration, tests, and Git status
before proposing changes. Source documents contain provisional choices as well as
confirmed requirements; examples are not evidence of implementation or results.

## Approval and workspace boundaries

- Ask Arya before making changes. Describe the files, behavior, and verification
  scope; an approval covers that change set, not unrelated follow-up work.
- Keep work and changes inside `NLP544-Final-Project/`. Use a project-local `.venv`
  and keep test temporary files and tool caches within the project.
- Use the existing `arya` branch for Arya's work; inspect the current branch first.
  Preserve existing local changes. Do not switch branches, commit, push, open a PR,
  or merge changes without authorization for that action.
- Review before editing and verify after editing. Report failures, skips, and
  untested behavior honestly. Do not mark a requirement complete without evidence.
- Never print, commit, or include real API keys in examples or logs. Do not ask Arya
  to paste a key into chat. Normal tests must remain offline and free.

## Ownership and implementation conventions

Arya owns [the model client](../src/llm_client.py),
[reply generation](../src/response_generator.py), and
[the reply prompt](../prompts/reply_v1.txt).
Aditya owns simulation, Magnus persona/trust updates, Jon data preparation,
Prakhar shared schemas and experiment integration, and Shadi judging/evaluation.
Coordinate changes to another owner's modules. Shared schema and fixture changes
need the integration owner's approval under the repository workflow.

- Prefer small classes, composition, dependency injection, and explicit interfaces.
  Avoid inheritance trees, global clients, duplicated schemas, and unnecessary files.
- Import shared types from [schemas.py](../src/schemas.py).
  `PersonaResponseGenerator` is callable and also exposes
  `generate_reply(history, actor, config) -> DialogueTurn`.
- Inject `LLMClient` implementations. `MockLLMClient` provides canned replies;
  `OpenAILLMClient` wraps an injected OpenAI SDK instance.
- Keep prompts in versioned prompt files and runtime settings explicit. Do not make
  API calls on import, silently replace models, or fabricate fallback replies.
- The current generator preserves inputs and assigns `last_turn_id + 1`.
  The simulator appends turns and calls the updater; the generator does neither.
- Keep usage metadata separate from dialogue through the optional per-run callback.
  The runner is responsible for saving metadata and failed-run records.
- Use type hints, clear errors, and concise docstrings. Extend existing files when
  practical; add files only when they have a clear purpose.

## Experimental safeguards

- Generation and state updates receive visible opening/generated turns only,
  never real outcome labels or future human turns.
- Match model, prompt, starting personas, history policy, and generation settings
  across paired conditions. Only the explicit trust-update policy differs.
- Dynamic trust must reach subsequent generation. Static agents still react to
  history while their explicit trust remains fixed.
- Judge-facing input is ordered dialogue plus the fixed rubric, without condition,
  trust values, provenance, or reference labels. Join labels after scoring.
- The current Responses adapter records requested seed but does not apply it.
  Temperature omission is explicit and recorded; keep it identical across arms.
- Prompt trust semantics, initialization, and turn-budget proposals require team
  agreement. A mock or small smoke test does not establish predictive superiority.

## Environment and verification

Run commands from the repository root. Use the existing Python 3.13 environment:

```bash
source .venv/bin/activate
python --version
python -m pytest tests/test_response_generator.py tests/test_llm_client.py -q --basetemp=.venv/test-tmp
```

For broader integration checks, keep library caches inside the repository:

```bash
MPLCONFIGDIR="$PWD/.venv/matplotlib" XDG_CACHE_HOME="$PWD/.venv/cache" HF_HOME="$PWD/.venv/huggingface" HF_HUB_OFFLINE=1 python -m pytest -q --basetemp=.venv/test-tmp
python -m pip --no-cache-dir check
ruff check src/llm_client.py src/response_generator.py tests/test_llm_client.py tests/test_response_generator.py --no-cache
git diff --check
```

If `.venv` is absent, propose a Python 3.13 environment and pinned-dependency
installation for approval; do not install globally. The current local interpreter
runtime is inside `.venv/runtime`, so do not treat this environment as portable.
For documentation-only changes, check local links, paths, and whitespace; rerun
code tests only when the change warrants it.

## Handoff status

At the last implementation verification: 86 tests passed and 6 corpus-dependent
tests skipped, including 31 passing generator/client tests. Recheck relevant tests
after changes rather than treating these counts as permanent guarantees.

Remaining work: configure a locally held API key and accessible GPT model; verify
one live reply; confirm trust wording and turn-ID convention with the owners;
integrate metadata and generation into a complete paired experiment; obtain review.
See [Arya.md](Arya.md) for the detailed checklist. Do not mark live generation,
full integration, or PR submission complete based on offline tests alone.
