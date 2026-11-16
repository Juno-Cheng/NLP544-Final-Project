import csv
import json
from dataclasses import replace

import pytest

from trustsim.config import AGENT, CRITIC, JUDGE, ROLES, with_overrides
from trustsim.llm.client import LLMSetupError, UsageMeter
from trustsim.llm.factory import build_clients
from trustsim.pipeline import (
    MANIFEST,
    METRICS,
    RESULTS,
    SIMULATIONS,
    SUMMARY,
    VERDICTS,
    RunMismatchError,
    build_roles,
    run_pipeline,
)
from trustsim.storage import read_jsonl

from helpers import ScriptedClient, default_answer


def run(run_dir, examples, cast, settings, client, meter=None):
    roles, prompts = build_roles({role: client for role in ROLES}, settings)
    return run_pipeline(run_dir, examples, cast, settings, roles, prompts, meter)


def mock_run(run_dir, examples, cast, settings):
    meter = UsageMeter()
    roles, prompts = build_roles(build_clients(settings.llm, meter), settings)
    return run_pipeline(run_dir, examples, cast, settings, roles, prompts, meter)


def test_mock_run_writes_every_output(tmp_path, examples, cast, experiment_settings):
    run_dir = tmp_path / "mock-run"
    metrics = mock_run(run_dir, examples, cast, experiment_settings)
    for name in (MANIFEST, SIMULATIONS, VERDICTS, METRICS, RESULTS, SUMMARY):
        assert (run_dir / name).is_file()
    assert metrics["compared_conversations"] == len(examples) == 6
    assert set(metrics["experiments"]) == {"static", "dynamic"}
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["usage"][JUDGE]["calls"] == 2 * len(examples)
    with (run_dir / RESULTS).open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(examples)


def test_static_runs_never_call_the_critic(tmp_path, examples, cast, experiment_settings):
    client = ScriptedClient()
    run(
        tmp_path / "r",
        examples,
        cast,
        with_overrides(experiment_settings, experiments=["static"]),
        client,
    )
    assert client.calls(CRITIC) == []
    assert len(client.calls(JUDGE)) == len(examples)


def test_no_model_sees_a_label_or_which_experiment_it_is_in(
    tmp_path, examples, cast, experiment_settings
):
    client = ScriptedClient()
    run(tmp_path / "r", examples, cast, experiment_settings, client)
    allowed = {
        AGENT: {"you", "persona", "current_mood", "conversation"},
        CRITIC: {"reader", "conversation"},
        JUDGE: {"conversation"},
    }
    for request in client.requests:
        assert set(json.loads(request.user)) == allowed[request.role]
        sent = (request.system + request.user).lower()
        for word in ("static", "dynamic", "real_outcome", "conversation_has_personal_attack"):
            assert word not in sent
    for request in client.calls(JUDGE):
        speakers = {turn["speaker"] for turn in json.loads(request.user)["conversation"]}
        assert speakers == {"Editor 1", "Editor 2"}
        assert "trust" not in request.user.lower()


def test_rerunning_a_finished_run_makes_no_model_calls(
    tmp_path, examples, cast, experiment_settings
):
    run(tmp_path / "r", examples, cast, experiment_settings, ScriptedClient())
    again = ScriptedClient()
    metrics = run(tmp_path / "r", examples, cast, experiment_settings, again)
    assert again.requests == []
    assert metrics["compared_conversations"] == len(examples)


def test_failed_judging_is_kept_and_retried_on_the_next_run(
    tmp_path, examples, cast, experiment_settings
):
    run_dir = tmp_path / "r"
    broken_judge = ScriptedClient(
        fallback=lambda request: "not json" if request.role == JUDGE else default_answer(request)
    )
    metrics = run(run_dir, examples, cast, experiment_settings, broken_judge)
    reasons = [reason for item in metrics["excluded"] for reason in item["reasons"].values()]
    assert metrics["compared_conversations"] == 0
    assert len(reasons) == 2 * len(examples)
    assert all(reason.startswith("judge failed") for reason in reasons)
    assert json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))["status"] == "incomplete"

    retry = ScriptedClient()
    metrics = run(run_dir, examples, cast, experiment_settings, retry)
    assert metrics["compared_conversations"] == len(examples)
    assert {request.role for request in retry.requests} == {JUDGE}


def test_failed_simulation_is_saved_with_its_error(tmp_path, examples, cast, experiment_settings):
    broken_agent = ScriptedClient(
        fallback=lambda request: "{}" if request.role == AGENT else default_answer(request)
    )
    run(tmp_path / "r", examples, cast, experiment_settings, broken_agent)
    records = read_jsonl(tmp_path / "r" / SIMULATIONS)
    assert len(records) == 2 * len(examples)
    assert all(record["status"] == "error" for record in records)
    assert all("UnusableAnswerError" in record["error"] for record in records)


def test_changed_settings_refuse_to_resume(tmp_path, examples, cast, experiment_settings):
    run(tmp_path / "r", examples, cast, experiment_settings, ScriptedClient())
    with pytest.raises(RunMismatchError, match="new --run-name"):
        run(
            tmp_path / "r",
            examples,
            cast,
            replace(experiment_settings, max_turns=3),
            ScriptedClient(),
        )


def test_experiments_can_run_as_separate_invocations(tmp_path, examples, cast, experiment_settings):
    run_dir = tmp_path / "r"
    static_only = with_overrides(experiment_settings, experiments=["static"])
    first = run(run_dir, examples, cast, static_only, ScriptedClient())
    assert set(first["experiments"]) == {"static"}
    assert "paired" not in first
    dynamic_only = with_overrides(experiment_settings, experiments=["dynamic"])
    second = run(run_dir, examples, cast, dynamic_only, ScriptedClient())
    assert set(second["experiments"]) == {"static", "dynamic"}
    assert second["paired"]["both_match"] + second["paired"]["neither_matches"] <= len(examples)


def test_parallel_workers_give_the_same_results_as_one(
    tmp_path, examples, cast, experiment_settings
):
    serial = mock_run(tmp_path / "serial", examples, cast, experiment_settings)
    parallel_settings = with_overrides(experiment_settings, workers=4)
    parallel = mock_run(tmp_path / "parallel", examples, cast, parallel_settings)
    assert parallel == serial
    assert len(read_jsonl(tmp_path / "parallel" / VERDICTS)) == 2 * len(examples)


def test_setup_error_stops_the_run(tmp_path, examples, cast, experiment_settings):
    class RejectedKeyClient:
        def complete(self, request):
            raise LLMSetupError("the API key was rejected")

    with pytest.raises(LLMSetupError):
        run(tmp_path / "r", examples, cast, experiment_settings, RejectedKeyClient())
    manifest = json.loads((tmp_path / "r" / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["status"] == "stopped"
    assert not (tmp_path / "r" / SIMULATIONS).exists()
