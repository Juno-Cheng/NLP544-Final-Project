"""Offline checks for the paired experiment command."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from run_experiment import generated_turns_for, load_examples, load_simulation_settings, main


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_mock_command_saves_matched_arms_and_hides_the_label(tmp_path):
    code = main(
        [
            "--mode",
            "mock",
            "--limit",
            "1",
            "--source",
            "fixture",
            "--experiment-id",
            "mock-one",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    run_dir = tmp_path / "mock-one"
    manifest = _load(run_dir / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["mode"] == "mock"
    assert manifest["source"] == "fixture"
    assert manifest["n_runs"] == 2
    assert manifest["n_scores"] == 2

    static = _load(run_dir / "simulations" / "example_001-static.json")
    dynamic = _load(run_dir / "simulations" / "example_001-dynamic.json")
    assert static["status"] == "ok"
    assert dynamic["status"] == "ok"
    assert static["condition"] == "static"
    assert dynamic["condition"] == "dynamic"
    assert static["state_history"][0] == dynamic["state_history"][0]
    for arm in (static, dynamic):
        assert len(arm["generated_turns"]) == arm["run_metadata"]["max_generated_turns"]
        assert "real_outcome" not in json.dumps(arm)
        initial = arm["state_history"][0]["speaker_states"]
        for snapshot in arm["state_history"]:
            assert snapshot["speaker_states"] == initial
    for field in (
        "model_id",
        "prompt_version",
        "temperature",
        "max_generated_turns",
        "max_output_tokens",
        "random_seed",
        "updater_version",
    ):
        assert static["run_metadata"][field] == dynamic["run_metadata"][field]

    for condition in ("static", "dynamic"):
        prompt = (run_dir / "judge_inputs" / f"example_001-{condition}.txt").read_text(encoding="utf-8")
        dialogue = json.loads(prompt.split("<TRANSCRIPT_JSON>\n")[1].split("\n</TRANSCRIPT_JSON>")[0])
        assert all(set(turn) == {"speaker_id", "text"} for turn in dialogue)
        assert "real_outcome" not in prompt
        assert "static" not in prompt
        assert "dynamic" not in prompt

    scores = [
        json.loads(line)
        for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["run_id"] for row in scores} == {
        "example_001-static",
        "example_001-dynamic",
    }
    summary = _load(run_dir / "evaluation.json")
    assert summary["coverage"]["n_completed_pairs"] == 1
    assert summary["coverage"]["n_excluded"] == 0
    assert summary["unverified_pair_settings"] == []
    assert _load(run_dir / "evaluation_report.json")["pairs"][0]["real_outcome"] == 1


@pytest.mark.parametrize(("real_turn_count", "expected"), [(2, 1), (3, 1), (6, 4), (18, 16)])
def test_match_real_generates_the_rest_of_the_real_conversation(real_turn_count, expected):
    settings = replace(load_simulation_settings(), turn_budget="match_real", min_generated_turns=1)
    example = replace(load_examples("fixture")[0], real_turn_count=real_turn_count)
    assert len(example.seed.opening_turns) == 2
    assert generated_turns_for(example, settings) == expected


def test_fixed_budget_ignores_real_length():
    settings = replace(load_simulation_settings(), turn_budget="fixed", max_generated_turns=5)
    example = replace(load_examples("fixture")[0], real_turn_count=18)
    assert generated_turns_for(example, settings) == 5


def test_match_real_without_real_length_is_an_error():
    settings = replace(load_simulation_settings(), turn_budget="match_real")
    example = replace(load_examples("fixture")[0], real_turn_count=None)
    with pytest.raises(ValueError, match="real_turn_count"):
        generated_turns_for(example, settings)


def test_match_pair_uses_the_longer_real_length_in_the_pair():
    settings = replace(load_simulation_settings(), turn_budget="match_pair", min_generated_turns=1)
    attack = replace(load_examples("fixture")[0], real_turn_count=9)
    civil = replace(
        attack,
        seed=replace(attack.seed, conversation_id="example_001_civil"),
        real_outcome=0,
        pair_id="example_001",
        real_turn_count=5,
    )
    counts = {
        attack.seed.conversation_id: attack.real_turn_count,
        civil.seed.conversation_id: civil.real_turn_count,
    }
    # Openings have 2 turns; pair length is 9, so both generate 7.
    assert generated_turns_for(attack, settings, counts) == 7
    assert generated_turns_for(civil, settings, counts) == 7


def test_match_pair_without_loaded_partner_uses_own_length():
    settings = replace(load_simulation_settings(), turn_budget="match_pair", min_generated_turns=1)
    example = replace(load_examples("fixture")[0], real_turn_count=6)
    assert generated_turns_for(example, settings, {example.seed.conversation_id: 6}) == 4


def test_both_arms_generate_the_pair_length(tmp_path, monkeypatch):
    fixture = load_examples("fixture")[0]
    attack = replace(fixture, real_turn_count=9)
    civil_seed = replace(fixture.seed, conversation_id="example_001_civil")
    civil = replace(fixture, seed=civil_seed, real_outcome=0, pair_id="example_001", real_turn_count=5)
    monkeypatch.setattr("run_experiment.load_examples", lambda _source: [attack, civil])
    assert main(
        [
            "--mode",
            "mock",
            "--limit",
            "2",
            "--source",
            "fixture",
            "--experiment-id",
            "pair-len",
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    for cid in ("example_001", "example_001_civil"):
        for condition in ("static", "dynamic"):
            arm = _load(tmp_path / "pair-len" / "simulations" / f"{cid}-{condition}.json")
            assert len(arm["transcript"]) == 9
            assert len(arm["generated_turns"]) == arm["run_metadata"]["max_generated_turns"] == 7
    manifest = _load(tmp_path / "pair-len" / "manifest.json")
    assert manifest["simulation"]["turn_budget"] == "match_pair"


def _mock_args(tmp_path, experiment_id):
    return ["--mode", "mock", "--limit", "1", "--source", "fixture",
            "--experiment-id", experiment_id, "--output-root", str(tmp_path)]


def test_both_arms_start_from_the_replayed_opening(tmp_path, monkeypatch):
    monkeypatch.setattr("run_experiment.MOCK_RATINGS", ("0",))
    assert main(_mock_args(tmp_path, "replay")) == 0
    run_dir = tmp_path / "replay"
    static = _load(run_dir / "simulations" / "example_001-static.json")
    dynamic = _load(run_dir / "simulations" / "example_001-dynamic.json")
    start = static["state_history"][0]["speaker_states"]
    assert dynamic["state_history"][0]["speaker_states"] == start
    assert all(state["trust"] < 0.75 for state in start.values())
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text().splitlines()]
    replay = [e for e in events if e["stage"] == "opening_replay"]
    assert replay[0]["status"] == "ok"
    assert replay[0]["initial_trust"] == {sid: state["trust"] for sid, state in start.items()}
    calls = [json.loads(line) for line in (run_dir / "logs" / "model_calls.jsonl").read_text().splitlines()]
    assert any(call["stage"] == "opening_rating" for call in calls)


def test_failed_opening_replay_is_logged_for_both_arms(tmp_path, monkeypatch):
    monkeypatch.setattr("run_experiment.MOCK_RATINGS", ("not a number",))
    assert main(_mock_args(tmp_path, "bad-replay")) == 0
    run_dir = tmp_path / "bad-replay"
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["stage"] == "opening_replay"]
    assert {e["condition"] for e in failed} == {"static", "dynamic"}
    assert all(e["status"] == "failed" and "non-numeric" in e["error"] for e in failed)
    assert _load(run_dir / "evaluation.json")["coverage"]["n_excluded"] == 1


def test_missing_processed_data_is_reported(tmp_path, monkeypatch):
    def missing():
        raise FileNotFoundError("data/processed/seeds.jsonl")

    monkeypatch.setattr("run_experiment.load_and_prepare", missing)
    code = main(
        [
            "--mode",
            "mock",
            "--limit",
            "1",
            "--source",
            "processed",
            "--experiment-id",
            "no-data",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert code == 1
    manifest = _load(tmp_path / "no-data" / "manifest.json")
    assert manifest["status"] == "failed"


def test_existing_experiment_directory_is_kept(tmp_path):
    args = [
        "--mode",
        "mock",
        "--limit",
        "1",
        "--source",
        "fixture",
        "--experiment-id",
        "mock-one",
        "--output-root",
        str(tmp_path),
    ]
    assert main(args) == 0
    assert main(args) == 1
    assert _load(tmp_path / "mock-one" / "manifest.json")["status"] == "completed"
