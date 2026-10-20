"""Offline checks for the paired experiment command."""

import json
from pathlib import Path

from run_experiment import main


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
