"""Condition-blind dialogue judge with injected model adapter for mock testing."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _record(value):
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Expected record-like value, got {type(value).__name__}")


def build_judge_prompt(transcript, rubric):
    """Only ordered speaker IDs and text are included: no condition, trust, or label."""
    if not transcript:
        raise ValueError("Cannot judge an empty transcript")
    lines = []
    for turn in transcript:
        turn = _record(turn)
        speaker = str(turn["speaker_id"])
        text = str(turn["text"])
        lines.append(f"{speaker}: {text}")
    return f"{rubric.strip()}\n\n<TRANSCRIPT>\n" + "\n".join(lines) + "\n</TRANSCRIPT>"


def score_escalation(run_id, transcript, config):
    """Score dialogue using config['judge_callable'](prompt).

    Adapter is intentionally injected until the team finalizes llm_client.py.
    """
    config = _record(config)
    judge = config.get("judge_callable")
    if not callable(judge):
        raise ValueError("JudgeConfig must provide a callable judge_callable for now")
    model_id = config["judge_model_id"]
    rubric_version = config["rubric_version"]
    rubric_path = Path(config.get("rubric_path", ROOT / "prompts/judge_rubric_v1.txt"))
    rubric = rubric_path.read_text(encoding="utf-8")
    prompt = build_judge_prompt(transcript, rubric)
    response = judge(prompt)
    if isinstance(response, str):
        response = json.loads(response)
    if not isinstance(response, Mapping):
        raise ValueError("Judge response must be a JSON object")
    score = response.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Judge score must be numeric")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Judge score must be finite and between 0 and 1")
    return {"run_id": run_id, "score": score,
            "judge_model_id": model_id, "rubric_version": rubric_version}
