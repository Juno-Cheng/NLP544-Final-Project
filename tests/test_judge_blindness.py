import json
import pytest
from src.escalation_judge import score_escalation


def test_judge_is_blind():
    received = []
    transcript = [{"turn_id": 0, "speaker_id": "A", "text": "I disagree.",
                   "condition": "dynamic", "real_outcome": 1, "trust": 0.1}]
    config = {"judge_model_id": "mock-v1", "rubric_version": "v1",
              "judge_callable": lambda prompt: received.append(prompt) or '{"score": 0.2}'}
    result = score_escalation("run-1", transcript, config)
    assert result == {"run_id": "run-1", "score": 0.2,
                      "judge_model_id": "mock-v1", "rubric_version": "v1"}
    prompt = received[0]
    for forbidden in ('"condition"', '"real_outcome"', '"trust"', 'dynamic', '0.1'):
        assert forbidden not in prompt
    assert "A: I disagree." in prompt


def test_rejects_out_of_range_score():
    cfg = {"judge_model_id": "mock", "rubric_version": "v1",
           "judge_callable": lambda _: '{"score": 2.0}'}
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_escalation("x", [{"speaker_id": "A", "text": "hi"}], cfg)


def test_empty_transcript_rejected():
    cfg = {"judge_model_id": "mock", "rubric_version": "v1",
           "judge_callable": lambda _: '{"score": 0.0}'}
    with pytest.raises(ValueError, match="empty"):
        score_escalation("x", [], cfg)
