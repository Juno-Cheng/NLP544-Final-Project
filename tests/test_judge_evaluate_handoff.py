"""Judge-to-evaluator integration with a fake LLM client."""

from src.escalation_judge import make_llm_judge_callable, score_escalation
from src.evaluate import evaluate_conditions, to_schema_summary
from src.llm_client import MockLLMClient
from src.schemas import DialogueTurn, GenerationConfig, JudgeConfig


def test_mocked_judge_to_evaluation_is_blind():
    examples = [
        {"seed": {"conversation_id": "attack"}, "real_outcome": 1},
        {"seed": {"conversation_id": "civil"}, "real_outcome": 0},
    ]
    runs = [
        {
            "run_id": f"{cid}-{arm}",
            "conversation_id": cid,
            "condition": arm,
            "status": "completed",
        }
        for cid in ("attack", "civil")
        for arm in ("static", "dynamic")
    ]

    client = MockLLMClient(
        [
            '{"score": 0.2}',
            '{"score": 0.9}',
            '{"score": 0.8}',
            '{"score": 0.1}',
        ]
    )
    judge = make_llm_judge_callable(
        client,
        GenerationConfig("mock-judge", 0.0, 128, "judge_rubric_v1", 544),
    )

    scores = [
        score_escalation(
            run["run_id"],
            [DialogueTurn(0, "A", "I disagree with that edit.")],
            JudgeConfig("mock-judge", "v1"),
            judge_callable=judge,
        )
        for run in runs
    ]

    assert len(client.requests) == 4
    assert all(
        '"real_outcome"' not in p and '"condition"' not in p and '"trust"' not in p
        for p in (request["payload"] for request in client.requests)
    )

    report = evaluate_conditions(examples, runs, scores)
    summary = to_schema_summary(report)

    assert (
        summary.coverage.n_completed_pairs,
        summary.metrics.roc_auc_static,
        summary.metrics.roc_auc_dynamic,
    ) == (2, 0.0, 1.0)
