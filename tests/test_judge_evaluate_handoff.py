"""Judge-to-evaluator integration with a fake LLM client."""

from types import SimpleNamespace

from src.escalation_judge import score_escalation, make_llm_judge_callable
from src.evaluate import evaluate_conditions, to_schema_summary
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

    class FakeClient:
        def __init__(self):
            self.prompts = []
            self.replies = iter([
                '{"score": 0.2}', '{"score": 0.9}',
                '{"score": 0.8}', '{"score": 0.1}',
            ])

        def complete(self, instructions, payload, config):
            self.prompts.append(payload)
            return SimpleNamespace(text=next(self.replies))

    client = FakeClient()
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

    assert len(client.prompts) == 4
    assert all(
        '"real_outcome"' not in p
        and '"condition"' not in p
        and '"trust"' not in p
        for p in client.prompts
    )

    report = evaluate_conditions(examples, runs, scores)
    summary = to_schema_summary(report)

    assert (
        summary.n_paired_complete,
        summary.auc_static,
        summary.auc_dynamic,
    ) == (2, 0.0, 1.0)
