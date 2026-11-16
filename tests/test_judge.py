import re
from pathlib import Path

import pytest

from trustsim.config import DYNAMIC, JUDGE
from trustsim.dialogue import Turn
from trustsim.judge.judge import Judge, Verdict, judge_payload, parse_verdict
from trustsim.llm.client import AnswerFormatError
from trustsim.simulation.conversation import simulate

from helpers import FakeCritic, FakeWriter, ScriptedClient, verdict_answer

ROOT = Path(__file__).resolve().parents[1]


def test_judge_sees_only_the_conversation_with_editors_renamed():
    transcript = [Turn("alice", "One."), Turn("bob", "Two."), Turn("alice", "Three.")]
    assert judge_payload(transcript) == {
        "conversation": [
            {"speaker": "Editor 1", "text": "One."},
            {"speaker": "Editor 2", "text": "Two."},
            {"speaker": "Editor 1", "text": "Three."},
        ]
    }


def test_judge_reads_the_opening_and_every_generated_turn(cast, seed):
    simulation = simulate(seed, DYNAMIC, cast, FakeWriter(), FakeCritic(), max_turns=4)
    texts = [turn["text"] for turn in judge_payload(simulation.transcript())["conversation"]]
    assert texts[: len(seed.opening)] == [turn.text for turn in seed.opening]
    assert len(texts) == len(seed.opening) + 4


def test_judge_request_holds_no_experiment_trust_persona_or_label(cast, seed):
    simulation = simulate(seed, DYNAMIC, cast, FakeWriter(), FakeCritic([0.0, 0.0]), max_turns=4)
    client = ScriptedClient({JUDGE: [verdict_answer()]})
    Judge(client, "instructions", attempts=1).judge(simulation.transcript())
    sent = client.requests[0].user.lower()
    hidden = ["static", "dynamic", "trust", "persona", "real_outcome", "alice", "bob"]
    hidden += [persona.temperament.lower() for persona in cast.personas.values()]
    for word in hidden:
        assert word not in sent


def test_attack_true_becomes_one_like_the_dataset_label():
    verdict = parse_verdict({"personal_attack": True, "walk_away": "none", "evidence": " idiot "})
    assert verdict == Verdict(1, "none", "idiot")


def test_judge_returns_the_parsed_verdict():
    client = ScriptedClient({JUDGE: [verdict_answer(False, "frustrated")]})
    verdict = Judge(client, "instructions", attempts=1).judge([Turn("a", "I'm done here.")])
    assert verdict == Verdict(0, "frustrated", "")


@pytest.mark.parametrize(
    "answer",
    [
        {"personal_attack": 1, "walk_away": "none", "evidence": ""},
        {"personal_attack": False, "walk_away": "angry", "evidence": ""},
        {"personal_attack": False, "walk_away": "none", "evidence": None},
        {"personal_attack": False, "walk_away": "none"},
    ],
)
def test_malformed_verdicts_are_rejected(answer):
    with pytest.raises(AnswerFormatError):
        parse_verdict(answer)


def test_empty_conversation_cannot_be_judged():
    with pytest.raises(ValueError, match="needs a conversation"):
        judge_payload([])


def test_judge_prompt_names_no_experiment_or_trust():
    prompt = (ROOT / "prompts" / "judge_v1.txt").read_text(encoding="utf-8").lower()
    for word in ("static", "dynamic", "trust", "persona", "personas", "simulation", "simulated"):
        assert not re.search(rf"\b{word}\b", prompt)
