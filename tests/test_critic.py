import json

import pytest

from trustsim.config import CRITIC
from trustsim.dialogue import Turn
from trustsim.llm.client import AnswerFormatError
from trustsim.simulation.critic import Critic, parse_rating, rating_payload

from helpers import ScriptedClient, rating_answer


def test_critic_rates_the_last_message_for_the_editor_it_was_written_to(seed):
    payload = rating_payload("bob", seed.opening)
    assert payload["reader"] == "bob"
    assert payload["conversation"][-1]["speaker"] == "alice"
    assert len(payload["conversation"]) == len(seed.opening)


def test_critic_is_never_shown_trust(seed):
    assert "trust" not in json.dumps(rating_payload("bob", seed.opening)).lower()


def test_an_editor_cannot_rate_its_own_message(seed):
    with pytest.raises(ValueError, match="cannot be its reader"):
        rating_payload("alice", seed.opening)


def test_critic_returns_the_rating(seed):
    client = ScriptedClient({CRITIC: [rating_answer(0.85)]})
    assert Critic(client, "instructions", attempts=1).rate("bob", seed.opening) == 0.85


def test_integer_ratings_are_accepted():
    assert parse_rating({"rating": 1}) == 1.0


@pytest.mark.parametrize("rating", [1.5, -0.2, True, "0.5", float("nan")])
def test_ratings_outside_zero_to_one_or_not_numbers_are_rejected(rating):
    with pytest.raises(AnswerFormatError):
        parse_rating({"rating": rating})


def test_rating_answer_with_extra_keys_is_rejected():
    with pytest.raises(AnswerFormatError):
        parse_rating({"rating": 0.5, "reason": "polite"})


def test_empty_conversation_cannot_be_rated():
    with pytest.raises(ValueError, match="needs a message"):
        rating_payload("bob", [])


def test_payload_keeps_every_earlier_turn_as_context():
    conversation = [Turn("a", "One."), Turn("b", "Two."), Turn("a", "Three.")]
    assert [turn["text"] for turn in rating_payload("b", conversation)["conversation"]] == [
        "One.",
        "Two.",
        "Three.",
    ]
