import json

import pytest

from trustsim.config import AGENT
from trustsim.llm.client import AnswerFormatError
from trustsim.simulation.agent import Reply, ReplyWriter, parse_reply, reply_payload

from helpers import ScriptedClient, reply_answer


def test_agent_sees_its_persona_mood_and_the_conversation_only(cast, seed):
    payload = reply_payload("bob", cast.personas["B"], "Irritated.", seed.opening)
    assert set(payload) == {"you", "persona", "current_mood", "conversation"}
    assert payload["you"] == "bob"
    assert payload["current_mood"] == "Irritated."
    assert payload["conversation"][-1] == {"speaker": "alice", "text": seed.opening[-1].text}
    text = json.dumps(payload).lower()
    for hidden in ("trust", "static", "dynamic", "alpha", "forgiveness", "real_outcome"):
        assert hidden not in text


def test_writer_returns_the_parsed_reply(cast, seed):
    client = ScriptedClient({AGENT: [reply_answer("  I disagree, see WP:RS.  ", leaving=False)]})
    writer = ReplyWriter(client, "instructions", attempts=1)
    reply = writer.write("bob", cast.personas["B"], "Calm.", seed.opening)
    assert reply == Reply("I disagree, see WP:RS.", False)
    assert client.requests[0].system == "instructions"


def test_leaving_reply_is_kept():
    assert parse_reply({"message": "I'm done here.", "leaving": True}).leaving is True


@pytest.mark.parametrize(
    "answer",
    [
        {"message": "   ", "leaving": False},
        {"message": "Fine.", "leaving": "no"},
        {"message": "Fine."},
        {"message": "Fine.", "leaving": False, "mood": "calm"},
    ],
)
def test_malformed_replies_are_rejected(answer):
    with pytest.raises(AnswerFormatError):
        parse_reply(answer)
