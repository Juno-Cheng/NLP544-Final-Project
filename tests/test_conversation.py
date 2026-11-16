import json

import pytest

from trustsim.config import DYNAMIC, EXPERIMENTS, STATIC
from trustsim.simulation.agent import Reply
from trustsim.simulation.conversation import TURN_LIMIT, WALK_AWAY, Simulation, simulate
from trustsim.simulation.trust import update_trust

from helpers import FakeCritic, FakeWriter, ForbiddenCritic


def test_static_never_calls_the_critic_and_trust_never_moves(cast, seed):
    simulation = simulate(seed, STATIC, cast, FakeWriter(), ForbiddenCritic(), max_turns=10)
    start = cast.starting_trust
    assert len(simulation.turns) == 10
    assert all(turn.rating is None for turn in simulation.turns)
    assert all(turn.trust == {"A": start, "B": start} for turn in simulation.turns)


def test_static_agents_keep_the_mood_of_the_starting_trust(cast, seed):
    writer = FakeWriter()
    simulate(seed, STATIC, cast, writer, None, max_turns=6)
    for call in writer.calls:
        assert call["mood"] == cast.mood(call["persona_id"], cast.starting_trust)[1]


def test_agents_alternate_starting_with_whoever_did_not_speak_last(cast, seed):
    writer = FakeWriter()
    simulate(seed, STATIC, cast, writer, None, max_turns=4)
    assert [call["speaker_id"] for call in writer.calls] == ["bob", "alice", "bob", "alice"]
    assert [call["persona_id"] for call in writer.calls] == ["B", "A", "B", "A"]


def test_each_agent_sees_the_whole_conversation_so_far(cast, seed):
    writer = FakeWriter()
    simulate(seed, STATIC, cast, writer, None, max_turns=3)
    assert writer.calls[0]["conversation"] == list(seed.opening)
    assert [len(call["conversation"]) for call in writer.calls] == [3, 4, 5]


def test_dynamic_rates_every_message_for_the_agent_receiving_it(cast, seed):
    critic = FakeCritic()
    simulate(seed, DYNAMIC, cast, FakeWriter(), critic, max_turns=4)
    assert critic.calls == [("alice", "bob"), ("bob", "alice"), ("alice", "bob"), ("bob", "alice")]


def test_dynamic_changes_only_the_receivers_trust(cast, seed):
    simulation = simulate(seed, DYNAMIC, cast, FakeWriter(), FakeCritic([0.0]), max_turns=1)
    expected = update_trust(cast.starting_trust, 0.0, cast.personas["A"], cast.starting_trust)
    assert simulation.turns[0].rating == 0.0
    assert simulation.turns[0].trust == {"A": expected, "B": cast.starting_trust}


def test_mood_follows_trust_down_after_hostile_messages(cast, seed):
    writer = FakeWriter()
    simulate(seed, DYNAMIC, cast, writer, FakeCritic(then=0.0), max_turns=8)
    alice_moods = [call["mood"] for call in writer.calls if call["speaker_id"] == "alice"]
    trust, expected = cast.starting_trust, []
    for _ in alice_moods:
        trust = update_trust(trust, 0.0, cast.personas["A"], cast.starting_trust)
        expected.append(cast.mood("A", trust)[1])
    assert alice_moods == expected
    assert len(set(alice_moods)) > 1


def test_trust_recovers_once_the_hostility_stops(cast, seed):
    critic = FakeCritic([0.0, 0.5, 0.0, 0.5], then=0.9)
    simulation = simulate(seed, DYNAMIC, cast, FakeWriter(), critic, max_turns=10)
    trust_a = [turn.trust["A"] for turn in simulation.turns]
    assert trust_a[-1] > min(trust_a)


def test_conversation_stops_at_the_turn_limit(cast, seed):
    simulation = simulate(seed, STATIC, cast, FakeWriter(), None, max_turns=10)
    assert len(simulation.turns) == 10
    assert simulation.ended_by == TURN_LIMIT


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_walking_away_ends_the_conversation_in_both_experiments(cast, seed, experiment):
    writer = FakeWriter([Reply("Fine.", False), Reply("I'm done here, do what you want.", True)])
    simulation = simulate(seed, experiment, cast, writer, FakeCritic(), max_turns=10)
    assert len(simulation.turns) == 2
    assert simulation.ended_by == WALK_AWAY
    assert simulation.transcript()[-1].text == "I'm done here, do what you want."


def test_both_experiments_start_from_the_same_opening_cast_and_trust(cast, seed):
    static = simulate(seed, STATIC, cast, FakeWriter(), None, max_turns=2)
    dynamic = simulate(seed, DYNAMIC, cast, FakeWriter(), FakeCritic(), max_turns=2)
    assert static.opening == dynamic.opening == seed.opening
    assert static.personas == dynamic.personas
    assert static.starting_trust == dynamic.starting_trust == cast.starting_trust


def test_transcript_is_the_opening_then_the_generated_turns(cast, seed):
    transcript = simulate(seed, STATIC, cast, FakeWriter(), None, max_turns=2).transcript()
    assert transcript[: len(seed.opening)] == list(seed.opening)
    assert [turn.text for turn in transcript[len(seed.opening) :]] == [
        "Comment number 1.",
        "Comment number 2.",
    ]


def test_lowest_trust_is_the_minimum_either_agent_reached(cast, seed):
    critic = FakeCritic([0.0, 0.0], then=1.0)
    dynamic = simulate(seed, DYNAMIC, cast, FakeWriter(), critic, max_turns=6)
    reached = [value for turn in dynamic.turns for value in turn.trust.values()]
    assert dynamic.lowest_trust() == min(reached) < cast.starting_trust
    static = simulate(seed, STATIC, cast, FakeWriter(), None, max_turns=6)
    assert static.lowest_trust() == cast.starting_trust


def test_simulation_survives_saving_and_loading(cast, seed):
    simulation = simulate(seed, DYNAMIC, cast, FakeWriter(), FakeCritic([0.2, 0.9]), max_turns=4)
    restored = Simulation.from_record(json.loads(json.dumps(simulation.to_record())))
    assert restored == simulation


def test_dynamic_without_a_critic_is_rejected(cast, seed):
    with pytest.raises(ValueError, match="needs a critic"):
        simulate(seed, DYNAMIC, cast, FakeWriter(), None, max_turns=2)


def test_unknown_experiment_is_rejected(cast, seed):
    with pytest.raises(ValueError, match="unknown experiment"):
        simulate(seed, "frozen", cast, FakeWriter(), None, max_turns=2)
