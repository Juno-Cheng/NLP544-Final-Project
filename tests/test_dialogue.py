from trustsim.dialogue import Seed, Turn, turns_payload


def test_speakers_are_listed_once_in_order_of_first_appearance():
    seed = Seed("c", (Turn("b", "1"), Turn("a", "2"), Turn("b", "3")))
    assert seed.speakers == ("b", "a")


def test_turns_payload_keeps_order_and_text():
    turns = [Turn("a", "Hello."), Turn("b", "Hi.")]
    assert turns_payload(turns) == [
        {"speaker": "a", "text": "Hello."},
        {"speaker": "b", "text": "Hi."},
    ]
