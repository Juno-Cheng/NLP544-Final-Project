import pytest

from trustsim.personas.cast import Persona
from trustsim.simulation.trust import update_trust

BASELINE = 0.75


def persona(alpha_up: float = 0.25, alpha_down: float = 0.50, forgiveness: float = 0.15) -> Persona:
    return Persona("A", "editor", "background", "calm", alpha_up, alpha_down, forgiveness, {})


def test_design_page_worked_example():
    rude = update_trust(0.75, 0.2, persona(), BASELINE)
    neutral = update_trust(rude, 0.5, persona(), BASELINE)
    apology = update_trust(neutral, 0.9, persona(), BASELINE)
    assert rude == pytest.approx(0.60)
    assert neutral == pytest.approx(0.6225)
    assert apology == pytest.approx(0.6225 + 0.25 * 0.4 + 0.15 * (0.75 - 0.6225))


def test_neutral_message_at_the_baseline_changes_nothing():
    assert update_trust(BASELINE, 0.5, persona(), BASELINE) == BASELINE


def test_trust_falls_faster_than_it_recovers():
    no_pull = persona(forgiveness=0)
    drop = BASELINE - update_trust(BASELINE, 0.3, no_pull, BASELINE)
    gain = update_trust(BASELINE, 0.7, no_pull, BASELINE) - BASELINE
    assert drop > gain > 0


def test_forgiveness_pulls_trust_back_toward_the_baseline_from_either_side():
    assert update_trust(0.30, 0.5, persona(), BASELINE) > 0.30
    assert update_trust(0.95, 0.5, persona(), BASELINE) < 0.95


def test_one_polite_message_after_hostility_starts_a_recovery():
    trust = BASELINE
    for _ in range(3):
        trust = update_trust(trust, 0.0, persona(), BASELINE)
    low = trust
    trust = update_trust(trust, 0.9, persona(), BASELINE)
    assert trust > low
    for _ in range(3):
        trust = update_trust(trust, 0.5, persona(), BASELINE)
    assert trust > low + 0.15


def test_trust_stays_between_zero_and_one():
    assert update_trust(0.05, 0.0, persona(alpha_down=1.0, forgiveness=0), BASELINE) == 0.0
    assert update_trust(0.95, 1.0, persona(alpha_up=1.0, forgiveness=0), BASELINE) == 1.0


@pytest.mark.parametrize("rating", [-0.1, 1.1])
def test_rating_outside_zero_to_one_is_rejected(rating):
    with pytest.raises(ValueError, match="rating"):
        update_trust(BASELINE, rating, persona(), BASELINE)
