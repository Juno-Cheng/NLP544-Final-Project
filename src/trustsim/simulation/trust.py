"""The trust update used in the dynamic experiment."""

from trustsim.personas.cast import Persona

NEUTRAL_RATING = 0.5  # the critic's anchor for ordinary, polite discussion
TRUST_DECIMALS = 6  # keeps float noise from pushing trust across a mood-band edge


def update_trust(trust: float, rating: float, persona: Persona, baseline: float) -> float:
    """The receiver's trust after one rated message, kept between 0 and 1:

        trust + alpha * (rating - 0.5) + forgiveness * (baseline - trust)

    alpha is the persona's alpha_down for ratings below 0.5 and alpha_up otherwise, so trust
    falls faster than it recovers. The forgiveness term pulls trust back toward the baseline,
    so one bad message fades unless it is repeated.
    """
    if not 0 <= rating <= 1:
        raise ValueError(f"rating must be between 0 and 1, got {rating}")
    alpha = persona.alpha_down if rating < NEUTRAL_RATING else persona.alpha_up
    moved = trust + alpha * (rating - NEUTRAL_RATING) + persona.forgiveness * (baseline - trust)
    return round(min(1.0, max(0.0, moved)), TRUST_DECIMALS)
