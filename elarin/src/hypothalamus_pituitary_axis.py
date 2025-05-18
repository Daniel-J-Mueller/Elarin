"""Simple neuro-modulator state machine."""


class HypothalamusPituitaryAxis:
    """Tracks four hormone levels and updates them."""

    def __init__(self):
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0

    def step(self, novelty: float, error: float) -> None:
        self.dopamine = 0.9 * self.dopamine + novelty
        self.norepinephrine = 0.9 * self.norepinephrine + error
        self.serotonin *= 0.99
        self.acetylcholine = 0.9 * self.acetylcholine + abs(novelty - error)
