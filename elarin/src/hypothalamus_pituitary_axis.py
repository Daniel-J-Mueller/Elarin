"""Simple neuro-modulator state machine."""


import torch


class HypothalamusPituitaryAxis:
    """Tracks four hormone levels and provides habituation filtering."""

    def __init__(
        self,
        habituation_decay: float = 0.95,
        habituation_recovery: float = 0.02,
    ) -> None:
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0
        # Habituation state to gradually suppress repetitive intero signals
        self.habituation = 1.0
        self.hab_decay = habituation_decay
        self.hab_recovery = habituation_recovery
        self.prev_intero: torch.Tensor | None = None

    def step(self, novelty: float, error: float) -> None:
        self.dopamine = 0.9 * self.dopamine + novelty
        self.norepinephrine = 0.9 * self.norepinephrine + error
        self.serotonin *= 0.99
        self.acetylcholine = 0.9 * self.acetylcholine + abs(novelty - error)

    def filter_intero(self, emb: torch.Tensor) -> torch.Tensor:
        """Attenuate repetitive motor embeddings.

        A cosine similarity check against the previous intero embedding
        determines whether ``self.habituation`` should decay or recover. The
        resulting factor scales the output so persistent loops gradually fade.
        """

        if self.prev_intero is not None:
            prev = self.prev_intero.to(emb.device)
            sim = torch.nn.functional.cosine_similarity(
                emb.view(-1), prev.view(-1), dim=0
            ).item()
            if sim > 0.95:
                self.habituation *= self.hab_decay
            else:
                self.habituation += (1.0 - self.habituation) * self.hab_recovery

        self.habituation = max(0.0, min(1.0, self.habituation))
        self.prev_intero = emb.detach().cpu()
        return emb * self.habituation
