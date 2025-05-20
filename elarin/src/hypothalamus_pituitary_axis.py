"""Simple neuro-modulator state machine."""

import logging
import torch


class HypothalamusPituitaryAxis:
    """Tracks four hormone levels and provides habituation filtering.

    A small moving-average baseline of novelty and prediction error is
    maintained so hormone adjustments respond to trends rather than just the
    current value.  This helps stabilise the system when novelty or error
    fluctuates for long periods and mirrors how the limbic system modulates
    behaviour based on context【F:human_brain_components_reference.txt†L108-L113】.
    """

    def __init__(
        self,
        habituation_decay: float = 0.9,
        habituation_recovery: float = 0.02,
        habituation_threshold: float = 0.92,
        trend_rate: float = 0.05,
    ) -> None:
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0
        # Habituation state to gradually suppress repetitive intero signals
        self.habituation = 1.0
        self.hab_decay = habituation_decay
        self.hab_recovery = habituation_recovery
        self.hab_threshold = habituation_threshold
        self.repeat_count = 0
        self.prev_intero: torch.Tensor | None = None

        self.trend_rate = trend_rate
        self.novelty_avg = 0.0
        self.novelty_var = 1e-6
        self.error_avg = 0.0
        self.error_var = 1e-6
        self.baseline_avg = 0.0

    def step(self, novelty: float, error: float) -> None:
        """Update hormones using trend-adjusted novelty and error."""

        # Exponential moving averages capture long term trends
        self.novelty_avg = (
            1 - self.trend_rate
        ) * self.novelty_avg + self.trend_rate * novelty
        self.error_avg = (
            1 - self.trend_rate
        ) * self.error_avg + self.trend_rate * error
        # Track variance to adapt update scale over time
        self.novelty_var = (
            1 - self.trend_rate
        ) * self.novelty_var + self.trend_rate * (
            novelty - self.novelty_avg
        ) ** 2
        self.error_var = (
            1 - self.trend_rate
        ) * self.error_var + self.trend_rate * (error - self.error_avg) ** 2

        novelty_delta = (novelty - self.novelty_avg) / (
            self.novelty_var**0.5 + 1e-6
        )
        error_delta = (error - self.error_avg) / (self.error_var**0.5 + 1e-6)

        self.dopamine = 0.9 * self.dopamine + novelty_delta
        self.norepinephrine = 0.9 * self.norepinephrine + error_delta
        self.serotonin = 0.995 * self.serotonin - 0.05 * novelty_delta
        self.acetylcholine = 0.9 * self.acetylcholine + abs(
            novelty_delta - error_delta
        )

        self.dopamine = max(0.0, min(1.0, self.dopamine))
        self.norepinephrine = max(0.0, min(1.0, self.norepinephrine))
        self.serotonin = max(0.0, min(1.0, self.serotonin))
        self.acetylcholine = max(0.0, min(1.0, self.acetylcholine))

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
            if sim > self.hab_threshold:
                self.repeat_count += 1
                self.habituation *= self.hab_decay**self.repeat_count
            else:
                self.repeat_count = 0
                self.habituation += (
                    1.0 - self.habituation
                ) * self.hab_recovery

        self.habituation = max(0.0, min(1.0, self.habituation))
        self.prev_intero = emb.detach().cpu()
        return emb * self.habituation

    def update_valence(self, valence: float) -> None:
        """Adjust dopamine and serotonin based on subjective valence."""
        pos = max(0.0, float(valence))
        neg = max(0.0, float(-valence))
        self.dopamine = 0.95 * self.dopamine + pos
        self.serotonin = 0.95 * self.serotonin + neg
        self.dopamine = max(0.0, min(1.0, self.dopamine))
        self.serotonin = max(0.0, min(1.0, self.serotonin))

    def adjust_inhibition(self, baseline: float) -> None:
        """Modify hormone levels based on subthalamic nucleus baseline.

        The baseline itself is filtered with the same ``trend_rate`` to produce
        a running mean.  Hormone levels react to deviations from that mean which
        approximates how inhibitory control adapts over time
        【F:human_brain_components_reference.txt†L246-L250】.
        """

        self.baseline_avg = (
            1 - self.trend_rate
        ) * self.baseline_avg + self.trend_rate * baseline
        delta = baseline - self.baseline_avg

        self.norepinephrine = (
            0.98 * self.norepinephrine + 0.02 * baseline + 0.05 * delta
        )
        self.acetylcholine = (
            0.98 * self.acetylcholine + 0.01 * baseline + 0.02 * delta
        )

        self.norepinephrine = max(0.0, min(1.0, self.norepinephrine))
        self.acetylcholine = max(0.0, min(1.0, self.acetylcholine))

    def log_levels(self, logger: "logging.Logger") -> None:
        """Emit current hormone levels to ``logger``."""
        logger.info(
            "dopamine=%.3f norepinephrine=%.3f serotonin=%.3f acetylcholine=%.3f",
            self.dopamine,
            self.norepinephrine,
            self.serotonin,
            self.acetylcholine,
        )
