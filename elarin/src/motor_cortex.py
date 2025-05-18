"""Text motor output using the back half of GPT-2."""

import torch
from torch import nn

from .language_areas.brocas_area import BrocasArea
from .language_areas.wernickes_area import WernickesArea
from .trainer import Trainer
from .utils.logger import get_logger


class MotorCortex:
    """Generates text from context embeddings and prints it."""

    def __init__(self, model_dir: str, wernicke: WernickesArea, device: str = "cpu"):
        self.logger = get_logger("motor_cortex")
        self.area = BrocasArea(model_dir, device=device)
        self.wernicke = wernicke
        self.device = device
        self.vision_to_text = nn.Linear(128, self.area.model.config.n_embd).to(device)

    @torch.no_grad()
    def act(self, hidden: torch.Tensor) -> tuple[str, torch.Tensor]:
        text = next(iter(self.area.decode(hidden.to(self.device))))
        self.logger.info(text)
        loop_emb = self.wernicke.encode([text]).mean(dim=1)
        return text, loop_emb

    @torch.no_grad()
    def learn_from_feedback(
        self,
        vision_feat: torch.Tensor,
        audio_emb: torch.Tensor,
        motor_emb: torch.Tensor,
        trainer: Trainer,
    ) -> None:
        """Align motor output with visual and auditory context."""
        vision_target = self.vision_to_text(vision_feat.to(self.device))
        trainer.align(
            [self.area.model.transformer, self.vision_to_text],
            vision_target,
            motor_emb,
        )
        trainer.align(
            [self.area.model.transformer],
            audio_emb.to(self.device),
            motor_emb,
        )
