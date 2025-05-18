"""Text motor output using the back half of GPT-2."""

import torch

from .language_areas.brocas_area import BrocasArea
from .language_areas.wernickes_area import WernickesArea
from .utils.logger import get_logger


class MotorCortex:
    """Generates text from context embeddings and prints it."""

    def __init__(self, model_dir: str, wernicke: WernickesArea, device: str = "cpu"):
        self.logger = get_logger("motor_cortex")
        self.area = BrocasArea(model_dir, device=device)
        self.wernicke = wernicke
        self.device = device

    @torch.no_grad()
    def act(self, hidden: torch.Tensor) -> tuple[str, torch.Tensor]:
        text = next(iter(self.area.decode(hidden.to(self.device))))
        self.logger.info(text)
        loop_emb = self.wernicke.encode([text]).mean(dim=1)
        return text, loop_emb
