"""Text motor output using the back half of GPT-2."""

import torch
from typing import Iterable

from .language_areas.brocas_area import BrocasArea
from .utils.logger import get_logger


class MotorCortex:
    """Generates text from context embeddings and prints it."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.logger = get_logger("motor_cortex")
        self.area = BrocasArea(model_dir)
        self.area.model.to(device)
        self.device = device

    @torch.no_grad()
    def act(self, hidden: torch.Tensor) -> str:
        text = next(iter(self.area.decode(hidden.to(self.device))))
        self.logger.info(text)
        return text
