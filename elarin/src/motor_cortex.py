"""Text motor output using the back half of GPT-2."""

import torch
from torch import nn
from pathlib import Path

from .language_areas.brocas_area import BrocasArea
from .language_areas.wernickes_area import WernickesArea
from .hypothalamus_pituitary_axis import HypothalamusPituitaryAxis
from .trainer import Trainer
from .utils.logger import get_logger


class MotorCortex:
    """Generates text from context embeddings and prints it."""

    def __init__(
        self,
        model_dir: str,
        wernicke: WernickesArea,
        device: str = "cpu",
        axis: HypothalamusPituitaryAxis | None = None,
        persist_path: str | None = None,
    ) -> None:
        self.logger = get_logger("motor_cortex")
        self.area = BrocasArea(model_dir, device=device)
        self.wernicke = wernicke
        self.axis = axis
        self.device = device
        self.vision_to_text = nn.Linear(128, self.area.model.config.n_embd).to(device)
        if persist_path and Path(persist_path).exists():
            state = torch.load(persist_path, map_location=device)
            self.area.model.load_state_dict(state.get("broca", {}), strict=False)
            self.vision_to_text.load_state_dict(state.get("vision_to_text", {}), strict=False)
        self.persist_path = persist_path
        
    def save(self, path: str | None = None) -> None:
        """Save adapter parameters for later reloading."""
        target = path or self.persist_path
        if not target:
            return
        torch.save(
            {
                "broca": self.area.model.state_dict(),
                "vision_to_text": self.vision_to_text.state_dict(),
            },
            target,
        )

    @torch.no_grad()
    def act(self, hidden: torch.Tensor, temperature: float | None = None) -> tuple[str, torch.Tensor]:
        """Generate a token from ``hidden`` and return it with its embedding."""
        temp = temperature
        if temp is None:
            if self.axis is not None:
                temp = 1.0 + float(self.axis.norepinephrine)
            else:
                temp = 1.0
        text = next(
            iter(self.area.decode(hidden.to(self.device), temperature=temp))
        )
        self.logger.info(text)
        # Convert the generated token back to its ID and pull the corresponding
        # embedding directly from Wernicke's input layer so that only the most
        # recent token influences the next context step.
        tok_id = self.wernicke.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.wernicke.device)[0, -1]
        loop_emb = self.wernicke.model.get_input_embeddings()(tok_id.unsqueeze(0))
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
