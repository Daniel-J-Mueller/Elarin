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
        num_candidates: int = 1,
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
        self.num_candidates = max(1, int(num_candidates))
        
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
    def act(
        self,
        hidden: torch.Tensor,
        temperature: float | None = None,
        num_candidates: int | None = None,
    ) -> tuple[str, torch.Tensor, torch.Tensor, int]:
        """Generate speculative tokens and return the chosen one.

        Each candidate token is re-embedded via :class:`WernickesArea` so that
        its semantic representation can be compared against ``hidden``. All
        candidate embeddings are returned for training, while only the best
        matching token is printed and looped back into the cortex. This
        preserves rich learning signal from every candidate without flooding the
        output stream.

        Parameters
        ----------
        hidden:
            Context embedding from the DMN.
        temperature:
            Optional sampling temperature.  ``None`` uses the current
            norepinephrine level.
        num_candidates:
            Number of speculative tokens to generate.  When ``None`` the
            ``MotorCortex`` instance's ``num_candidates`` value is used.
        Returns
        -------
        tuple
            ``(text, chosen_emb, all_embs, index)`` where ``text`` is the
            selected string, ``chosen_emb`` is its embedding, ``all_embs`` are
            embeddings for every candidate, and ``index`` is the chosen
            candidate's position within ``all_embs``.
        """

        temp = temperature
        if temp is None:
            if self.axis is not None:
                temp = 1.0 + float(self.axis.norepinephrine)
            else:
                temp = 1.0

        n = num_candidates if num_candidates is not None else self.num_candidates

        candidates = list(
            self.area.decode(hidden.to(self.device), temperature=temp, num_samples=n)
        )
        texts = [t for t, _ in candidates]

        # Re-encode each candidate through Wernicke's Area to obtain semantic
        # embeddings for comparison with the DMN context.
        enc = self.wernicke.encode(texts)
        enc_means = enc.mean(dim=1)

        # Select the candidate whose meaning most closely matches the context.
        context_vec = hidden.to(enc_means.device)
        if context_vec.dim() == 3:
            context_vec = context_vec.mean(dim=1)
        sims = torch.nn.functional.cosine_similarity(enc_means, context_vec.squeeze(0), dim=1)
        best_idx = int(torch.argmax(sims).item())
        best_text = texts[best_idx]
        self.logger.info(best_text)

        chosen_emb = enc[best_idx : best_idx + 1]
        return best_text, chosen_emb, enc, best_idx

    @torch.no_grad()
    def learn_from_feedback(
        self,
        vision_feat: torch.Tensor,
        audio_emb: torch.Tensor,
        motor_embs: torch.Tensor,
        trainer: Trainer,
    ) -> None:
        """Align motor output with visual and auditory context.

        ``motor_embs`` contains the embeddings for *all* speculative tokens
        generated during :meth:`act`. Each candidate token contributes to
        adaptation by being aligned separately against the visual and auditory
        cues. This ensures learning incorporates every possibility even though
        only one token is ultimately emitted.
        """
        vision_target = self.vision_to_text(vision_feat.to(self.device))
        for emb in motor_embs:
            # ``emb`` has shape ``(seq_len, hidden)``. Preserve token-level
            # information by aligning each token separately.
            for tok in emb:
                tok = tok.unsqueeze(0)
                trainer.align(
                    [self.area.model.transformer, self.vision_to_text],
                    vision_target,
                    tok,
                )
                trainer.align(
                    [self.area.model.transformer],
                    audio_emb.to(self.device),
                    tok,
                )
