import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class SpeechRecognizer:
    """Transcribe audio to text using Whisper."""

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        *,
        language: str = "en",
        task: str = "transcribe",
        max_new_tokens: int = 16,
    ) -> str:
        """Return the transcription for ``audio``."""
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        prompt_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
        generation_args = {"forced_decoder_ids": prompt_ids, "max_new_tokens": max_new_tokens}
        if attention_mask is not None:
            generation_args["attention_mask"] = attention_mask
        pred_ids = self.model.generate(input_features, **generation_args)
        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        return text
