"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch
import time
import cv2
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .language_areas.wernickes_area import WernickesArea
from .default_mode_network import DefaultModeNetwork
from .motor_cortex import MotorCortex
from .thalamus import Thalamus
from .trainer import Trainer
from .utils.config import load_config
from .utils.logger import get_logger
from .viewer import render, show


def main() -> None:
    cfg = load_config("configs/default.yaml")
    devices = cfg["devices"]
    models = cfg["models"]

    logger = get_logger("brain")

    retina = Retina(models["clip"], device=devices["retina"])
    occipital = OccipitalLobe(device=devices["occipital_lobe"])

    wernicke = WernickesArea(models["gpt2"], device=devices["language_areas"])

    dmn = DefaultModeNetwork().to(devices["dmn"])
    motor = MotorCortex(models["gpt2"], device=devices["motor_cortex"])

    thalamus = Thalamus()
    trainer = Trainer()

    logger.info("starting live loop; press Ctrl+C to stop")
    dmn_device = devices["dmn"]

    cam = cv2.VideoCapture(0)
    asr_processor = WhisperProcessor.from_pretrained(models["whisper"])
    asr_model = WhisperForConditionalGeneration.from_pretrained(models["whisper"])
    asr_device = devices.get("cochlea", "cpu")
    asr_model.to(asr_device)
    asr_model.eval()

    try:
        while True:
            ret, frame_bgr = cam.read()
            if not ret:
                logger.warning("camera frame not captured")
                img = Image.new("RGB", (224, 224), color="white")
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).resize((224, 224))

            vision_emb = retina.encode([img]).to(devices["occipital_lobe"])
            vision_feat = occipital.process(vision_emb)
            thalamus.submit("vision", vision_feat)

            audio_samples = sd.rec(int(1.0 * 16000), samplerate=16000, channels=1)
            sd.wait()
            audio_np = audio_samples.squeeze().astype(np.float32)
            inputs = asr_processor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(asr_device)
            predicted_ids = asr_model.generate(input_features)
            spoken = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            text_emb = wernicke.encode([spoken]).mean(dim=1)
            thalamus.submit("audio", text_emb)

            vision = thalamus.relay("vision")
            if vision is None:
                vision = torch.zeros(1, 128, device=dmn_device)
            else:
                vision = vision.to(dmn_device)

            audio = thalamus.relay("audio")
            if audio is None:
                audio = torch.zeros(1, text_emb.size(-1), device=dmn_device)
            else:
                audio = audio.to(dmn_device)

            intero = torch.zeros(1, 64, device=dmn_device)

            context = dmn(vision, audio, intero)
            out_text = motor.act(context)
            trainer.step([dmn.fusion], context)

            frame = render(img, out_text)
            show(frame)
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("demo interrupted")


if __name__ == "__main__":
    main()
