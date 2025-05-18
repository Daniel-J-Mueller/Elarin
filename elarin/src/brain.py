"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch
import time
import cv2
import sounddevice as sd
import numpy as np
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .language_areas.wernickes_area import WernickesArea
from .default_mode_network import DefaultModeNetwork
from .motor_cortex import MotorCortex
from .hippocampus import Hippocampus
from .thalamus import Thalamus
from .trainer import Trainer
from .utils.config import load_config
from .utils.logger import get_logger
from .viewer import Viewer
from .utils.camera import Camera


def main() -> None:
    cfg = load_config("configs/default.yaml")
    devices = cfg["devices"]
    models = cfg["models"]

    logger = get_logger("brain")

    retina = Retina(models["clip"], device=devices["retina"])
    occipital = OccipitalLobe(device=devices["occipital_lobe"])

    wernicke = WernickesArea(models["gpt2"], device=devices["language_areas"])

    dmn = DefaultModeNetwork(intero_dim=768).to(devices["dmn"])
    hippocampus = Hippocampus(
        dims={
            "vision": 128,
            "audio": 768,
            "intero": 768,
            "context": 768,
            "motor": 768,
        }
    )
    motor = MotorCortex(models["gpt2"], wernicke, device=devices["motor_cortex"])

    thalamus = Thalamus()
    trainer = Trainer()

    logger.info("starting live loop; press Ctrl+C to stop")
    dmn_device = devices["dmn"]

    prev_emb: torch.Tensor | None = None
    repeat_count = 0

    cam = Camera()
    viewer = Viewer(224, 224)
    asr_processor = WhisperProcessor.from_pretrained(models["whisper"])
    asr_model = WhisperForConditionalGeneration.from_pretrained(models["whisper"])
    asr_device = devices.get("cochlea", "cpu")
    asr_model.to(asr_device)
    asr_model.eval()

    try:
        while True:
            frame_bgr = cam.read()
            if frame_bgr is None:
                logger.warning("camera frame not captured")
                img = Image.new("RGB", (224, 224), color="white")
                frame_rgb = np.array(img)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).resize((224, 224))

            vision_emb = retina.encode([img]).to(devices["occipital_lobe"])
            vision_feat = occipital.process(vision_emb)
            thalamus.submit("vision", vision_feat)

            audio_samples = sd.rec(int(1.0 * 16000), samplerate=16000, channels=1)
            sd.wait()
            audio_np = audio_samples.squeeze().astype(np.float32)
            # Compute a simple RMS volume estimate and boost the gain for display
            audio_level = float(np.sqrt(np.mean(audio_np ** 2))) * 10.0
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

            intero = thalamus.relay("intero")
            if intero is None:
                intero = torch.zeros(1, 768, device=dmn_device)
            else:
                intero = intero.to(dmn_device)

            context = dmn(vision, audio, intero)
            recalled = hippocampus.query(
                "context", context.squeeze(0).cpu().numpy(), k=5
            )
            if recalled:
                addition = []
                for key, val in recalled.items():
                    addition.append(torch.tensor(val, device=dmn_device).unsqueeze(0))
                context = context + sum(addition)

            out_text, out_emb = motor.act(context)
            # Track repetition of motor output
            if prev_emb is not None:
                sim = F.cosine_similarity(out_emb.squeeze(0), prev_emb, dim=0)
                if sim > 0.98:
                    repeat_count += 1
                else:
                    repeat_count = 0
            prev_emb = out_emb.squeeze(0)

            if repeat_count >= 10:
                logger.warning("output repetition detected")
                noise = torch.randn_like(out_emb) * 0.01
                thalamus.submit("intero", out_emb + noise)
                repeat_count = 0

            hippocampus.add_episode(
                {
                    "vision": vision.squeeze(0).cpu().numpy(),
                    "audio": audio.squeeze(0).cpu().numpy(),
                    "intero": intero.squeeze(0).cpu().numpy(),
                    "context": context.squeeze(0).cpu().numpy(),
                    "motor": out_emb.squeeze(0).cpu().numpy(),
                }
            )
            thalamus.submit("intero", out_emb)
            trainer.step([dmn.fusion], context)

            hippocampus.decay()

            viewer.update(frame_rgb, out_text, audio_level)
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("demo interrupted")
    finally:
        cam.release()
        viewer.close()


if __name__ == "__main__":
    main()
