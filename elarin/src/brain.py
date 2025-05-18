"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch
import time

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

    try:
        while True:
            # placeholder sensory capture
            img = Image.new("RGB", (224, 224), color="white")
            vision_emb = retina.encode([img]).to(devices["occipital_lobe"])
            vision_feat = occipital.process(vision_emb)
            thalamus.submit("vision", vision_feat)

            text_emb = wernicke.encode(["hello world"]).mean(dim=1)
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
