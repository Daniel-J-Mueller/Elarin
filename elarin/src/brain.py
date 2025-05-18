"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .language_areas.wernickes_area import WernickesArea
from .default_mode_network import DefaultModeNetwork
from .motor_cortex import MotorCortex
from .thalamus import Thalamus
from .utils.config import load_config
from .utils.logger import get_logger
from .viewer import render


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

    # Example: single step with dummy image and text
    img = Image.new("RGB", (224, 224), color="white")
    vision_emb = retina.encode([img]).to(devices["occipital_lobe"])
    vision_feat = occipital.process(vision_emb)
    thalamus.submit("vision", vision_feat)

    text_emb = wernicke.encode(["hello world"]).mean(dim=1)
    thalamus.submit("audio", text_emb)

    vision = thalamus.relay("vision")
    if vision is None:
        vision = torch.zeros(1, 128)

    audio = thalamus.relay("audio")
    if audio is None:
        audio = torch.zeros(1, text_emb.size(-1))
    intero = torch.zeros(1, 64)

    context = dmn(vision, audio, intero)
    out_text = motor.act(context)

    frame = render(img, out_text)
    frame.show()
    logger.info("demo complete")


if __name__ == "__main__":
    main()
