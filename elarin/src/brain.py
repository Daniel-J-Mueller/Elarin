"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch
import time
import cv2
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .language_areas.wernickes_area import WernickesArea
from .language_areas.augmenter import LanguageAugmenter
from .default_mode_network import DefaultModeNetwork
from .motor_cortex import MotorCortex
from .hypothalamus_pituitary_axis import HypothalamusPituitaryAxis
from .hippocampus import Hippocampus
from .thalamus import Thalamus
from .trainer import Trainer
from .utils.config import load_config
from .utils.logger import get_logger
from .viewer import Viewer
from .utils.camera import Camera
from .utils.audio_buffer import AudioBuffer


def main() -> None:
    cfg = load_config("configs/default.yaml")
    devices = cfg["devices"]
    models = cfg["models"]
    persist_dir = Path(cfg.get("persistent_dir", "persistent"))
    settings = cfg.get("settings", {})
    loop_interval = float(settings.get("loop_interval", 0.05))
    audio_duration = float(settings.get("audio_duration", 1.0))
    debug_no_video = bool(settings.get("debug_no_video", False))
    hippocampus_capacity = int(settings.get("hippocampus_capacity", 1000))
    motor_candidates = int(settings.get("motor_candidates", 1))

    if not persist_dir.is_absolute():
        from .utils.config import BASE_DIR
        persist_dir = BASE_DIR / persist_dir

    logger = get_logger("brain")

    retina = Retina(models["clip"], device=devices["retina"])
    occipital = OccipitalLobe(device=devices["occipital_lobe"])

    wernicke = WernickesArea(models["gpt2"], device=devices["language_areas"])

    dmn = DefaultModeNetwork(intero_dim=768, hidden_dim=2048, output_dim=768, num_layers=4).to(devices["dmn"])
    # Weight sensory inputs slightly higher than internal interoceptive signals
    dmn.set_modality_weights(vision=1.2, audio=1.2, intero=1.0)
    hippocampus = Hippocampus(
        dims={
            "vision": 128,
            "audio": 768,
            "intero": 768,
            "context": 768,
            "motor": 768,
        },
        capacity=hippocampus_capacity,
        persist_path=f"{persist_dir}/hippocampus.npy",
    )
    axis = HypothalamusPituitaryAxis()
    augmenter = LanguageAugmenter(
        device=devices["language_areas"],
        persist_path=f"{persist_dir}/angular_gyrus.pt",
    )
    motor = MotorCortex(
        models["gpt2"],
        wernicke,
        device=devices["motor_cortex"],
        axis=axis,
        persist_path=f"{persist_dir}/motor.pt",
        num_candidates=motor_candidates,
    )

    thalamus = Thalamus()
    trainer = Trainer()

    logger.info("starting live loop; press Ctrl+C to stop")
    dmn_device = devices["dmn"]


    cam = None
    viewer = None
    if not debug_no_video:
        cam = Camera()
        viewer = Viewer(224, 224)
    asr_processor = WhisperProcessor.from_pretrained(models["whisper"])
    asr_model = WhisperForConditionalGeneration.from_pretrained(models["whisper"])
    asr_device = devices.get("cochlea", "cpu")
    asr_model.to(asr_device)
    asr_model.eval()
    audio_buf = AudioBuffer(samplerate=16000, channels=1, buffer_seconds=audio_duration * 2)

    try:
        while True:
            if not debug_no_video:
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
            else:
                frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                vision_feat = torch.zeros(1, 128, device=devices["occipital_lobe"])

            audio_np = audio_buf.read(audio_duration)
            # Compute a simple RMS volume estimate and boost the gain for display
            audio_level = float(np.sqrt(np.mean(audio_np ** 2))) * 10.0
            inputs = asr_processor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(asr_device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(asr_device)

            prompt_ids = asr_processor.get_decoder_prompt_ids(
                language="en", task="transcribe"
            )

            generation_args = {
                "forced_decoder_ids": prompt_ids,
                "max_new_tokens": 16,
            }
            if attention_mask is not None:
                generation_args["attention_mask"] = attention_mask

            predicted_ids = asr_model.generate(
                input_features,
                **generation_args,
            )
            spoken = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            if spoken:
                text_emb = wernicke.encode([spoken]).mean(dim=1)
            else:
                text_emb = torch.zeros(
                    1,
                    wernicke.model.config.n_embd,
                    device=wernicke.device,
                )
            text_emb = augmenter(text_emb)
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
                "context", context.squeeze(0).detach().cpu().numpy(), k=5
            )
            if recalled:
                if "context" in recalled:
                    recall_ctx = torch.tensor(recalled["context"], device=dmn_device).unsqueeze(0)
                    # Prioritize new sensory context over recalled thoughts
                    context = 0.7 * context + 0.3 * recall_ctx
                # Push other modalities back through the thalamus for replay
                for modality in ("vision", "audio", "intero", "motor"):
                    if modality in recalled:
                        thalamus.submit(modality, torch.tensor(recalled[modality], device=dmn_device).unsqueeze(0))

            out_text, out_emb, cand_embs, best_idx = motor.act(context)
            cand_aug = augmenter(cand_embs)
            out_aug = cand_aug[best_idx : best_idx + 1]
            motor.learn_from_feedback(vision_feat, text_emb, cand_aug, trainer)

            hippocampus.add_episode(
                {
                    "vision": vision.squeeze(0).detach().cpu().numpy(),
                    "audio": audio.squeeze(0).detach().cpu().numpy(),
                    "intero": intero.squeeze(0).detach().cpu().numpy(),
                    "context": context.squeeze(0).detach().cpu().numpy(),
                    "motor": out_aug.squeeze(0).detach().cpu().numpy(),
                }
            )
            thalamus.submit("intero", out_aug)
            trainer.step([dmn.fusion, augmenter], context)
            # Align DMN output with the embeddings of the speculative tokens so
            # future contexts better predict likely next words. ``Trainer.align``
            # updates parameters to make ``actual`` closer to ``target``. Each
            # candidate embedding acts as a desired target while the DMN context
            # serves as the current prediction.
            for emb in cand_aug:
                # ``cand_aug`` has shape ``(num_candidates, seq_len, hidden)``.
                # Align each token separately to preserve detailed feedback.
                for tok in emb:
                    tok = tok.unsqueeze(0)
                    trainer.align(
                        [dmn.fusion, motor.area.model.transformer, augmenter],
                        tok,
                        context,
                    )

            hippocampus.decay()

            if viewer:
                viewer.update(frame_rgb, out_text, audio_level)
                taught = viewer.poll_text_input()
            else:
                taught = None
            if taught:
                teach_emb = wernicke.encode([taught]).mean(dim=1)
                teach_emb = augmenter(teach_emb)
                hippocampus.add_episode({"motor": teach_emb.squeeze(0).detach().cpu().numpy()})
                thalamus.submit("intero", teach_emb)
                trainer.step(
                    [dmn.fusion, motor.area.model.transformer, augmenter],
                    teach_emb,
                    lr_scale=2.0,
                )
            time.sleep(loop_interval)
    except KeyboardInterrupt:
        logger.info("demo interrupted")
    finally:
        if cam:
            cam.release()
        if viewer:
            viewer.close()
        audio_buf.close()
        hippocampus.save()
        motor.save()
        augmenter.save()


if __name__ == "__main__":
    main()
