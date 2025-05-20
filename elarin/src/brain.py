"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch
import time
import cv2
import numpy as np
from pathlib import Path
from .sensors.speech_recognizer import SpeechRecognizer

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .language_areas.wernickes_area import WernickesArea
from .language_areas.augmenter import LanguageAugmenter
from .insular_cortex import InsularCortex
from .default_mode_network import DefaultModeNetwork
from .motor_cortex import MotorCortex
from .hypothalamus_pituitary_axis import HypothalamusPituitaryAxis
from .insular_cortex import InsularCortex
from .hippocampus import Hippocampus
from .semantic_flow import SemanticFlow
from .thalamus import Thalamus
from .trainer import Trainer
from .temporal_lobe import TemporalLobe
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

    wernicke = WernickesArea(
        models["gpt2"],
        device=devices["language_areas"],
        token_table_path=f"{persist_dir}/token_embeddings.npy",
    )

    dmn = DefaultModeNetwork(intero_dim=768, hidden_dim=2048, output_dim=768, num_layers=4).to(devices["dmn"])
    # Weight sensory inputs slightly higher than internal interoceptive signals
    # Increase audio and vision emphasis to encourage responsiveness
    dmn.set_modality_weights(vision=1.4, audio=1.4, intero=1.0)
    hippocampus = Hippocampus(
        dims={
            "vision": 128,
            "audio": 768,
            "intero": 768,
            "context": 768,
            "motor": 768,
            "speech": 768,
        },
        capacity=hippocampus_capacity,
        persist_path=f"{persist_dir}/hippocampus.npy",
    )
    axis = HypothalamusPituitaryAxis()
    insular = InsularCortex(
        device=devices["dmn"],
        persist_path=f"{persist_dir}/insular.pt",
    )
    temporal = TemporalLobe()
    flow = SemanticFlow(len(wernicke.tokenizer), persist_path=f"{persist_dir}/semantic_flow.json")
    augmenter = LanguageAugmenter(
        device=devices["language_areas"],
        persist_path=f"{persist_dir}/angular_gyrus.pt",
    )
    insula = InsularCortex(
        device=devices["motor_cortex"],
        persist_path=f"{persist_dir}/insula.pt",
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
    prev_context = None


    cam = None
    viewer = None
    if not debug_no_video:
        cam = Camera()
        viewer = Viewer(224, 224)
    speech_recognizer = SpeechRecognizer(models["whisper"], device=devices.get("cochlea", "cpu"))
    audio_buf = AudioBuffer(samplerate=16000, channels=1, buffer_seconds=audio_duration * 2)

    try:
        while True:
            # Adjust DMN modality weights based on hormone levels
            vis_w = 1.4 + 0.4 * axis.dopamine - 0.2 * axis.serotonin
            aud_w = 1.4 + 0.4 * axis.dopamine - 0.2 * axis.serotonin
            intero_w = 1.0 + 0.5 * axis.serotonin - 0.2 * axis.dopamine - 0.1 * axis.acetylcholine
            dmn.set_modality_weights(vis_w, aud_w, intero_w)
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
            spoken = ""
            # Ignore near-silent audio to avoid hallucinated transcripts
            if audio_level > 0.02:
                spoken = speech_recognizer.transcribe(audio_np)
            if spoken:
                flow.observe(wernicke.tokenizer.encode(spoken))
                text_emb = wernicke.encode([spoken]).mean(dim=1)
            else:
                text_emb = torch.zeros(
                    1,
                    wernicke.model.config.n_embd,
                    device=wernicke.device,
                )
            user_emb = augmenter(text_emb)
            spec_emb = temporal.embedding(wernicke)
            spec_emb = augmenter(spec_emb)
            if spoken:
                combined_audio = 0.8 * user_emb + 0.2 * spec_emb
            else:
                combined_audio = spec_emb
            thalamus.submit("audio", combined_audio)

            vision = thalamus.relay("vision")
            if vision is None:
                vision = torch.zeros(1, 128, device=dmn_device)
            else:
                vision = vision.to(dmn_device)

            audio = thalamus.relay("audio")
            if audio is None:
                audio = torch.zeros(1, user_emb.size(-1), device=dmn_device)
            else:
                audio = audio.to(dmn_device)

            intero = thalamus.relay("intero")
            if intero is None:
                intero = torch.zeros(1, 768, device=dmn_device)
            else:
                intero = intero.to(dmn_device)
                if intero.dim() == 3:
                    intero = intero.mean(dim=1)

            context = dmn(vision, audio, intero)

            if prev_context is None:
                novelty = 1.0
            else:
                sim = torch.nn.functional.cosine_similarity(
                    context.view(-1), prev_context.view(-1), dim=0
                ).clamp(min=0.0)
                novelty = float(1.0 - sim.item())

            axis.step(novelty, 0.0)
            prev_context = context.detach()
            recalled = hippocampus.query(
                "context", context.squeeze(0).detach().cpu().numpy(), k=5
            )
            if recalled:
                if "context" in recalled:
                    recall_ctx = torch.tensor(recalled["context"], device=dmn_device).unsqueeze(0)
                    # Prioritize new sensory context over recalled thoughts
                    context = 0.7 * context + 0.3 * recall_ctx
                # Push other modalities back through the thalamus for replay
                for modality in ("vision", "audio", "intero", "motor", "speech"):
                    if modality in recalled:
                        tensor_val = torch.tensor(recalled[modality], device=dmn_device).unsqueeze(0)
                        if modality == "motor":
                            motor_intero = insular(tensor_val)
                            filtered = axis.filter_intero(motor_intero)
                            # Negate feedback to dampen repeated thoughts
                            thalamus.submit("intero", -filtered)
                        else:
                            if modality == "speech":
                                thalamus.submit("audio", tensor_val)
                            else:
                                thalamus.submit(modality, tensor_val)

            out_text, out_emb, cand_embs, best_idx, cand_texts = motor.act(context)
            flow.observe(wernicke.tokenizer.encode(out_text))
            temporal.add_speculation(cand_texts)
            temporal.consume(out_text)
            cand_aug = augmenter(cand_embs)
            out_aug = cand_aug[best_idx : best_idx + 1]
            motor.learn_from_feedback(vision_feat, user_emb, cand_aug, trainer)

            insula_emb = insula(out_aug)
            hippocampus.add_episode(
                {
                    "vision": vision.squeeze(0).detach().cpu().numpy(),
                    "audio": audio.squeeze(0).detach().cpu().numpy(),
                    "intero": intero.squeeze(0).detach().cpu().numpy(),
                    "context": context.squeeze(0).detach().cpu().numpy(),
                    "motor": insula_emb.squeeze(0).detach().cpu().numpy(),
                    "speech": user_emb.squeeze(0).detach().cpu().numpy(),
                }
            )
            motor_intero = insular(out_aug)
            filtered = axis.filter_intero(motor_intero)
            # Negate feedback to dampen repeated thoughts
            thalamus.submit("intero", -filtered)
            trainer.step(
                [
                    dmn.fusion,
                    augmenter,
                    motor.damp_lora,
                    motor.long_lora,
                    insular.short_lora,
                    insular.long_lora,
                    insula.short_lora,
                    insula.long_lora,
                ],
                context,
            )

            hippocampus.decay()

            if viewer:
                viewer.update(frame_rgb, out_text, audio_level)
                taught = viewer.poll_text_input()
            else:
                taught = None
            if taught:
                flow.observe(wernicke.tokenizer.encode(taught))
                teach_emb = wernicke.encode([taught]).mean(dim=1)
                teach_emb = augmenter(teach_emb)
                hippocampus.add_episode({
                    "motor": teach_emb.squeeze(0).detach().cpu().numpy(),
                    "speech": teach_emb.squeeze(0).detach().cpu().numpy(),
                })
                motor_intero = insular(teach_emb)
                filtered = axis.filter_intero(motor_intero)
                # Negate feedback to dampen repeated thoughts
                thalamus.submit("intero", -filtered)
                trainer.step(
                    [
                        dmn.fusion,
                        motor.area.model.transformer,
                        augmenter,
                        motor.damp_lora,
                        motor.long_lora,
                        insular.short_lora,
                        insular.long_lora,
                        insula.short_lora,
                        insula.long_lora,
                    ],
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
        insular.save()
        insula.save()
        hippocampus.save()
        motor.save()
        augmenter.save()
        flow.save()


if __name__ == "__main__":
    main()
