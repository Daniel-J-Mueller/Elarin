"""Simple integration demo linking sensors, DMN and motor cortex."""

from PIL import Image
import torch
import time
import cv2
import numpy as np
from pathlib import Path

from .utils.message_bus import MessageBus
from .sensors.cochlea import Cochlea
from .auditory_cortex import AuditoryCortex
from .primary_auditory_cortex import PrimaryAuditoryCortex

from .sensors.retina import Retina
from .occipital_lobe import OccipitalLobe
from .primary_visual_cortex import PrimaryVisualCortex
from .language_areas.wernickes_area import WernickesArea
from .language_areas.wernicke_adapter import WernickeAdapter
from .insular_cortex import InsularCortex
from .basal_ganglia import BasalGanglia
from .subthalamic_nucleus import SubthalamicNucleus
from .cerebellum import Cerebellum
from .corpus_callosum import CorpusCallosum
from .amygdala import Amygdala
from .frontal_lobe import FrontalLobe
from .prefrontal_cortex import PrefrontalCortex
from .default_mode_network import DefaultModeNetwork
from .motor_cortex import MotorCortex
from .hypothalamus_pituitary_axis import HypothalamusPituitaryAxis
from .insular_cortex import InsularCortex
from .hippocampus import Hippocampus, DistributedHippocampus
from .thalamus import Thalamus
from .trainer import Trainer
from .temporal_lobe import TemporalLobe
from .parietal_lobe import ParietalLobe
from .entorhinal_cortex import EntorhinalCortex
from .cingulate_cortex import CingulateCortex
from .midbrain import Midbrain
from .pons import Pons
from .medulla_oblongata import MedullaOblongata
from .pituitary_gland import PituitaryGland
from .utils.config import load_config, BASE_DIR
from .utils.logger import get_logger, enable_file_logging
from .viewer import Viewer
from .utils.camera import Camera
from .utils.audio_buffer import AudioBuffer


def main() -> None:
    cfg = load_config("configs/default.yaml")
    devices = cfg["devices"]
    models = cfg["models"]
    persist_dir = Path(cfg.get("persistent_dir", "persistent"))
    log_dir = Path(cfg.get("log_dir", "logs"))
    settings = cfg.get("settings", {})
    loop_interval = float(settings.get("loop_interval", 0.05))
    audio_duration = float(settings.get("audio_duration", 1.0))
    debug_no_video = bool(settings.get("debug_no_video", False))
    hippocampus_capacity = int(settings.get("hippocampus_capacity", 1000))
    recall_threshold = float(settings.get("hippocampus_recall_threshold", 0.0))
    hippocampus_shards = int(settings.get("hippocampus_shards", 1))
    salience_thresh = float(
        settings.get("hippocampus_salience_threshold", 0.0)
    )
    motor_candidates = int(settings.get("motor_candidates", 1))
    log_to_file = bool(settings.get("log_to_file", False))

    if not persist_dir.is_absolute():
        persist_dir = BASE_DIR / persist_dir

    if not log_dir.is_absolute():
        log_dir = BASE_DIR / log_dir

    if log_to_file:
        enable_file_logging(str(log_dir))

    logger = get_logger("brain")
    bus = MessageBus()

    retina = Retina(models["clip"], device=devices["retina"])
    primary_vis = PrimaryVisualCortex(device=devices["occipital_lobe"])
    occipital = OccipitalLobe(device=devices["occipital_lobe"])
    cochlea = Cochlea(models["whisper"], device=devices["cochlea"])
    primary_aud = PrimaryAuditoryCortex(device=devices["auditory_cortex"])
    auditory = AuditoryCortex(device=devices["auditory_cortex"])

    wernicke = WernickesArea(
        models["gpt2"],
        device=devices["language_areas"],
        token_table_path=f"{persist_dir}/token_embeddings.npy",
    )
    like_emb = wernicke.encode(["I like that"]).mean(dim=1).to(devices["dmn"])
    dislike_emb = (
        wernicke.encode(["I don't like that"]).mean(dim=1).to(devices["dmn"])
    )

    dmn = DefaultModeNetwork(
        audio_dim=896,
        intero_dim=768,
        hidden_dim=2048,
        output_dim=768,
        num_layers=4,
    ).to(devices["dmn"])
    # Weight sensory inputs slightly higher than internal interoceptive signals
    # Increase audio and vision emphasis to encourage responsiveness
    dmn.set_modality_weights(vision=1.4, audio=1.4, intero=1.0)
    hip_dims = {
        "vision": 128,
        "audio": 896,
        "intero": 768,
        "context": 768,
        "motor": 768,
        "speech": 768,
    }
    if hippocampus_shards > 1:
        shard_paths = [
            f"{persist_dir}/hippocampus_memory_shard_{i}.npz"
            for i in range(hippocampus_shards)
        ]
        hippocampus = DistributedHippocampus(
            hip_dims,
            num_shards=hippocampus_shards,
            shard_paths=shard_paths,
            capacity=hippocampus_capacity,
            recall_threshold=recall_threshold,
            salience_threshold=salience_thresh,
            compressed=True,
        )
    else:
        hippocampus = Hippocampus(
            dims=hip_dims,
            capacity=hippocampus_capacity,
            recall_threshold=recall_threshold,
            persist_path=f"{persist_dir}/hippocampus_memory.npz",
            salience_threshold=salience_thresh,
        )
    amygdala = Amygdala(
        device=devices["dmn"], persist_path=f"{persist_dir}/amygdala_emotion.pt"
    )
    frontal = FrontalLobe(
        device=devices["dmn"], persist_path=f"{persist_dir}/frontal_lobe.pt"
    )
    pfc = frontal.prefrontal
    corpus = CorpusCallosum(
        embed_dim=768,
        device=devices["dmn"],
        persist_path=f"{persist_dir}/corpus_callosum_bridge.pt",
    )
    axis = HypothalamusPituitaryAxis()
    pituitary = PituitaryGland(device=devices["dmn"], persist_path=f"{persist_dir}/pituitary_gland.pt")
    entorhinal = EntorhinalCortex(device=devices["dmn"], persist_path=f"{persist_dir}/entorhinal_cortex.pt")
    parietal = ParietalLobe(device=devices["occipital_lobe"], persist_path=f"{persist_dir}/parietal_lobe.pt")
    cingulate = CingulateCortex(device=devices["dmn"], persist_path=f"{persist_dir}/cingulate_cortex.pt")
    midbrain = Midbrain(device=devices["dmn"], persist_path=f"{persist_dir}/midbrain.pt")
    pons = Pons(device=devices["dmn"], persist_path=f"{persist_dir}/pons.pt")
    medulla = MedullaOblongata(device=devices["dmn"], persist_path=f"{persist_dir}/medulla_oblongata.pt")
    stn = SubthalamicNucleus(device=devices["dmn"])
    basal = BasalGanglia(
        input_dim=768,
        device=devices["dmn"],
        axis=axis,
        prefrontal=pfc,
        stn=stn,
        persist_path=f"{persist_dir}/basal_ganglia_gating.pt",
    )
    insular = InsularCortex(
        device=devices["dmn"],
        persist_path=f"{persist_dir}/insular_mapping.pt",
    )
    temporal = TemporalLobe()
    augmenter = WernickeAdapter(
        device=devices["language_areas"],
        persist_path=f"{persist_dir}/wernicke_adapter.pt",
    )
    insula = InsularCortex(
        device=devices["motor_cortex"],
        persist_path=f"{persist_dir}/motor_insula.pt",
    )
    cerebellum = Cerebellum(
        device=devices.get("cerebellum", devices["motor_cortex"]),
        persist_path=f"{persist_dir}/cerebellum_correction.pt",
    )
    motor = MotorCortex(
        models["gpt2"],
        wernicke,
        device=devices["motor_cortex"],
        axis=axis,
        persist_path=f"{persist_dir}/motor_cortex_generator.pt",
        num_candidates=motor_candidates,
    )

    thalamus = Thalamus()
    trainer = Trainer()

    logger.info("starting live loop; press Ctrl+C to stop")
    dmn_device = devices["dmn"]
    prev_context = None
    silent_steps = 0

    cam = None
    viewer = None
    if not debug_no_video:
        cam = Camera()
        viewer = Viewer(224, 224)
    audio_buf = AudioBuffer(
        samplerate=16000, channels=1, buffer_seconds=audio_duration * 2
    )

    step = 0

    try:
        while True:
            step += 1
            # Adjust DMN modality weights based on hormone levels
            vis_w = 1.4 + 0.4 * axis.dopamine - 0.2 * axis.serotonin
            aud_w = 1.4 + 0.4 * axis.dopamine - 0.2 * axis.serotonin
            intero_w = (
                1.0
                + 0.5 * axis.serotonin
                - 0.2 * axis.dopamine
                - 0.1 * axis.acetylcholine
            )
            dmn.set_modality_weights(vis_w, aud_w, intero_w)

            # simple speculative step removed due to obsolete SemanticFlow
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
                prim_vis = primary_vis.extract(vision_emb)
                vision_feat = occipital.process(prim_vis)
                vision_feat = parietal.attend(vision_feat)
                thalamus.submit("vision", vision_feat)
            else:
                frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                vision_feat = torch.zeros(
                    1, 128, device=devices["occipital_lobe"]
                )

            audio_np = audio_buf.read(audio_duration)
            # Compute a simple RMS volume estimate and boost the gain for display
            audio_level = float(np.sqrt(np.mean(audio_np**2))) * 10.0
            spoken = ""
            audio_feat = torch.zeros(1, 128, device=devices["auditory_cortex"])
            # Ignore near-silent audio to avoid hallucinated transcripts
            if audio_level > 0.02:
                audio_tensor = (
                    torch.from_numpy(audio_np).float().to(cochlea.device)
                )
                spoken = cochlea.transcribe(audio_tensor)
                emb = cochlea.encode([audio_tensor])
                prim_aud = primary_aud.extract(emb)
                audio_feat = auditory.process(prim_aud)
                if audio_feat.dim() == 3:
                    audio_feat = audio_feat.mean(dim=1)
            if spoken:
                text_emb = wernicke.encode([spoken]).mean(dim=1)
                temporal.clear()
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
                text_mix = 0.9 * user_emb + 0.1 * spec_emb
            else:
                text_mix = spec_emb
            combined_audio = torch.cat(
                [
                    text_mix.to(audio_feat.device),
                    audio_feat,
                ],
                dim=-1,
            )
            thalamus.submit("audio", combined_audio)

            vision = thalamus.relay("vision")
            if vision is None:
                vision = torch.zeros(1, 128, device=dmn_device)
            else:
                vision = vision.to(dmn_device)

            audio = thalamus.relay("audio")
            if audio is None:
                audio = torch.zeros(1, 896, device=dmn_device)
            else:
                audio = audio.to(dmn_device)

            intero = thalamus.relay("intero")
            if intero is None:
                intero = torch.zeros(1, 768, device=dmn_device)
            else:
                intero = intero.to(dmn_device)
                if intero.dim() == 3:
                    intero = intero.mean(dim=1)

            # Executive filtering of sensations based on previous context
            if prev_context is not None:
                weights = pfc.filter_weights(prev_context)
                vision = vision * weights["vision"]
                audio = audio * weights["audio"]
                intero = intero * weights["intero"]

            context = dmn(vision, audio, intero)
            context = corpus.transfer(context)

            context_np = context.squeeze(0).detach().cpu().numpy()
            bus.publish("context", context_np.tobytes())

            if prev_context is None:
                novelty = 1.0
            else:
                sim = torch.nn.functional.cosine_similarity(
                    context.view(-1), prev_context.view(-1), dim=0
                ).clamp(min=0.0)
                novelty = float(1.0 - sim.item())

            axis.step(novelty, 0.0)
            axis.norepinephrine = max(
                0.0,
                min(1.0, axis.norepinephrine + pons.boost(torch.tensor([[novelty]])))
            )
            like_sim = torch.nn.functional.cosine_similarity(
                context.view(-1), like_emb.view(-1), dim=0
            ).item()
            dislike_sim = torch.nn.functional.cosine_similarity(
                context.view(-1), dislike_emb.view(-1), dim=0
            ).item()
            axis.update_valence(like_sim - dislike_sim)

            # Prediction step using simple similarity to like/dislike prototypes
            prev_context = context.detach()
            recalled = hippocampus.query(
                "context", context.squeeze(0).detach().cpu().numpy(), k=5
            )
            if recalled:
                if "context" in recalled:
                    recall_ctx = torch.tensor(
                        recalled["context"], device=dmn_device
                    ).unsqueeze(0)
                    # Prioritize new sensory context over recalled thoughts
                    context = 0.7 * context + 0.3 * recall_ctx
                if "valence" in recalled:
                    axis.update_valence(float(recalled["valence"]))
                # Push other modalities back through the thalamus for replay
                for modality in (
                    "vision",
                    "audio",
                    "intero",
                    "motor",
                    "speech",
                ):
                    if modality in recalled:
                        tensor_val = torch.tensor(
                            recalled[modality], device=dmn_device
                        ).unsqueeze(0)
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

            if basal.gate(context):
                out_text, out_emb, cand_embs, best_idx, cand_texts = motor.act(
                    context
                )
                temporal.add_speculation(cand_texts)
                temporal.consume(out_text)
                cand_aug = augmenter(cand_embs)
                out_aug = cand_aug[best_idx : best_idx + 1]
                out_aug = cerebellum.adjust(out_aug, vision_feat)
                vis_as_motor = motor.vision_to_text(
                    vision_feat.to(motor.device)
                )
                trainer.align(
                    [cerebellum.short_lora, cerebellum.long_lora],
                    vis_as_motor.to(cerebellum.device),
                    out_aug.to(cerebellum.device),
                )
                motor.learn_from_feedback(
                    vision_feat, user_emb, cand_aug, trainer
                )
            else:
                out_text = ""
                out_aug = torch.zeros(1, 768, device=devices["motor_cortex"])
            if out_text:
                silent_steps = 0
            else:
                silent_steps += 1
                if silent_steps > 20:
                    axis.dopamine = min(1.0, axis.dopamine + 0.2)
                    silent_steps = 10

            insula_emb = insula(out_aug)
            valence = amygdala.evaluate(context)
            pain_mod = cingulate.modulate(torch.tensor([[valence]]))
            axis.dopamine = max(0.0, min(1.0, axis.dopamine + midbrain.adjust(context)))
            ctx_store = entorhinal.funnel(context)
            hippocampus.add_episode(
                {
                    "vision": vision.squeeze(0).detach().cpu().numpy(),
                    "audio": audio.squeeze(0).detach().cpu().numpy(),
                    "intero": intero.squeeze(0).detach().cpu().numpy(),
                    "context": ctx_store.squeeze(0).detach().cpu().numpy(),
                    "motor": insula_emb.squeeze(0).detach().cpu().numpy(),
                    "speech": user_emb.squeeze(0).detach().cpu().numpy(),
                },
                valence=valence,
                salience=novelty,
            )
            axis.update_valence(valence + pain_mod)
            stn.reinforce(valence)
            axis.adjust_inhibition(stn.baseline)
            axis.memory_pressure(hippocampus.memory_usage_gb())
            motor_intero = insular(out_aug)
            filtered = axis.filter_intero(motor_intero)
            # Negate feedback to dampen repeated thoughts
            thalamus.submit("intero", -filtered)
            trainer.step(
                [
                    dmn.fusion,
                    augmenter,
                    cerebellum.short_lora,
                    cerebellum.long_lora,
                    corpus.short_lora,
                    corpus.long_lora,
                    pfc.short_lora,
                    pfc.long_lora,
                    amygdala.short_lora,
                    amygdala.long_lora,
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

            if log_to_file and step % 50 == 0:
                logger.info(
                    "stn_baseline=%.3f hippo_mem=%.2fGB",
                    stn.baseline,
                    hippocampus.memory_usage_gb(),
                )
                axis.memory_pressure(hippocampus.memory_usage_gb())
                axis.log_levels(logger)

            if viewer:
                viewer.update(frame_rgb, out_text, audio_level)
                taught = viewer.poll_text_input()
            else:
                taught = None
            if taught:
                teach_emb = wernicke.encode([taught]).mean(dim=1)
                teach_emb = augmenter(teach_emb)
                teach_val = amygdala.evaluate(teach_emb)
                tokens = wernicke.tokenizer.encode(taught)
                # training data now collected directly without transition table
                ctx_store = entorhinal.funnel(teach_emb)
                hippocampus.add_episode(
                    {
                        "motor": teach_emb.squeeze(0).detach().cpu().numpy(),
                        "speech": teach_emb.squeeze(0).detach().cpu().numpy(),
                        "context": ctx_store.squeeze(0).detach().cpu().numpy(),
                    },
                    valence=teach_val,
                    salience=1.0,
                )
                axis.update_valence(teach_val + pain_mod)
                stn.reinforce(teach_val)
                axis.adjust_inhibition(stn.baseline)
                axis.memory_pressure(hippocampus.memory_usage_gb())
                motor_intero = insular(teach_emb)
                filtered = axis.filter_intero(motor_intero)
                # Negate feedback to dampen repeated thoughts
                thalamus.submit("intero", -filtered)
                trainer.step(
                    [
                        dmn.fusion,
                        motor.area.model.transformer,
                        augmenter,
                        cerebellum.short_lora,
                        cerebellum.long_lora,
                        corpus.short_lora,
                        corpus.long_lora,
                        pfc.short_lora,
                        pfc.long_lora,
                        amygdala.short_lora,
                        amygdala.long_lora,
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
        amygdala.save()
        pfc.save()
        corpus.save()
        basal.save()
        cerebellum.save()


if __name__ == "__main__":
    main()
