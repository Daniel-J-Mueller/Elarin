# Neurosymbolic Expansion Plan

This document analyses which human brain functionalities are currently represented in the Elarin code base and outlines a path to extend the framework toward a more complete neurosymbolic architecture.

## 1. Functional Coverage

The reference `human_brain_components_reference.txt` describes many anatomical regions. Elarin already implements several core functions:

- **Retina & Cochlea** – sensory capture and encoding of visual and auditory input.
- **Occipital Lobe & Auditory Cortex** – early sensory feature extraction.
- **Thalamus** – central relay with drop-when-busy behaviour.
- **Default Mode Network (DMN)** – multimodal fusion and routing.
- **Context & Salience Cortex** – temporal context and novelty scoring.
- **Hippocampus** – episodic memory with replay capability.
- **Basal Ganglia** – action gating via dopaminergic modulation.
- **Reticular Activating System** and **Hypothalamus/Pituitary Axis** – arousal and hormone control.
- **Wernicke’s & Broca’s Areas** – semantic encoding and decoding using GPT‑2.
- **Motor Cortex & Insular Cortex** – text generation and interoceptive feedback.
- **Trainer** – simple Hebbian/adapter updates.

These cover vision, hearing, memory, motor output, and hormonal regulation. Some regions (e.g. DMN) are overextended relative to biology but serve the correct processing order.

## 2. Missing or Underrepresented Regions

Comparing against the reference file, the following areas are not yet modelled:

- **Cerebellum** – motor error correction and fine tuning.
- **Corpus Callosum** – cross-hemisphere communication layer.
- **Parietal and Somatosensory Cortex** – integration of touch and proprioception.
- **Amygdala** – fear/reward tagging of memories.
- **Prefrontal/Orbitofrontal Cortex** – executive decision making and inhibition control.
- **Pons/Medulla** – low level autonomic functions (may be omitted initially).

Where neuron counts are unknown they will be supplied later.

## 3. Bootstrapping Strategy

Existing modules rely on publicly available checkpoints (CLIP, Whisper, GPT-2). For new regions we can reuse smaller transformer/MLP components:

- **Cerebellum** – lightweight MLP receiving vision and motor embeddings to predict corrective adjustments.
- **Amygdala** – small network reading hippocampal outputs to assign valence scores.
- **Prefrontal Cortex** – transformer layer over DMN context to simulate planning/inhibition.

If no suitable pretrained model exists the module begins with random weights and learns online via the existing `Trainer` adapters.

## 4. Connection Plan

Communication between regions continues to use high dimensional embeddings. Each new module exposes a simple interface similar to existing classes:

```python
cereb = Cerebellum()
corrected = cereb.adjust(motor_emb, vision_feat)
```

Outputs are routed through the thalamus or DMN so the overall flow remains consistent. Specific channels (vision → cerebellum, hippocampus → amygdala, etc.) approximate the anatomical wiring while maintaining the semantic-layer abstraction.

## 5. Implementation Steps

- [x] **Introduce cerebellum module** for motor error correction. Integrate after motor cortex output so feedback is refined before being sent to the insular cortex.
- [x] **Stub corpus callosum** as a pass-through layer to allow future left/right specialisation.
- [x] **Extend memory schema** in `Hippocampus` to tag episodes with emotional valence from the amygdala.
- [x] **Add prefrontal cortex module** that modulates basal ganglia gating based on long-term goals.
- [ ] Continue bootstrapping using LoRA adapters for rapid online learning of all new modules.

This staged approach keeps the current processing order while gradually filling in the missing anatomical regions.
