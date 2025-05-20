# Revised Neurosymbolic Plan

This document outlines how we will evolve Elarin from the current single-model approach into a set of cooperating region models. Each region will roughly match the size and function of its biological counterpart as described in `human_brain_components_reference.txt`. Persistent checkpoints currently located in `elarin/persistent/` (e.g. `motor.pt`, `insula.pt`, `angular_gyrus.pt`) are treated as temporary bootstrap weights.

## 1. Core Principles

- Keep each brain region independent with its own model file.
- Use the base models in `elarin/models/` as seed weights for specialised LoRA adapters.
- When possible, new models should share a common format (PyTorch `.pt` or NumPy `.npy`) to avoid additional loaders.
- Reserve unusual sentinel values for untrained connections so that a region can skip processing them until reinforced.
- Align neighbouring regions on the same GPU to minimise cross-device copies.

## 2. Region Allocation

We divide the system into smaller modules. For each area we define the approximate parameter budget relative to the 16 billion cortical neurons.

| Region                      | Model Type/Size (approx.) | GPU |
|-----------------------------|---------------------------|-----|
| Sensory Cortex (visual/auditory) | Small CNN/Conv1D (~5M) | 0 |
| Thalamus & DMN              | Transformer (~15M)        | 1 |
| Hippocampus                 | FAISS index + MLP (~10M)  | 1 |
| Basal Ganglia               | Gating MLP (~5M)          | 2 |
| Cerebellum                  | MLP (~3M)                 | 2 |
| Prefrontal Cortex & OFC     | Transformer (~10M)        | 2 |
| Motor & Insular Cortex      | GPT‑2 half + projections (~60M) | 3 |

This layout keeps sensory preprocessing together, while higher order decision and motor areas share GPUs to reduce latency between them. The run scripts already enforce these assignments using ``CUDA_VISIBLE_DEVICES``.

## 3. Bootstrapping Status

The groundwork described in the previous revision has been implemented. Regions
load seed weights from ``elarin/models`` and resume adapters from
``elarin/persistent``. Sensors publish embeddings through the ``MessageBus`` and
trainer updates run continually with sentinel-aware down-regulation. The
hippocampus now ignores low-similarity memories according to a configurable
``recall_threshold`` and the Cochlea module consolidates audio embeddings and
transcription in one class.  The prefrontal cortex emits modality weights that
filter incoming sensations, while the subthalamic nucleus keeps a moving-average
baseline to better time inhibitory pulses.

## 4. Data Flow Updates

Outputs from one region remain high dimensional embeddings. For example:

```python
vision_feat = occipital_lobe(frame_emb)        # 128‑d
combined = dmn.route([vision_feat, audio_feat])# 512‑d
command = basal_ganglia.decide(combined)       # 32‑d
token_logits = motor_cortex.act(command)       # vocabulary × weights
```

Each connection mirrors the anatomical ordering described in the reference text. The thalamus filters sensor load before routing to cortical regions, the hippocampus indexes all fused vectors for later retrieval, and the cerebellum adjusts the motor plan before token emission. The corpus callosum service simply relays embeddings between hemispheric modules.

## 5. Next Steps

- Investigate distributed hippocampal training strategies to scale memory while
  preserving the entorhinal gating of salient events【F:human_brain_components_reference.txt†L108-L113】.
- Track how the subthalamic nucleus baseline drifts over long sessions and tie
  it to hormone modulators for more realistic inhibition control【F:human_brain_components_reference.txt†L246-L250】.
- Measure the impact of modality filtering and the unified Cochlea on reaction
  time, then refine the executive gating network accordingly【F:human_brain_components_reference.txt†L53-L56】.

This approach scales the architecture toward a more biologically faithful organisation while retaining the lightweight modular design. Each region can be trained or swapped independently, allowing experimentation with different model types without disrupting the overall system.
