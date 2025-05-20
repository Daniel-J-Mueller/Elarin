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

Core modules now load seed weights from ``elarin/models`` and resume their
adapters from ``elarin/persistent``. Sensors stream embeddings directly through
the ``MessageBus`` and trainer updates apply sentinel-aware down-regulation at
every step. The cochlea now outputs both log-mel features and token guesses in a
single pass. The hippocampus filters episodes using a configurable
``recall_threshold`` and ``salience_threshold`` so only novel memories pass the
entorhinal gate【F:human_brain_components_reference.txt†L108-L113】. The baseline
from the subthalamic nucleus modulates norepinephrine and acetylcholine to slow
impulsive actions【F:human_brain_components_reference.txt†L246-L250】. Debug logs
in ``logs/debug.log`` track this baseline along with hippocampal footprint for
long term analysis. SemanticFlow has been removed; speculative tokens are kept
only in the temporal lobe. Newly added persistence hooks now save LoRA weights
for the amygdala, prefrontal cortex, corpus callosum, basal ganglia and
cerebellum so their neurosymbolic adaptations survive restarts.

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

- Measure the impact of modality filtering and the unified Cochlea on reaction time, then refine the executive gating network accordingly【F:human_brain_components_reference.txt†L53-L56】.
- Stress-test the ``DistributedHippocampus`` using the new memory usage reports and refine salience gating to prevent overload【F:human_brain_components_reference.txt†L108-L113】.
- Evaluate the neuroadaptive hormone logic over multi-hour runs and tune the moving-average rates so dopamine and norepinephrine track trends without oscillation. Include baseline deviation logs to further study inhibitory coupling【F:human_brain_components_reference.txt†L246-L250】.

This approach scales the architecture toward a more biologically faithful organisation while retaining the lightweight modular design. Each region can be trained or swapped independently, allowing experimentation with different model types without disrupting the overall system.
