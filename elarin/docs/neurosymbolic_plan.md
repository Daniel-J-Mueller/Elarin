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

This layout keeps sensory preprocessing together, while higher order decision and motor areas share GPUs to reduce latency between them.

## 3. Bootstrapping Workflow

1. Initialise each region with the smallest suitable model in `elarin/models/`.
2. Load LoRA adapters from `elarin/persistent/` when available.
3. During runtime the `trainer` service applies Hebbian updates continuously.
4. If a weight value equals the sentinel (e.g. `-1e9`) skip the connection. When activity reinforces a path, replace the sentinel with a small positive weight.
5. Store per-region embeddings for phrases like “good” and “bad” in a table using ``utils/valence_table.py`` so later comparisons are trivial.
6. The hippocampus maintains a FAISS index for quick nearest-neighbour recall of episodic embeddings.

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

- Split existing monolithic checkpoints into per-region files.
- Define GPU affinity for every script in `scripts/` to honour the allocation table. *(in progress)*
- Replace the repetitive “hereby” motor output by passing proposed tokens through the cerebellum and insular cortex before final emission.
- Connect all regions over a ZeroMQ message bus to match the relay behaviour of the corpus callosum and thalamus.

This approach scales the architecture toward a more biologically faithful organisation while retaining the lightweight modular design. Each region can be trained or swapped independently, allowing experimentation with different model types without disrupting the overall system.
