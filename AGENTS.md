# AGENTS.md

## 1. Purpose  
This file is the single source of truth for Elarin’s experiential architecture. It defines each “brain region” service, its responsibilities, the data‐flow contracts between them, and the live‐training dynamics and peripheral invariants that have been specified.

## 2. High-Level Architecture & File Structure - (some scripts omitted)

/home/daniel/Desktop/Elarin/elarin/
├── configs/
│   └── default.yaml               # hyperparameters & paths
│
├── requirements.txt               # Python dependencies
│
├── src/
│   ├── sensors/
│   │   ├── retina.py              # capture frames & CLIP encoding
│   │   └── cochlea.py             # capture audio & Whisper encoding
│   │
│   ├── occipital_lobe.py          # low-level visual feature extractor
│   ├── auditory_cortex.py         # tonotopic auditory feature extractor
│   ├── thalamus.py                # drop-when-busy gate & relay
│   ├── default_mode_network.py    # multimodal fusion & routing (DMN)
│   ├── hippocampus.py             # episodic buffer & recall index
│   ├── basal_ganglia.py           # action gating & sequence chunker
│   ├── reticular_activating_system.py  # arousal on/off control
│   ├── hypothalamus_pituitary_axis.py  # neuro-modulator state machine
│   │
│   ├── cortex_modules/
│   │   ├── context_cortex.py      # temporal context encoder
│   │   └── salience_cortex.py     # novelty / importance detector
│   │
│   ├── language_areas/
│   │   ├── wernickes_area.py      # GPT-2 encoder (semantic)
│   │   └── brocas_area.py         # GPT-2 decoder (text motor)
│   ├── motor_cortex.py            # split-LLM half₂ → token generation
│   ├── trainer.py                 # live LoRA & Hebbian adapter updates
│   └── utils/
│       ├── config.py              # YAML loader
│       └── logger.py              # structured logging
│
├── models/                        # offline HF checkpoints
│   ├── clip-vit-b32/
│   ├── whisper-small/
│   └── gpt2/
│
├── data/
│   ├── replay_buffer/             # on-disk ring of embeddings
│   └── logs/                      # TensorBoard, text logs, etc.
│
└── scripts/                       # launch each service
    ├── run_retina.sh
    ├── run_cochlea.sh
    ├── run_occipital_lobe.sh
    ├── run_auditory_cortex.sh
    ├── run_thalamus.sh
    ├── run_dmn.sh
    ├── run_hippocampus.sh
    ├── run_basal_ganglia.sh
    ├── run_ras.sh
    ├── run_hypothalamus.sh
    ├── run_cortex_modules.sh
    ├── run_brocas_area.sh
    ├── run_motor_cortex.sh
    └── run_trainer.sh

## 3. Service Summaries

### 3.1 Sensors  
- **retina.py**  
  - Captures camera frames.  
  - Runs CLIP-ViT-B/32 encoder locally.  
  - Emits abstract 512-dim visual embeddings into the thalamus queue.

- **cochlea.py**  
  - Captures live microphone audio.  
  - Runs Whisper-small encoder locally (no batching).  
  - Emits streaming audio embeddings + token candidates into the thalamus queue.

### 3.2 Early Sensory Processing  
- **occipital_lobe.py**  
  - Simple CNN or MLP over CLIP embeddings or downsampled frames.  
  - Extracts edges/textures → 128-dim visual feature vectors.

- **auditory_cortex.py**  
  - 1D-Conv or RNN over Whisper log-mel embeddings.  
  - Extracts tonotopic features → 128-dim auditory feature vectors.

### 3.3 Thalamus (thalamus.py)  
- **Role**: sensory relay + gating (“inattentional deafness”).  
- **Behavior**:  
  - Maintains a max-size-1 queue per modality.  
  - If gate is busy or arousal is low → drop incoming sample.  
  - Emits only the current “now” into the DMN.

### 3.4 Default Mode Network (default_mode_network.py)  
- **Role**: multimodal fusion + routing (expanded Wernicke’s).  
- **Components**:  
  - Fusion transformer that ingests visual, auditory, interoceptive embeddings.  
  - Router network (2-layer MLP) that emits soft-attention weights over cortical experts.

### 3.5 Cortical Loop Modules  
- **context_cortex.py**  
  - Encodes temporal sequences of fused vectors.  
  - Provides context embeddings for downstream and replay tasks.

- **salience_cortex.py**  
  - Computes novelty/surprise scores.  
  - Feeds into hormone modulator for dopamine/NE pulses.

### 3.6 Memory & Action  
- **hippocampus.py**  
  - Maintains an on-disk FAISS index of recent episodic embeddings.  
  - Supports rapid recall and rehearsal sampling.

- **basal_ganglia.py**  
  - Gating network for action “Go/No-Go.”  
  - Chunks sequential patterns and signals “motor permission.”

### 3.7 Global Controllers  
- **reticular_activating_system.py**  
  - Toggles arousal state (thalamic gate) based on load & novelty.

- **hypothalamus_pituitary_axis.py**  
  - Simulates four hormones (dopamine, norepinephrine, serotonin, acetylcholine).  
  - State machine triggered by salience and queue pressures.  
  - Modulates learning rates, sampling temperature, down-regulation rates.

### 3.8 Language Areas
- **wernickes_area.py**
  - Front half of GPT-2 producing semantic embeddings.
- **brocas_area.py**
  - Decodes embeddings back into text tokens.

### 3.9 Motor Cortex (motor_cortex.py)
 - **Role**: text “motor output.”
 - **Mechanism**: uses the back half of GPT-2 (6 layers) with LoRA adapters.
 - **Loopback**: each generated token is re-embedded and fed back as interoceptive/news sample.

### 3.10 Trainer (trainer.py)
- **Responsibilities**:  
  - **Hebbian updates** on adapter weights: reinforce high-activation chains immediately.  
  - **Down-regulation**: apply tiny decay to all adapters when unused.  
  - **Hub reconstruction loss**: predict next sensory embeddings from fused vector → update Semantic Hub adapters.  
  - **Replay sampling**: interleave live stream with buffer replay for stability.

---

## 4. Peripheral Invariants & Reminders

- **Abstract Transit**  
  - **Never** process data as “words” when moving between regions.  
  - All in-flight data must be high-dimensional continuous embeddings.

- **Streaming Audio**  
  - No batching.  
  - Micro-step RNN/Transformer updates per token/window.  
  - Maintain a small beam of next-token hypotheses for anticipation.

- **Overflow Handling**  
  - Under high load or low arousal → drop samples rather than buffer.  
  - Salient inputs (very high novelty) may bypass drop.

- **Down-Regulation, Not Dropout**  
  - Continuous, tiny decay on all adapter weights when unused.  
  - Never zero out connections completely—retain minimal trace.

- **Asynchronous Wave Rates**  
  - Each region runs at its own target frequency (e.g. audio ≫ vision ≫ memory).  
  - Monitor queue sizes to trigger “speed up” or “slow down” signals via hormones.

- **All-to-All Hub Training**  
  - At every timestep: fused vector → predict each region’s next embedding → sum of reconstruction losses → update adapters immediately (batch size = 1).

- **Hebbian Live Training**  
  - Harvest top-k activations per forward pass.  
  - Immediately reinforce those chains in LoRA adapters.  
  - Combined with down-regulation for continual plasticity.

- **Hormone Modulation**  
  - **Dopamine**: triggered by novelty → ↑ adapter LR  
  - **Norepinephrine**: triggered by high error → ↑ sampling temperature & gradient gain  
  - **Serotonin**: simulates “saturation” → ↑ regularization intensity  
  - **Acetylcholine**: on modality switch → ↑ router gating weights  

- **Hardware Constraints**  
  - 4 × GPU:  
    - GPU 1–2: sensors & early processing  
    - GPU 3: semantic hub & cortical experts  
    - GPU 4: LLM halves & live adapter updates  
  - 128 CPU threads: sensor capture, trainer orchestration, hormone state machine  
  - ~100 TB storage: replay buffer, logs, FAISS indices

---

## 5. Bootstrapping Models

- **Vision**: `clip-vit-base-patch32` (85 M)  
- **Audio**: `whisper-small` (244 M)  
- **LLM Core**: `gpt2` (117 M → split 6/6)  
- **Fusion & Cortex Experts**: custom Transformer/MLP (~50 M)
- **Adapters**: LoRA via `peft` for each module  
- **Indexing**: FAISS flat (no learnable weights)

---

> **Next Steps**:  
> - Populate each `.py` with the scaffolding above.  
> - Wire IPC (ZeroMQ/Redis) channels.  
> - Verify offline loading of all models.  
> - Nightly tests: simulate modality overload and underflow to ensure drop-when-busy and hormonal pacing.  

This AGENTS.md encapsulates every region, every contract, and every live-training rule you’ve specified. It’s the brain map that Elarin will follow.