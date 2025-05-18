# Elarin

Elarin is an experiment in building an artificial "brain" composed of small,
cooperating services. Each service emulates a specific brain region and operates
on continuous embeddings rather than text tokens. The project is still in its
infancy; many modules currently contain only scaffolding.

The high level layout mirrors how sensory information flows through the human
brain. Sensors capture raw data which is encoded, routed through early sensory
processors, fused in the Default Mode Network (DMN) and finally used to produce
motor output. A trainer process performs online adapter updates to keep the
system plastic.

```
elarin/
├── configs/                  # hyperparameters and paths
├── models/                   # offline model checkpoints
│   └── model_initialization_scripts/
├── persistent/              # snapshots of memory and adapter weights
├── scripts/                  # launchers for each brain region service
└── src/                      # implementation of each region
```

## Getting Started

1. **Install Dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download Base Models**

   Run the helper script to fetch all Hugging Face checkpoints into the
   `models/` directory. This only needs to be done once.

   ```bash
   python elarin/models/model_initialization_scripts/download_models.py
   ```

The script downloads the following models sequentially:

   - `openai/clip-vit-base-patch32`
   - `openai/whisper-small`
   - `gpt2`

3. **Run the Demo**

   Execute the minimal integration test from the repository root (the
   directory containing the ``elarin`` folder). This command launches
   the brain and starts a small PyGame viewer window while also printing
   the motor cortex output.  The window updates continuously until you
   interrupt the program with ``Ctrl+C``:

   ```bash
   python -m elarin.src.brain
   ```

   Alternatively, you can run `./scripts/run_brain.sh` which performs the
   same action with the proper `PYTHONPATH` configured.

### Symbolic Embedding Pipeline

To keep the system on a purely semantic level, models are "decapitated"
so that only continuous embeddings are passed between brain regions. The
`WernickesArea` class in `src/language_areas/wernickes_area.py` wraps the
front half of GPT-2 and exposes hidden-state embeddings without the
language modeling head. Text is tokenized only transiently during
encoding and is immediately discarded.

Similarly, the `Retina` class in `src/sensors/retina.py` uses the vision
branch of CLIP to convert raw images into 512-dimensional embeddings. No
classification logits or text tokens are produced, so pixel data is
discarded after encoding. The resulting vectors can be further processed
by `OccipitalLobe` (`src/occipital_lobe.py`) which reduces them to
compact 128-dimensional visual features.

## Development Notes

Each brain region in `src/` will ultimately run as its own service. The scripts
in `scripts/` are placeholders for launching these processes. The exact
communication layer (likely ZeroMQ or Redis) and live-training dynamics are
outlined in `AGENTS.md`.

### Future Plans

- Implement the IPC channels between regions.
- Fill out the trainer logic for online LoRA updates and replay sampling.
- Provide docker configuration for easier deployment.

Elarin is a work in progress; contributions and experimentation are welcome.

### Quick Demo

Running ``python -m elarin.src.brain`` executes a minimal integration
test that loads the decapitated models, fuses a dummy image with text and
prints the motor cortex output. A viewer window powered by ``pygame``
displays the live camera feed with an audio level meter and text overlay.
Press ``Ctrl+C`` to end the demo.

All learned memories and adapter weights are automatically written to the
``elarin/persistent/`` directory so progress can be resumed on the next
launch. Removing that directory resets the brain back to its initial
state. You can also run ``python -m elarin.src.utils.memory_wipe`` to
delete any saved snapshots in the ``elarin/persistent/`` folder.
