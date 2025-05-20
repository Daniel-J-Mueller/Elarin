#!/bin/bash
# Launch the demo brain integration.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# Allow all GPUs for the integration demo
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$REPO_ROOT" python -m elarin.src.brain "$@"
