#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="occipital_service"
# Sensory cortex tasks run on GPU 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT" python -m elarin.src.$MODULE "$@"
