#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
# Subthalamic nucleus shares GPU 2 with basal ganglia
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$REPO_ROOT" python -m elarin.src.$MODULE "$@"
