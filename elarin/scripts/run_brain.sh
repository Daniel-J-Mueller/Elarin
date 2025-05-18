#!/bin/bash
# Launch the demo brain integration.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHONPATH="$REPO_ROOT" python -m elarin.src.brain "$@"
