#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
PYTHONPATH="$REPO_ROOT" python -m elarin.src.$MODULE "$@"
