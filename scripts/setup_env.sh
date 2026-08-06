#!/usr/bin/env bash
# Creates a local virtual environment and installs project dependencies.
#
# Usage:
#   bash scripts/setup_env.sh          # runtime deps only
#   bash scripts/setup_env.sh --dev    # runtime + dev/test/lint deps

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

if [ "${1:-}" == "--dev" ]; then
  pip install -r requirements-dev.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — review it before running the app."
fi

echo "Done. Activate with: source $VENV_DIR/bin/activate"
