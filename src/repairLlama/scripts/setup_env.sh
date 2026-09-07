#!/usr/bin/env bash
# Create a virtualenv and install repairllama-java in editable mode.
#
#   ./scripts/setup_env.sh            # runtime + dev extras (no torch)
#   ./scripts/setup_env.sh --full     # everything in requirements.txt
#
# torch has no wheels for Python 3.13+; use 3.10-3.12 for --full.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

if [[ ! -d "$VENV" ]]; then
  echo "creating virtualenv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

if [[ "${1:-}" == "--full" ]]; then
  echo "installing full stack from requirements.txt"
  pip install -r requirements.txt
  pip install -e .
else
  echo "installing runtime + dev extras (skip torch; pass --full for the model stack)"
  pip install -e ".[dev]"
fi

echo
echo "done. activate with:  source $VENV/bin/activate"
repairllama --version
