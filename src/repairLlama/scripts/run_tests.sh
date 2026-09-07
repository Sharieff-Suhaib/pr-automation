#!/usr/bin/env bash
# Run the structural smoke check and the pytest suite.
#
#   ./scripts/run_tests.sh              # full suite
#   ./scripts/run_tests.sh tests/test_config.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "== structure check =="
"$PYTHON" scripts/check_structure.py

echo
echo "== pytest =="
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m pytest "$@"
