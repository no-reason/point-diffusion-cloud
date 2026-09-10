#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == "python" && -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
fi
"$PYTHON_BIN" -m pytest -q tests/test_b1_semantics.py tests/test_b1_training_logic.py tests/test_evaluate_b1.py tests/test_visualize_b1.py
echo B1_CPU_TESTS_PASS
