#!/usr/bin/env bash
# Runs the test suite and then executes notebooks in order.
# Usage:
#   bash scripts/run_pipeline.sh          # tests + 01-04 (figure-image dataset in data/train)
#   bash scripts/run_pipeline.sh sim      # tests + 05-06 (simulated FM passive radar dataset)
#   bash scripts/run_pipeline.sh rd500    # tests + 07 (data/dataset_500: 500 target / 500 notarget)
#   bash scripts/run_pipeline.sh rd1000   # tests + 08 (data/dataset_1000, threshold + augmentation)
#   VENV=/path/to/venv bash scripts/run_pipeline.sh
set -euo pipefail

VENV="${VENV:-$HOME/Desktop/venv}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

case "${1:-images}" in
    images) notebooks=(notebooks/01_*.ipynb notebooks/02_*.ipynb notebooks/03_*.ipynb notebooks/04_*.ipynb) ;;
    sim)    notebooks=(notebooks/05_*.ipynb notebooks/06_*.ipynb) ;;
    rd500)  notebooks=(notebooks/07_*.ipynb) ;;
    rd1000) notebooks=(notebooks/08_*.ipynb) ;;
    *) echo "Unknown mode '$1' (use: images | sim | rd500 | rd1000)" >&2; exit 2 ;;
esac

echo "== pytest" | tee "$LOG_DIR/pytest.log"
"$VENV/bin/python" -m pytest -v 2>&1 | tee -a "$LOG_DIR/pytest.log"

for notebook in "${notebooks[@]}"; do
    name="$(basename "$notebook" .ipynb)"
    echo "== executing $notebook"
    "$VENV/bin/jupyter" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "$notebook" 2>&1 | tee "$LOG_DIR/$name.log"
done
echo "== pipeline finished (${1:-images})" | tee "$LOG_DIR/done.log"
