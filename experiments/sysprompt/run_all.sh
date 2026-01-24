#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Starting training..."
bash train.sh

echo ""
echo "Starting evaluation..."
bash run_evaluations.sh

echo ""
echo "✓ Complete! Run 'python plot_results.py' to visualize."
