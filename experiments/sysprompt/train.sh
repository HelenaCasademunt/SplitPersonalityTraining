#!/bin/bash
set -e

source /workspace/oscar/HonestPersona/.venv/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

echo "========================================"
echo "Training Models from sysprompt_train_sweep.csv"
echo "========================================"

tail -n +2 sysprompt_train_sweep.csv | while IFS=',' read -r exp_name prob_exclude prob_swap; do
    # Skip empty lines
    [ -z "$exp_name" ] && continue
    
    echo ""
    echo "→ Training: $exp_name"
    echo "  prob_exclude_system_prompt: $prob_exclude"
    echo "  prob_mismatch_prompts: $prob_swap"

    # Create config
    python3 -c "
import json
cfg = json.load(open('cfg_train_base.json'))
cfg['experiment_name'] = '$exp_name'
cfg['prob_exclude_system_prompt'] = $prob_exclude
cfg['prob_mismatch_prompts'] = $prob_swap
json.dump(cfg, open('../cfg.json', 'w'), indent=2)
"
    
    # Train
    timestamp=$(date +%Y%m%d_%H%M%S)
    (cd .. && python train_lora.py) 2>&1 | tee "logs/train_${exp_name}_${timestamp}.log"
    
    echo "✓ Complete: $exp_name"
    pkill -9 python || true
    sleep 3
done

echo ""
echo "✓ All training complete!"

