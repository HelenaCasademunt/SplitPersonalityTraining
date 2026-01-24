# Probe vs Qwen Comparison

Comparing linear probe predictions (on response activations) vs Qwen split-personality model predictions.

## Config
- **Probe**: Layer 20, reg=100, activation_source=response
- **Eval data**: `eval_results_qwen_all_topics_75pct.json` (720 samples)

## Results Summary

| Metric | Value | Significant? |
|--------|-------|--------------|
| Qwen accuracy | 93.9% | - |
| Probe accuracy | 88.2% | - |
| Phi coefficient | 0.356 | Yes (p=1.3e-20) |
| Partial correlation | -0.084 | Yes (p=0.024) |
| Excess agreement | 2.75% | - |

## Metrics Explained

### Phi Coefficient (0.356)
Measures agreement between Qwen and Probe errors beyond what you'd expect from their individual accuracies. Ranges from -1 to 1.
- **0.356** = moderate positive agreement
- They tend to get the same samples right/wrong more than chance

### Partial Correlation (-0.084)
Correlation between Qwen correctness and probe score, *after removing the effect of ground truth*. This isolates whether they share signal beyond both predicting the true label.
- **-0.084** = slight negative relationship
- After accounting for ground truth, when Qwen is correct the probe score is slightly *lower*
- Suggests they capture somewhat different aspects of the task

### Excess Agreement (2.75%)
- Expected both correct (if independent): 82.8%
- Actual both correct: 85.6%
- **Excess: 2.75%** = modest shared signal, but not redundant

## Plots

| Plot | Description |
|------|-------------|
| `accuracy_comparison.png` | Side-by-side accuracy per topic |
| `contingency_heatmap.png` | 2x2 table of joint errors |
| `phi_by_topic.png` | Phi coefficient per topic |
| `summary_table.png` | All metrics in one table |

## Interpretation

The probe and Qwen model both perform well but capture partially different signals:
1. High phi (0.356) means they agree on which samples are easy/hard
2. Near-zero partial correlation means knowing one method's confidence doesn't help predict the other's correctness beyond knowing the true label
3. The 2.75% excess agreement is modest - they share some signal but aren't redundant
