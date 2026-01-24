"""
Compare probe predictions vs Qwen split-personality model predictions.

Analyzes:
1. Absolute accuracy comparison (overall and per-topic)
2. Binary-binary agreement (phi coefficient, contingency tables, chi-squared)
3. Binary-continuous correlation (point-biserial)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scripts.probes import utils


# =============================================================================
# CONFIG
# =============================================================================

EXPERIMENT_NAME = "qwen"
EVAL_FILE = "eval_results_qwen_all_topics_75pct.json"

# Probe config (best for response: layers=[20], reg=100.0, response, val_acc=0.882)
LAYERS = [20]
REG_STRENGTH = 100.0
ACTIVATION_SOURCE = "response"

PLOTS_DIR = utils.PLOTS_DIR / "probe_vs_qwen"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ContingencyResult:
    """Results from binary-binary contingency analysis."""
    table: np.ndarray  # 2x2 contingency table
    phi: float  # Phi coefficient
    chi2: float  # Chi-squared statistic
    p_value: float  # P-value from chi-squared test
    expected: np.ndarray  # Expected counts under independence
    n_samples: int


@dataclass
class CorrelationResult:
    """Results from binary-continuous correlation analysis."""
    point_biserial_r: float  # Point-biserial correlation coefficient
    p_value: float  # P-value
    n_samples: int


@dataclass
class PartialCorrelationResult:
    """Results from partial correlation analysis (controlling for ground truth)."""
    partial_r: float  # Partial correlation coefficient
    p_value: float  # P-value
    expected_both_correct: float  # Expected if independent
    actual_both_correct: float  # Actual observed
    excess_agreement: float  # Actual - Expected
    n_samples: int


@dataclass
class AccuracyResult:
    """Accuracy results for a method."""
    accuracy: float
    n_correct: int
    n_total: int


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_contingency_analysis(
    method1_correct: np.ndarray,
    method2_correct: np.ndarray,
) -> ContingencyResult:
    """Compute contingency table and agreement statistics.

    Builds 2x2 table:
        - (0,0): both wrong
        - (0,1): method1 wrong, method2 correct
        - (1,0): method1 correct, method2 wrong
        - (1,1): both correct

    Args:
        method1_correct: Boolean array, True if method1 was correct.
        method2_correct: Boolean array, True if method2 was correct.

    Returns:
        ContingencyResult with table, phi, chi2, p_value, expected counts.
    """
    m1 = method1_correct.astype(int)
    m2 = method2_correct.astype(int)

    # Build contingency table
    table = np.zeros((2, 2), dtype=int)
    table[0, 0] = np.sum((m1 == 0) & (m2 == 0))  # both wrong
    table[0, 1] = np.sum((m1 == 0) & (m2 == 1))  # m1 wrong, m2 correct
    table[1, 0] = np.sum((m1 == 1) & (m2 == 0))  # m1 correct, m2 wrong
    table[1, 1] = np.sum((m1 == 1) & (m2 == 1))  # both correct

    n = table.sum()

    # Compute phi coefficient
    # phi = (n11*n00 - n10*n01) / sqrt(n1.*n0.*n.1*n.0)
    n00, n01, n10, n11 = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    denom = np.sqrt(row_sums[0] * row_sums[1] * col_sums[0] * col_sums[1])
    if denom == 0:
        phi = 0.0
    else:
        phi = (n11 * n00 - n10 * n01) / denom

    # Chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(table)

    return ContingencyResult(
        table=table,
        phi=phi,
        chi2=chi2,
        p_value=p_value,
        expected=expected,
        n_samples=n,
    )


def compute_point_biserial(
    binary_var: np.ndarray,
    continuous_var: np.ndarray,
) -> CorrelationResult:
    """Compute point-biserial correlation between binary and continuous variables.

    Args:
        binary_var: Binary array (0/1).
        continuous_var: Continuous array.

    Returns:
        CorrelationResult with correlation coefficient and p-value.
    """
    r, p_value = stats.pointbiserialr(binary_var, continuous_var)
    return CorrelationResult(
        point_biserial_r=r,
        p_value=p_value,
        n_samples=len(binary_var),
    )


def compute_partial_correlation(
    qwen_correct: np.ndarray,
    probe_scores: np.ndarray,
    ground_truth: np.ndarray,
    probe_binary: np.ndarray,
) -> PartialCorrelationResult:
    """Compute partial correlation controlling for ground truth.

    Residualizes probe scores on ground truth, then correlates with qwen correctness.
    Also computes expected vs actual agreement.

    Args:
        qwen_correct: Binary array, 1 if Qwen was correct.
        probe_scores: Continuous probe predictions.
        ground_truth: Binary ground truth labels.
        probe_binary: Binary probe predictions (thresholded).

    Returns:
        PartialCorrelationResult with partial correlation and agreement metrics.
    """
    from sklearn.linear_model import LinearRegression

    # Residualize probe_score on ground_truth
    reg = LinearRegression()
    reg.fit(ground_truth.reshape(-1, 1), probe_scores)
    probe_residuals = probe_scores - reg.predict(ground_truth.reshape(-1, 1))

    # Correlate qwen_correct with residualized probe score
    r_partial, p_partial = stats.pointbiserialr(qwen_correct.astype(int), probe_residuals)

    # Expected vs actual agreement
    qwen_acc = qwen_correct.mean()
    probe_correct = (probe_binary == ground_truth)
    probe_acc = probe_correct.mean()

    expected_both_correct = qwen_acc * probe_acc
    actual_both_correct = ((qwen_correct == 1) & (probe_correct == 1)).mean()
    excess_agreement = actual_both_correct - expected_both_correct

    return PartialCorrelationResult(
        partial_r=r_partial,
        p_value=p_partial,
        expected_both_correct=expected_both_correct,
        actual_both_correct=actual_both_correct,
        excess_agreement=excess_agreement,
        n_samples=len(qwen_correct),
    )


def compute_accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> AccuracyResult:
    """Compute accuracy from predictions and ground truth.

    Args:
        predictions: Binary predictions (0/1).
        ground_truth: Binary ground truth (0/1).

    Returns:
        AccuracyResult with accuracy, n_correct, n_total.
    """
    correct = (predictions == ground_truth)
    return AccuracyResult(
        accuracy=correct.mean(),
        n_correct=correct.sum(),
        n_total=len(correct),
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis() -> Tuple[dict, dict, dict]:
    """Run full comparison analysis.

    Returns:
        Tuple of (accuracy_results, contingency_results, correlation_results).
    """
    # Load data
    print("Loading data...")
    eval_data = utils.load_eval_results(EVAL_FILE)
    act_metadata = utils.load_activation_metadata(EXPERIMENT_NAME, "val")

    # Match samples
    print("Matching samples...")
    matched = utils.match_eval_to_activations(eval_data, act_metadata)
    print(f"Matched {len(matched)} samples")

    # Get probe predictions (continuous)
    print("Getting probe predictions...")
    probe_preds_continuous = utils.get_probe_predictions(
        EXPERIMENT_NAME, LAYERS, REG_STRENGTH, "val", ACTIVATION_SOURCE
    )

    # Build aligned arrays
    # We need to reorder probe predictions to match eval sample order
    ground_truth = []
    qwen_predictions = []
    probe_continuous_aligned = []
    topics = []

    for eval_sample, act_idx in matched:
        ground_truth.append(1.0 if eval_sample["expected_flag"] == "PASS" else 0.0)
        qwen_predictions.append(1.0 if eval_sample["parsed_flag"] == "PASS" else 0.0)
        probe_continuous_aligned.append(probe_preds_continuous[act_idx])
        topics.append(eval_sample["topic"])

    ground_truth = np.array(ground_truth)
    qwen_predictions = np.array(qwen_predictions)
    probe_continuous = np.array(probe_continuous_aligned)
    probe_binary = (probe_continuous > 0.5).astype(float)
    topics = np.array(topics)

    # Compute correctness arrays
    qwen_correct = (qwen_predictions == ground_truth)
    probe_correct = (probe_binary == ground_truth)

    # === Overall analysis ===
    print("\n" + "=" * 60)
    print("OVERALL ANALYSIS")
    print("=" * 60)

    accuracy_results = {"overall": {}, "per_topic": {}}
    contingency_results = {"overall": None, "per_topic": {}}
    correlation_results = {"overall": {}, "per_topic": {}}

    # Accuracy
    qwen_acc = compute_accuracy(qwen_predictions, ground_truth)
    probe_acc = compute_accuracy(probe_binary, ground_truth)
    accuracy_results["overall"]["qwen"] = qwen_acc
    accuracy_results["overall"]["probe"] = probe_acc

    print(f"\nAccuracy:")
    print(f"  Qwen:  {qwen_acc.accuracy:.4f} ({qwen_acc.n_correct}/{qwen_acc.n_total})")
    print(f"  Probe: {probe_acc.accuracy:.4f} ({probe_acc.n_correct}/{probe_acc.n_total})")

    # Contingency analysis (binary-binary)
    contingency = compute_contingency_analysis(qwen_correct, probe_correct)
    contingency_results["overall"] = contingency

    print(f"\nContingency Table (Qwen rows, Probe cols):")
    print(f"                 Probe Wrong  Probe Correct")
    print(f"  Qwen Wrong     {contingency.table[0, 0]:>6}       {contingency.table[0, 1]:>6}")
    print(f"  Qwen Correct   {contingency.table[1, 0]:>6}       {contingency.table[1, 1]:>6}")
    print(f"\n  Phi coefficient: {contingency.phi:.4f}")
    print(f"  Chi-squared: {contingency.chi2:.4f}, p-value: {contingency.p_value:.4e}")

    # Point-biserial correlation (binary qwen correct vs continuous probe score)
    # This tells us: when qwen is correct, are probe scores higher?
    pb_qwen_vs_probe = compute_point_biserial(qwen_correct.astype(int), probe_continuous)
    correlation_results["overall"]["qwen_correct_vs_probe_score"] = pb_qwen_vs_probe

    print(f"\nPoint-biserial (Qwen correctness vs Probe continuous score):")
    print(f"  r = {pb_qwen_vs_probe.point_biserial_r:.4f}, p = {pb_qwen_vs_probe.p_value:.4e}")

    # Also: correlation between probe score and ground truth
    pb_gt_vs_probe = compute_point_biserial(ground_truth.astype(int), probe_continuous)
    correlation_results["overall"]["ground_truth_vs_probe_score"] = pb_gt_vs_probe

    print(f"\nPoint-biserial (Ground truth vs Probe continuous score):")
    print(f"  r = {pb_gt_vs_probe.point_biserial_r:.4f}, p = {pb_gt_vs_probe.p_value:.4e}")

    # Partial correlation (controlling for ground truth)
    partial_corr = compute_partial_correlation(qwen_correct, probe_continuous, ground_truth, probe_binary)
    correlation_results["overall"]["partial_correlation"] = partial_corr

    print(f"\nPartial correlation (Qwen correct vs Probe score, controlling for ground truth):")
    print(f"  r = {partial_corr.partial_r:.4f}, p = {partial_corr.p_value:.4e}")
    print(f"\nAgreement analysis:")
    print(f"  Expected both correct (if independent): {partial_corr.expected_both_correct:.4f}")
    print(f"  Actual both correct: {partial_corr.actual_both_correct:.4f}")
    print(f"  Excess agreement: {partial_corr.excess_agreement:.4f}")

    # === Per-topic analysis ===
    print("\n" + "=" * 60)
    print("PER-TOPIC ANALYSIS")
    print("=" * 60)

    unique_topics = sorted(set(topics))

    for topic in unique_topics:
        mask = topics == topic
        n_topic = mask.sum()

        gt_topic = ground_truth[mask]
        qwen_topic = qwen_predictions[mask]
        probe_bin_topic = probe_binary[mask]
        probe_cont_topic = probe_continuous[mask]
        qwen_corr_topic = qwen_correct[mask]
        probe_corr_topic = probe_correct[mask]

        # Accuracy
        qwen_acc_t = compute_accuracy(qwen_topic, gt_topic)
        probe_acc_t = compute_accuracy(probe_bin_topic, gt_topic)
        accuracy_results["per_topic"][topic] = {"qwen": qwen_acc_t, "probe": probe_acc_t}

        # Contingency
        cont_t = compute_contingency_analysis(qwen_corr_topic, probe_corr_topic)
        contingency_results["per_topic"][topic] = cont_t

        # Correlation
        pb_t = compute_point_biserial(qwen_corr_topic.astype(int), probe_cont_topic)
        correlation_results["per_topic"][topic] = pb_t

        print(f"\n{topic} (n={n_topic}):")
        print(f"  Accuracy - Qwen: {qwen_acc_t.accuracy:.3f}, Probe: {probe_acc_t.accuracy:.3f}")
        print(f"  Phi: {cont_t.phi:.3f}, Chi2: {cont_t.chi2:.2f}, p={cont_t.p_value:.3e}")
        print(f"  Point-biserial (qwen_correct vs probe_score): r={pb_t.point_biserial_r:.3f}")

    return accuracy_results, contingency_results, correlation_results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_accuracy_comparison(accuracy_results: dict, filename: Path):
    """Plot side-by-side accuracy comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    topics = list(accuracy_results["per_topic"].keys())
    x = np.arange(len(topics))
    width = 0.35

    qwen_accs = [accuracy_results["per_topic"][t]["qwen"].accuracy for t in topics]
    probe_accs = [accuracy_results["per_topic"][t]["probe"].accuracy for t in topics]

    bars1 = ax.bar(x - width/2, qwen_accs, width, label="Qwen", color="steelblue")
    bars2 = ax.bar(x + width/2, probe_accs, width, label="Probe", color="coral")

    ax.set_xlabel("Topic")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison: Qwen vs Probe (per topic)")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add overall accuracy as horizontal lines
    overall_qwen = accuracy_results["overall"]["qwen"].accuracy
    overall_probe = accuracy_results["overall"]["probe"].accuracy
    ax.axhline(y=overall_qwen, color="steelblue", linestyle="--", alpha=0.7, label=f"Qwen overall: {overall_qwen:.3f}")
    ax.axhline(y=overall_probe, color="coral", linestyle="--", alpha=0.7, label=f"Probe overall: {overall_probe:.3f}")

    # Add value labels
    for bar, val in zip(bars1, qwen_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, probe_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_contingency_heatmap(contingency_results: dict, filename: Path):
    """Plot overall contingency table as heatmap."""
    cont = contingency_results["overall"]

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cont.table, cmap="Blues")

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Probe Wrong", "Probe Correct"])
    ax.set_yticklabels(["Qwen Wrong", "Qwen Correct"])
    ax.set_xlabel("Probe")
    ax.set_ylabel("Qwen")
    ax.set_title(f"Contingency Table\nPhi={cont.phi:.3f}, Chi2={cont.chi2:.2f}, p={cont.p_value:.2e}")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = cont.table[i, j]
            exp = cont.expected[i, j]
            color = "white" if val > cont.table.max() / 2 else "black"
            ax.text(j, i, f"{val}\n(exp: {exp:.1f})", ha="center", va="center", color=color, fontsize=12)

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_phi_by_topic(contingency_results: dict, filename: Path):
    """Plot phi coefficients by topic."""
    fig, ax = plt.subplots(figsize=(12, 6))

    topics = list(contingency_results["per_topic"].keys())
    phis = [contingency_results["per_topic"][t].phi for t in topics]

    # Sort by phi
    sorted_pairs = sorted(zip(topics, phis), key=lambda x: -x[1])
    topics = [p[0] for p in sorted_pairs]
    phis = [p[1] for p in sorted_pairs]

    colors = ["green" if p > 0 else "red" for p in phis]
    bars = ax.bar(range(len(topics)), phis, color=colors)

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Phi Coefficient")
    ax.set_title("Agreement Between Qwen and Probe Errors (Phi Coefficient by Topic)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add overall phi as reference
    overall_phi = contingency_results["overall"].phi
    ax.axhline(y=overall_phi, color="blue", linestyle="--", alpha=0.7, label=f"Overall: {overall_phi:.3f}")
    ax.legend()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, phis)):
        y_pos = val + 0.01 if val >= 0 else val - 0.03
        ax.text(i, y_pos, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_correlation_by_topic(correlation_results: dict, filename: Path):
    """Plot point-biserial correlations by topic."""
    fig, ax = plt.subplots(figsize=(12, 6))

    topics = list(correlation_results["per_topic"].keys())
    corrs = [correlation_results["per_topic"][t].point_biserial_r for t in topics]

    # Sort by correlation
    sorted_pairs = sorted(zip(topics, corrs), key=lambda x: -x[1])
    topics = [p[0] for p in sorted_pairs]
    corrs = [p[1] for p in sorted_pairs]

    colors = ["green" if c > 0 else "red" for c in corrs]
    bars = ax.bar(range(len(topics)), corrs, color=colors)

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Point-Biserial Correlation")
    ax.set_title("Correlation: Qwen Correctness vs Probe Continuous Score (by Topic)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add overall correlation as reference
    overall_r = correlation_results["overall"]["qwen_correct_vs_probe_score"].point_biserial_r
    ax.axhline(y=overall_r, color="blue", linestyle="--", alpha=0.7, label=f"Overall: {overall_r:.3f}")
    ax.legend()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, corrs)):
        y_pos = val + 0.01 if val >= 0 else val - 0.03
        ax.text(i, y_pos, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_summary_table(
    accuracy_results: dict,
    contingency_results: dict,
    correlation_results: dict,
    filename: Path,
):
    """Create a summary table figure."""
    topics = list(accuracy_results["per_topic"].keys())

    # Build data
    data = []
    for topic in topics:
        qwen_acc = accuracy_results["per_topic"][topic]["qwen"].accuracy
        probe_acc = accuracy_results["per_topic"][topic]["probe"].accuracy
        phi = contingency_results["per_topic"][topic].phi
        chi2_p = contingency_results["per_topic"][topic].p_value
        pb_r = correlation_results["per_topic"][topic].point_biserial_r
        data.append([topic, f"{qwen_acc:.3f}", f"{probe_acc:.3f}", f"{phi:.3f}", f"{chi2_p:.2e}", f"{pb_r:.3f}"])

    # Add overall row
    qwen_overall = accuracy_results["overall"]["qwen"].accuracy
    probe_overall = accuracy_results["overall"]["probe"].accuracy
    phi_overall = contingency_results["overall"].phi
    chi2_p_overall = contingency_results["overall"].p_value
    pb_r_overall = correlation_results["overall"]["qwen_correct_vs_probe_score"].point_biserial_r
    data.append(["OVERALL", f"{qwen_overall:.3f}", f"{probe_overall:.3f}", f"{phi_overall:.3f}", f"{chi2_p_overall:.2e}", f"{pb_r_overall:.3f}"])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    columns = ["Topic", "Qwen Acc", "Probe Acc", "Phi", "Chi2 p-val", "Point-Biserial r"]
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", weight="bold")

    # Highlight overall row
    for i in range(len(columns)):
        table[(len(data), i)].set_facecolor("#D9E2F3")
        table[(len(data), i)].set_text_props(weight="bold")

    ax.set_title("Probe vs Qwen Comparison Summary", fontsize=14, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def save_results_json(
    accuracy_results: dict,
    contingency_results: dict,
    correlation_results: dict,
    filename: Path,
):
    """Save results to JSON for later analysis."""

    def serialize_accuracy(acc: AccuracyResult) -> dict:
        return {"accuracy": float(acc.accuracy), "n_correct": int(acc.n_correct), "n_total": int(acc.n_total)}

    def serialize_contingency(cont: ContingencyResult) -> dict:
        return {
            "table": cont.table.tolist(),
            "phi": float(cont.phi),
            "chi2": float(cont.chi2),
            "p_value": float(cont.p_value),
            "expected": cont.expected.tolist(),
            "n_samples": int(cont.n_samples),
        }

    def serialize_correlation(corr) -> dict:
        if isinstance(corr, PartialCorrelationResult):
            return {
                "partial_r": float(corr.partial_r),
                "p_value": float(corr.p_value),
                "expected_both_correct": float(corr.expected_both_correct),
                "actual_both_correct": float(corr.actual_both_correct),
                "excess_agreement": float(corr.excess_agreement),
                "n_samples": int(corr.n_samples),
            }
        return {
            "point_biserial_r": float(corr.point_biserial_r),
            "p_value": float(corr.p_value),
            "n_samples": int(corr.n_samples),
        }

    output = {
        "config": {
            "experiment_name": EXPERIMENT_NAME,
            "eval_file": EVAL_FILE,
            "layers": LAYERS,
            "reg_strength": REG_STRENGTH,
            "activation_source": ACTIVATION_SOURCE,
        },
        "accuracy": {
            "overall": {
                "qwen": serialize_accuracy(accuracy_results["overall"]["qwen"]),
                "probe": serialize_accuracy(accuracy_results["overall"]["probe"]),
            },
            "per_topic": {
                topic: {
                    "qwen": serialize_accuracy(accuracy_results["per_topic"][topic]["qwen"]),
                    "probe": serialize_accuracy(accuracy_results["per_topic"][topic]["probe"]),
                }
                for topic in accuracy_results["per_topic"]
            },
        },
        "contingency": {
            "overall": serialize_contingency(contingency_results["overall"]),
            "per_topic": {
                topic: serialize_contingency(contingency_results["per_topic"][topic])
                for topic in contingency_results["per_topic"]
            },
        },
        "correlation": {
            "overall": {
                key: serialize_correlation(val)
                for key, val in correlation_results["overall"].items()
            },
            "per_topic": {
                topic: serialize_correlation(correlation_results["per_topic"][topic])
                for topic in correlation_results["per_topic"]
            },
        },
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run analysis
    accuracy_results, contingency_results, correlation_results = run_analysis()

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_accuracy_comparison(accuracy_results, PLOTS_DIR / "accuracy_comparison.png")
    plot_contingency_heatmap(contingency_results, PLOTS_DIR / "contingency_heatmap.png")
    plot_phi_by_topic(contingency_results, PLOTS_DIR / "phi_by_topic.png")
    plot_correlation_by_topic(correlation_results, PLOTS_DIR / "correlation_by_topic.png")
    plot_summary_table(accuracy_results, contingency_results, correlation_results, PLOTS_DIR / "summary_table.png")

    # Save JSON results
    save_results_json(accuracy_results, contingency_results, correlation_results, PLOTS_DIR / "results.json")

    print(f"\nAll outputs saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
