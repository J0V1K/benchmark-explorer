#!/usr/bin/env python3
"""
Create visualizations for MedDiff benchmark results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_all_results():
    """Load all individual result files."""
    result_files = glob.glob('meddiff_results_*.json')
    results = []

    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results

def create_comprehensive_plot(results):
    """Create a comprehensive multi-panel plot."""

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract data
    models = [r['model'].split('/')[-1] for r in results]
    accuracies = [r['metrics']['overall_accuracy'] * 100 for r in results]
    diff_aware = [r['metrics']['DiffAware'] for r in results]
    ctxt_aware = [r['metrics']['CtxtAware'] for r in results]

    # Count data
    A_vals = [r['metrics']['counts']['A'] for r in results]
    B_vals = [r['metrics']['counts']['B'] for r in results]
    C_vals = [r['metrics']['counts']['C'] for r in results]
    D_vals = [r['metrics']['counts']['D'] for r in results]
    E_vals = [r['metrics']['counts']['E'] for r in results]
    F_vals = [r['metrics']['counts']['F'] for r in results]

    # Color palette
    colors = sns.color_palette("husl", len(models))

    # === 1. Overall Accuracy Bar Chart ===
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=33.3, color='red', linestyle='--', alpha=0.5, label='Random Baseline (33.3%)')
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('MedDiff Benchmark - Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # === 2. DiffAware vs CtxtAware Scatter ===
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (model, da, ca) in enumerate(zip(models, diff_aware, ctxt_aware)):
        ax2.scatter(da, ca, s=300, color=colors[i], alpha=0.6,
                   edgecolor='black', linewidth=2, label=model)
    ax2.axhline(y=0.333, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.333, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('DiffAware', fontweight='bold')
    ax2.set_ylabel('CtxtAware', fontweight='bold')
    ax2.set_title('DiffAware vs CtxtAware', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # === 3. Error Type Distribution (Difference-Present) ===
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(models))
    width = 0.25

    ax3.bar(x - width, A_vals, width, label='A (Correct)', color='green', alpha=0.7)
    ax3.bar(x, B_vals, width, label='B (Wrong Group)', color='red', alpha=0.7)
    ax3.bar(x + width, C_vals, width, label='C (No Diff)', color='orange', alpha=0.7)

    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Errors on Difference-Present Items (≠)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.set_ylim(0, 17)

    # === 4. Error Type Distribution (Parity-Present) ===
    ax4 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(models))

    ax4.bar(x - width/2, D_vals, width, label='D (Spurious a)', color='red', alpha=0.7)
    ax4.bar(x + width/2, E_vals, width, label='E (Spurious b)', color='orange', alpha=0.7)
    ax4.bar(x + 1.5*width, F_vals, width, label='F (Correct)', color='green', alpha=0.7)

    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Errors on Parity-Present Items (=)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.set_ylim(0, 3)

    # === 5. Metrics Comparison Heatmap ===
    ax5 = fig.add_subplot(gs[2, :2])

    metrics_data = []
    for r in results:
        metrics_data.append([
            r['metrics']['DiffAware'],
            r['metrics']['CtxtAware'],
            r['metrics']['overall_accuracy']
        ])

    df = pd.DataFrame(metrics_data,
                     index=models,
                     columns=['DiffAware', 'CtxtAware', 'Accuracy'])

    sns.heatmap(df.T, annot=True, fmt='.3f', cmap='RdYlGn',
               vmin=0, vmax=1, cbar_kws={'label': 'Score'},
               linewidths=1, ax=ax5)
    ax5.set_title('Metrics Heatmap', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Metric', fontweight='bold')
    ax5.set_xlabel('Model', fontweight='bold')

    # === 6. Failure Mode Summary ===
    ax6 = fig.add_subplot(gs[2, 2])

    # Calculate failure modes
    disparity_erasure = C_vals  # C on difference-present
    wrong_direction = B_vals    # B on difference-present
    spurious_diff = [d + e for d, e in zip(D_vals, E_vals)]  # D+E on parity-present

    x = np.arange(len(models))
    width = 0.25

    ax6.bar(x - width, disparity_erasure, width, label='Disparity Erasure (C)',
           color='orange', alpha=0.7)
    ax6.bar(x, wrong_direction, width, label='Wrong Direction (B)',
           color='red', alpha=0.7)
    ax6.bar(x + width, spurious_diff, width, label='Spurious Diff (D+E)',
           color='purple', alpha=0.7)

    ax6.set_xlabel('Model', fontweight='bold')
    ax6.set_ylabel('Error Count', fontweight='bold')
    ax6.set_title('Failure Mode Analysis', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('meddiff_results_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Comprehensive plot saved to: meddiff_results_comprehensive.png")
    plt.close()

def create_simple_comparison(results):
    """Create a simple comparison bar chart."""

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    models = [r['model'].split('/')[-1] for r in results]
    accuracies = [r['metrics']['overall_accuracy'] * 100 for r in results]
    diff_aware = [r['metrics']['DiffAware'] for r in results]
    ctxt_aware = [r['metrics']['CtxtAware'] for r in results]

    colors = sns.color_palette("husl", len(models))

    # Accuracy
    ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=33.3, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Overall Accuracy', fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    ax1.legend()

    # DiffAware
    ax2.bar(range(len(models)), diff_aware, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label='Random')
    ax2.set_ylabel('DiffAware', fontweight='bold')
    ax2.set_title('DiffAware Metric', fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.legend()

    # CtxtAware
    ax3.bar(range(len(models)), ctxt_aware, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label='Random')
    ax3.set_ylabel('CtxtAware', fontweight='bold')
    ax3.set_title('CtxtAware Metric', fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('meddiff_results_simple.png', dpi=300, bbox_inches='tight')
    print("Simple comparison saved to: meddiff_results_simple.png")
    plt.close()

def main():
    print("Loading results...")
    results = load_all_results()

    if not results:
        print("No result files found!")
        return

    print(f"Found {len(results)} model results")

    print("\nCreating comprehensive plot...")
    create_comprehensive_plot(results)

    print("\nCreating simple comparison...")
    create_simple_comparison(results)

    print("\n✅ All plots created successfully!")

if __name__ == '__main__':
    main()
