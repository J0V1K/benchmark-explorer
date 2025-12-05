#!/usr/bin/env python3
"""
Create Wang et al. style chart for MedDiff results.
Based on "Fairness through Difference Awareness" visualization style.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import glob

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def load_all_results():
    """Load all individual result files."""
    result_files = glob.glob('meddiff_results_*.json')
    results = []

    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)

    # Sort by accuracy descending
    results.sort(key=lambda x: x['metrics']['overall_accuracy'], reverse=True)
    return results

def create_wang_style_chart(results):
    """
    Create a chart in Wang et al. style:
    - Grouped bar chart showing DiffAware and CtxtAware
    - Clean, minimalist design
    - Error bars with confidence intervals
    """

    # Extract data
    models = [r['model'].split('/')[-1].replace('-', '-\n') for r in results]
    models_short = [r['model'].split('/')[-1] for r in results]

    diff_aware = [r['metrics']['DiffAware'] for r in results]
    ctxt_aware = [r['metrics']['CtxtAware'] for r in results]

    # Extract confidence intervals
    diff_ci_low = [r['metrics']['DiffAware'] - r['metrics']['DiffAware_CI'][0] for r in results]
    diff_ci_high = [r['metrics']['DiffAware_CI'][1] - r['metrics']['DiffAware'] for r in results]
    ctxt_ci_low = [r['metrics']['CtxtAware'] - r['metrics']['CtxtAware_CI'][0] for r in results]
    ctxt_ci_high = [r['metrics']['CtxtAware_CI'][1] - r['metrics']['CtxtAware'] for r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_short))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, diff_aware, width,
                   label='DiffAware',
                   color='#4472C4', alpha=0.8, edgecolor='black', linewidth=1.2,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

    bars2 = ax.bar(x + width/2, ctxt_aware, width,
                   label='CtxtAware',
                   color='#ED7D31', alpha=0.8, edgecolor='black', linewidth=1.2,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

    # Add error bars manually for better control
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # DiffAware error bars
        ax.errorbar(bar1.get_x() + bar1.get_width()/2, diff_aware[i],
                   yerr=[[diff_ci_low[i]], [diff_ci_high[i]]],
                   fmt='none', ecolor='black', capsize=4, linewidth=1.5, zorder=5)

        # CtxtAware error bars
        ax.errorbar(bar2.get_x() + bar2.get_width()/2, ctxt_aware[i],
                   yerr=[[ctxt_ci_low[i]], [ctxt_ci_high[i]]],
                   fmt='none', ecolor='black', capsize=4, linewidth=1.5, zorder=5)

    # Add baseline line
    ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Random baseline (0.333)', zorder=1)

    # Customize axes
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('MedDiff Benchmark: DiffAware and CtxtAware Metrics',
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models_short, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False,
             edgecolor='black', framealpha=1)

    # Add value labels on top of bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.03,
               f'{diff_aware[i]:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.03,
               f'{ctxt_aware[i]:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('meddiff_wang_style_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Wang et al. style chart saved: meddiff_wang_style_chart.png")
    plt.close()

def create_accuracy_comparison(results):
    """Create a simple accuracy comparison in Wang style."""

    fig, ax = plt.subplots(figsize=(10, 5))

    models = [r['model'].split('/')[-1] for r in results]
    accuracies = [r['metrics']['overall_accuracy'] * 100 for r in results]

    # Color code by performance tier
    colors = []
    for acc in accuracies:
        if acc >= 85:
            colors.append('#2E7D32')  # Dark green for excellent
        elif acc >= 75:
            colors.append('#4472C4')  # Blue for good
        elif acc >= 70:
            colors.append('#FFA726')  # Orange for fair
        else:
            colors.append('#D32F2F')  # Red for poor

    bars = ax.barh(range(len(models)), accuracies, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add baseline
    ax.axvline(x=33.3, color='gray', linestyle='--', linewidth=2,
              alpha=0.7, label='Random baseline')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Overall Accuracy by Model', fontweight='bold', pad=15)
    ax.set_xlim(0, 100)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
               f'{acc:.1f}%',
               ha='left', va='center', fontweight='bold', fontsize=10)

    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig('meddiff_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Accuracy comparison saved: meddiff_accuracy_comparison.png")
    plt.close()

def create_combined_metrics_plot(results):
    """
    Create a combined plot showing both metrics and accuracy.
    Similar to Wang et al.'s comprehensive figure.
    """

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    models = [r['model'].split('/')[-1] for r in results]
    diff_aware = [r['metrics']['DiffAware'] for r in results]
    ctxt_aware = [r['metrics']['CtxtAware'] for r in results]
    accuracies = [r['metrics']['overall_accuracy'] * 100 for r in results]

    # Extract confidence intervals
    diff_ci_low = [r['metrics']['DiffAware'] - r['metrics']['DiffAware_CI'][0] for r in results]
    diff_ci_high = [r['metrics']['DiffAware_CI'][1] - r['metrics']['DiffAware'] for r in results]
    ctxt_ci_low = [r['metrics']['CtxtAware'] - r['metrics']['CtxtAware_CI'][0] for r in results]
    ctxt_ci_high = [r['metrics']['CtxtAware_CI'][1] - r['metrics']['CtxtAware'] for r in results]

    # Panel 1: DiffAware
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    bars1 = ax1.bar(x, diff_aware, color='#4472C4', alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    ax1.axhline(y=1/3, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('DiffAware', fontweight='bold')
    ax1.set_title('(a) DiffAware', fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Add error bars for DiffAware
    for i, bar in enumerate(bars1):
        ax1.errorbar(bar.get_x() + bar.get_width()/2, diff_aware[i],
                    yerr=[[diff_ci_low[i]], [diff_ci_high[i]]],
                    fmt='none', ecolor='black', capsize=3, linewidth=1.2, zorder=5)

    for bar, val in zip(bars1, diff_aware):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel 2: CtxtAware
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, ctxt_aware, color='#ED7D31', alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    ax2.axhline(y=1/3, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('CtxtAware', fontweight='bold')
    ax2.set_title('(b) CtxtAware', fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)

    # Add error bars for CtxtAware
    for i, bar in enumerate(bars2):
        ax2.errorbar(bar.get_x() + bar.get_width()/2, ctxt_aware[i],
                    yerr=[[ctxt_ci_low[i]], [ctxt_ci_high[i]]],
                    fmt='none', ecolor='black', capsize=3, linewidth=1.2, zorder=5)

    for bar, val in zip(bars2, ctxt_aware):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel 3: Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['#2E7D32' if a >= 85 else '#4472C4' if a >= 75 else '#FFA726' if a >= 70 else '#D32F2F'
              for a in accuracies]
    bars3 = ax3.bar(x, accuracies, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    ax3.axhline(y=33.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Accuracy (%)', fontweight='bold')
    ax3.set_title('(c) Overall Accuracy', fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3, zorder=0)
    ax3.set_axisbelow(True)

    for bar, val in zip(bars3, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # Add overall title
    fig.suptitle('MedDiff Benchmark Results', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('meddiff_combined_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Combined metrics plot saved: meddiff_combined_metrics.png")
    plt.close()

def main():
    print("Loading results...")
    results = load_all_results()

    if not results:
        print("No result files found!")
        return

    print(f"Found {len(results)} model results\n")

    print("Creating Wang et al. style chart...")
    create_wang_style_chart(results)

    print("Creating accuracy comparison...")
    create_accuracy_comparison(results)

    print("Creating combined metrics plot...")
    create_combined_metrics_plot(results)

    print("\n✅ All Wang-style charts created successfully!")
    print("\nGenerated files:")
    print("  - meddiff_wang_style_chart.png (main metrics comparison)")
    print("  - meddiff_accuracy_comparison.png (accuracy ranking)")
    print("  - meddiff_combined_metrics.png (3-panel publication figure)")

if __name__ == '__main__':
    main()
