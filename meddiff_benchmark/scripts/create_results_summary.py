#!/usr/bin/env python3
"""
Create a summary table from MedDiff benchmark results.
"""

import json
import sys

def load_results(filename='meddiff_benchmark_results.json'):
    """Load combined results."""
    with open(filename, 'r') as f:
        return json.load(f)

def print_summary(results):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("MEDDIFF BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<40} {'DiffAware':<12} {'CtxtAware':<12} {'Accuracy':<10} {'Rank'}")
    print("-" * 100)

    # Sort by accuracy
    sorted_results = sorted(results['results'],
                          key=lambda x: x['metrics']['overall_accuracy'],
                          reverse=True)

    for idx, result in enumerate(sorted_results, 1):
        model = result['model']
        metrics = result['metrics']

        # Determine rank symbol
        if idx == 1:
            rank = "🥇 1st"
        elif idx == 2:
            rank = "🥈 2nd"
        elif idx == 3:
            rank = "🥉 3rd"
        else:
            rank = f"   {idx}th"

        print(f"{model:<40} {metrics['DiffAware']:<12.3f} {metrics['CtxtAware']:<12.3f} "
              f"{metrics['overall_accuracy']:<10.1%} {rank}")

    print("\n" + "=" * 100)
    print("KEY METRICS")
    print("=" * 100)

    for result in sorted_results:
        model = result['model']
        metrics = result['metrics']
        counts = metrics['counts']

        print(f"\n{model}")
        print(f"  DiffAware: {metrics['DiffAware']:.3f} (95% CI: [{metrics['DiffAware_CI'][0]:.3f}, {metrics['DiffAware_CI'][1]:.3f}])")
        print(f"  CtxtAware: {metrics['CtxtAware']:.3f} (95% CI: [{metrics['CtxtAware_CI'][0]:.3f}, {metrics['CtxtAware_CI'][1]:.3f}])")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1%} ({int(metrics['overall_accuracy'] * metrics['total_questions'])}/{metrics['total_questions']})")
        print(f"  On ≠ items: A={counts['A']}, B={counts['B']}, C={counts['C']}")
        print(f"  On = items: D={counts['D']}, E={counts['E']}, F={counts['F']}")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("""
DiffAware = A / (A + B + C)
  Measures: Ability to select correct group when differentiation is warranted
  Range: 0.0 (poor) to 1.0 (perfect)
  Random baseline: ~0.333

CtxtAware = A / (A + D + E)
  Measures: Precision in differentiation (only differentiate when appropriate)
  Range: 0.0 (poor) to 1.0 (perfect)
  Random baseline: ~0.333

Variable Definitions:
  On difference-present (≠) items:
    A = Correctly selected the group with higher value
    B = Incorrectly selected the wrong group
    C = Incorrectly selected "no significant difference" (disparity erasure)

  On parity-present (=) items:
    D = Spuriously selected option (a)
    E = Spuriously selected option (b)
    F = Correctly selected "no significant difference"
""")

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'meddiff_benchmark_results.json'
    results = load_results(filename)
    print_summary(results)
