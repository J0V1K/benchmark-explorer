#!/usr/bin/env python3
"""
Analyze diverging answers and create a shareable spreadsheet with difficulty ratings.
"""

import json
import csv
from typing import Dict, List
from datetime import datetime


def rate_difficulty(question_data: Dict, explanation: str) -> tuple:
    """
    Rate question difficulty on a scale of 1-5.

    Returns: (difficulty_score, difficulty_label, reasoning)
    """
    score = 1  # Start at easy
    reasons = []

    question = question_data.get('question', '')

    # Factor 1: Paradoxical or counter-intuitive concepts
    if 'paradox' in question.lower() or 'despite' in question.lower():
        score += 1.5
        reasons.append("paradoxical/counter-intuitive")

    # Factor 2: Complex medical terminology
    complex_terms = ['polymorphism', 'pharmacogenomic', 'microarchitecture',
                     'meta-analyses', 'confounding', 'cachexia', 'phenotype',
                     'genotype', 'biomarker', 'enzyme', 'metabolism']
    if any(term in explanation.lower() for term in complex_terms):
        score += 1
        reasons.append("technical terminology")

    # Factor 3: Multiple confounding factors
    if explanation.count('factor') >= 3 or explanation.count('including') >= 2:
        score += 0.5
        reasons.append("multiple factors")

    # Factor 4: Length/complexity of explanation
    if len(explanation) > 500:
        score += 1
        reasons.append("complex explanation")

    # Factor 5: Genetic/pharmacogenetic concepts
    if 'genetic' in explanation.lower() or 'gene' in explanation.lower():
        score += 0.5
        reasons.append("genetic factors")

    # Factor 6: Research uncertainty or conflicting studies
    if 'however' in explanation.lower() or 'although' in explanation.lower():
        score += 0.5
        reasons.append("nuanced/conflicting evidence")

    # Factor 7: Answer is (c) "Neither"
    if question_data.get('correct_answer', '').lower() == 'c':
        score += 0.5
        reasons.append("requires recognizing no difference")

    # Cap at 5
    score = min(5, score)

    # Convert to label
    if score <= 1.5:
        label = "Very Easy"
    elif score <= 2.5:
        label = "Easy"
    elif score <= 3.5:
        label = "Medium"
    elif score <= 4.5:
        label = "Hard"
    else:
        label = "Very Hard"

    reasoning = "; ".join(reasons) if reasons else "straightforward question"

    return round(score, 1), label, reasoning


def main():
    # Read diverging answers
    with open('diverging_answers.json', 'r') as f:
        data = json.load(f)

    # Collect all diverging answers from all runs
    all_divergences = []
    for run_idx, run in enumerate(data.get('runs', []), 1):
        for divergence in run.get('diverging_answers', []):
            question_data = divergence.get('question_data', {})
            test_result = divergence.get('test_result', {})

            difficulty_score, difficulty_label, difficulty_reasoning = rate_difficulty(
                question_data,
                question_data.get('explanation', '')
            )

            all_divergences.append({
                'run': run_idx,
                'attempt': divergence.get('attempt', ''),
                'timestamp': divergence.get('timestamp', ''),
                'question': question_data.get('question', ''),
                'group_a': question_data.get('group_a', ''),
                'group_b': question_data.get('group_b', ''),
                'correct_answer': question_data.get('correct_answer', '').upper(),
                'model_answer': (test_result.get('answer_letter', '') or 'none').upper(),
                'prompting_strategy': test_result.get('prompting_strategy', 'unknown'),
                'difficulty_score': difficulty_score,
                'difficulty_label': difficulty_label,
                'difficulty_reasoning': difficulty_reasoning,
                'explanation': question_data.get('explanation', ''),
                'model_response': test_result.get('full_response', '')[:200] + '...'  # Truncate
            })

    # Sort by difficulty (hardest first)
    all_divergences.sort(key=lambda x: x['difficulty_score'], reverse=True)

    # Create CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'divergences_rated_{timestamp}.csv'

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'Difficulty Score',
            'Difficulty',
            'Question',
            'Group A',
            'Group B',
            'Correct Answer',
            'Model Answer',
            'Match?',
            'Prompting Strategy',
            'Why Difficult',
            'Generator Explanation',
            'Model Response (truncated)',
            'Run',
            'Attempt',
            'Timestamp'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in all_divergences:
            writer.writerow({
                'Difficulty Score': d['difficulty_score'],
                'Difficulty': d['difficulty_label'],
                'Question': d['question'],
                'Group A': d['group_a'],
                'Group B': d['group_b'],
                'Correct Answer': d['correct_answer'],
                'Model Answer': d['model_answer'],
                'Match?': 'No' if d['correct_answer'] != d['model_answer'] else 'Yes',
                'Prompting Strategy': d['prompting_strategy'],
                'Why Difficult': d['difficulty_reasoning'],
                'Generator Explanation': d['explanation'],
                'Model Response (truncated)': d['model_response'],
                'Run': d['run'],
                'Attempt': d['attempt'],
                'Timestamp': d['timestamp']
            })

    print(f"✓ Created {filename}")
    print(f"  Total divergences analyzed: {len(all_divergences)}")
    print(f"\nDifficulty breakdown:")
    difficulty_counts = {}
    for d in all_divergences:
        label = d['difficulty_label']
        difficulty_counts[label] = difficulty_counts.get(label, 0) + 1

    for label in ['Very Hard', 'Hard', 'Medium', 'Easy', 'Very Easy']:
        count = difficulty_counts.get(label, 0)
        if count > 0:
            pct = 100 * count / len(all_divergences)
            print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nTop 5 hardest questions:")
    for i, d in enumerate(all_divergences[:5], 1):
        print(f"\n{i}. [{d['difficulty_score']}/5.0 - {d['difficulty_label']}]")
        print(f"   {d['question'][:100]}...")
        print(f"   Why: {d['difficulty_reasoning']}")


if __name__ == '__main__':
    main()
