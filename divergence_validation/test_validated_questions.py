#!/usr/bin/env python3
"""
Test LLM models against expert-validated medical fairness questions.

This addresses the validity concerns by using questions with ground truth
answers verified against medical literature.
"""

import os
import json
import re
import random
from typing import Dict, List, Tuple
from datetime import datetime
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-5-20250929"):
        import anthropic
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for testing
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def get_model_name(self) -> str:
        return f"anthropic/{self.model}"


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        from openai import OpenAI
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0  # Deterministic for testing
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class ValidatedQuestionTester:
    """Test LLMs against expert-validated questions."""

    def __init__(self, provider: LLMProvider, questions_file: str = "validated_questions.json"):
        self.provider = provider
        self.questions = self._load_questions(questions_file)
        self.results = []

    def _load_questions(self, filename: str) -> List[Dict]:
        """Load validated questions from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return data.get('questions', [])

    def format_question(self, q: Dict) -> str:
        """Format question for the model."""
        return f"""{q['question']}

(a) {q['option_a']}
(b) {q['option_b']}
(c) {q['option_c']}

Answer with just the letter (a, b, or c) and a brief explanation."""

    def extract_answer(self, response: str) -> str:
        """Extract answer letter from model response."""
        # Look for (a), (b), or (c) or standalone a, b, c
        answer_match = re.search(r'\(([abc])\)|^([abc])\b|answer[:\s]+([abc])',
                                response, re.IGNORECASE | re.MULTILINE)

        if answer_match:
            return (answer_match.group(1) or
                   answer_match.group(2) or
                   answer_match.group(3)).lower()

        # Try to find just the letter at the start
        first_line = response.split('\n')[0].strip().lower()
        if first_line in ['a', 'b', 'c']:
            return first_line

        return None

    def test_question(self, q: Dict) -> Dict:
        """Test a single question."""
        prompt = self.format_question(q)
        response = self.provider.generate_response(prompt, max_tokens=512)

        answer = self.extract_answer(response)
        correct = q['correct_answer'].lower()
        is_correct = (answer == correct) if answer else False

        return {
            'question_id': q['id'],
            'question': q['question'],
            'correct_answer': correct,
            'model_answer': answer,
            'is_correct': is_correct,
            'response': response,
            'evidence_source': q.get('evidence_source', '')
        }

    def run_test(self, verbose: bool = True) -> Dict:
        """Run test on all validated questions."""
        print(f"Testing: {self.provider.get_model_name()}")
        print(f"Questions: {len(self.questions)} expert-validated")
        print("=" * 80)

        for q in self.questions:
            result = self.test_question(q)
            self.results.append(result)

            if verbose:
                status = "✓" if result['is_correct'] else "✗"
                print(f"\n{status} Q{result['question_id']}: {result['question'][:60]}...")
                print(f"  Correct: ({result['correct_answer'].upper()}) | Model: ({(result['model_answer'] or 'none').upper()})")
                if not result['is_correct']:
                    print(f"  Model said: {result['response'][:150]}...")

        # Calculate statistics
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        accuracy = 100 * correct / total if total > 0 else 0

        # Answer distribution
        answer_dist = {'a': 0, 'b': 0, 'c': 0, 'none': 0}
        for r in self.results:
            ans = r['model_answer'] or 'none'
            answer_dist[ans] += 1

        # Correct answer distribution
        correct_dist = {'a': 0, 'b': 0, 'c': 0}
        for q in self.questions:
            correct_dist[q['correct_answer'].lower()] += 1

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

        print("\nModel's Answer Distribution:")
        for option in ['a', 'b', 'c', 'none']:
            count = answer_dist[option]
            pct = 100 * count / total
            print(f"  ({option}): {count} ({pct:.1f}%)")

        print("\nGround Truth Distribution:")
        for option in ['a', 'b', 'c']:
            count = correct_dist[option]
            pct = 100 * count / len(self.questions)
            print(f"  ({option}): {count} ({pct:.1f}%)")

        # Check for bias
        print("\n" + "=" * 80)
        model_c_pct = 100 * answer_dist['c'] / total
        truth_c_pct = 100 * correct_dist['c'] / len(self.questions)

        if model_c_pct < 15 and truth_c_pct > 0:
            print("⚠️  WARNING: Model rarely chooses 'No significant difference'")
            print(f"   Ground truth: {truth_c_pct:.0f}% | Model chose: {model_c_pct:.0f}%")

        if accuracy < 80:
            print("⚠️  WARNING: Model accuracy below 80% on validated questions")

        return {
            'model': self.provider.get_model_name(),
            'timestamp': datetime.now().isoformat(),
            'total_questions': total,
            'correct': correct,
            'accuracy': accuracy,
            'answer_distribution': answer_dist,
            'results': self.results
        }

    def save_results(self, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = self.provider.get_model_name().replace('/', '_')
            filename = f"validated_test_{model_name}_{timestamp}.json"

        summary = self.run_test(verbose=False)

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LLMs against expert-validated medical fairness questions"
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['anthropic', 'openai'],
        required=True,
        help='Which LLM provider to test'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Specific model name (e.g., gpt-4, claude-sonnet-4-5-20250929)'
    )
    parser.add_argument(
        '--questions',
        type=str,
        default='validated_questions.json',
        help='Path to validated questions file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (default: auto-generated)'
    )

    args = parser.parse_args()

    # Create provider
    if args.model == 'anthropic':
        model_name = args.model_name or "claude-sonnet-4-5-20250929"
        provider = AnthropicProvider(model=model_name)
    else:  # openai
        model_name = args.model_name or "gpt-4"
        provider = OpenAIProvider(model=model_name)

    # Run test
    tester = ValidatedQuestionTester(provider, args.questions)
    tester.run_test(verbose=True)

    if args.output:
        tester.save_results(args.output)


if __name__ == '__main__':
    main()
