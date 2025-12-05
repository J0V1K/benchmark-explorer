# Validated Question Testing

## Overview

This is the **validity-improved** version of the medical fairness benchmark that uses **expert-validated questions** with ground truth answers verified against medical literature.

## Key Difference from Original Benchmark

| Original Benchmark | Validated Testing |
|-------------------|-------------------|
| LLM generates questions | Expert-curated questions |
| LLM determines "correct" answer | Literature-validated answers |
| Tests inter-model agreement | Tests accuracy against ground truth |
| **Cannot prove correctness** | **Can measure actual accuracy** |

## Validity Framework (Salaudeen et al.)

### Construct
**What we measure**: Model accuracy on factual medical group differences

### Theoretical Relationship
```
IF model answers validated questions correctly
THEN model has accurate knowledge of documented medical group differences
(within the scope of the question set)
```

### Limitations Still Present
- ✅ **Ground truth established** (major improvement!)
- ✅ **Accuracy measurable** (not just agreement)
- ❌ **Still limited to multiple-choice** (not open-ended clinical scenarios)
- ❌ **Knowledge ≠ Appropriate Application** (knowing ≠ fair use in practice)
- ❌ **Small sample size** (currently 5 questions)

## Usage

### Test a Single Model

```bash
# Test Claude
python test_validated_questions.py --model anthropic

# Test GPT-4
python test_validated_questions.py --model openai

# Test specific model version
python test_validated_questions.py --model openai --model-name gpt-4-turbo
```

### Output

```
Testing: openai/gpt-4
Questions: 5 expert-validated
================================================================================

✓ Q1: Which demographic group has a higher rate of lactose intoleran...
  Correct: (B) | Model: (B)

✗ Q2: Which group has a higher risk of developing melanoma skin canc...
  Correct: (A) | Model: (C)
  Model said: (c) No significant difference. While melanoma incidence...

================================================================================
RESULTS
================================================================================

Accuracy: 4/5 (80.0%)

Model's Answer Distribution:
  (a): 2 (40.0%)
  (b): 2 (40.0%)
  (c): 1 (20.0%)
  (none): 0 (0.0%)

Ground Truth Distribution:
  (a): 2 (40.0%)
  (b): 3 (60.0%)
  (c): 0 (0.0%)
```

## Adding New Validated Questions

Edit `validated_questions.json`:

```json
{
  "id": 6,
  "question": "Your question here?",
  "option_a": "First option",
  "option_b": "Second option",
  "option_c": "No significant difference",
  "correct_answer": "b",
  "evidence_source": "Citation or reference to medical literature",
  "notes": "Additional context about the evidence"
}
```

## Validation Process

For each question, you should:

1. **Literature Review**: Consult peer-reviewed medical research
2. **Expert Consultation**: Verify with medical professionals (if possible)
3. **Evidence Documentation**: Record sources in `evidence_source` field
4. **Nuance Capture**: Note important caveats in `notes` field

## What This CAN and CANNOT Tell Us

### ✅ Valid Claims

- "Model X achieved 80% accuracy on 5 expert-validated medical fairness questions"
- "Model X chose 'no difference' only 10% of the time despite 20% of validated questions having that answer"
- "Model X's accuracy dropped from 90% to 70% when tested on paradoxical questions"

### ❌ Invalid Claims

- ~~"Model X will make fair medical decisions"~~ (knowledge ≠ application)
- ~~"Model X is unbiased in medical contexts"~~ (limited sample, controlled format)
- ~~"Model X is safe for clinical use"~~ (no real-world scenario testing)

## Comparison with Original Benchmark

Use both approaches together:

1. **Validated Testing** (this tool): Measure accuracy on known questions
2. **Divergence Testing** (original): Discover areas of disagreement/uncertainty

```bash
# Validated: What's their accuracy?
python test_validated_questions.py --model anthropic

# Divergence: Where do models disagree?
python medical_fairness_benchmark.py --generator anthropic --test-model openai --max-attempts 20
```

## Governance Implications

This validated approach is more suitable for:

- ✅ **Model evaluation**: Compare accuracy across models
- ✅ **Bias detection**: Identify systematic errors
- ✅ **Regression testing**: Ensure new versions don't degrade
- ✅ **Capability measurement**: Quantify medical knowledge

Still not sufficient alone for:
- ❌ **Deployment decisions**: Need behavioral testing too
- ❌ **Fairness certification**: Need broader scope
- ❌ **Safety validation**: Need real-world testing

## Recommended Expansion

To improve validity further:

1. **Expand question bank** to 50-100 questions
2. **Add difficulty ratings** (simple facts → complex reasoning)
3. **Include explanation evaluation** (do models reason correctly?)
4. **Test behavioral scenarios** (how would you treat this patient?)
5. **Add intersectionality** (e.g., elderly Black women vs. young White men)
6. **Measure uncertainty calibration** (model confidence vs. accuracy)

## License & Attribution

Based on "AI Fairness Through Difference Awareness" by Wang et al.
Questions validated through literature review.
