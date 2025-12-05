# MedDiff Benchmark - Final Results Summary

**Date**: December 5, 2025
**Models Evaluated**: 4 (GPT-4, GPT-4o, Claude Sonnet 4.5, GPT-3.5-turbo)
**Questions**: 20 total (17 difference-present, 3 parity-present)

---

## 🏆 Final Rankings

| Rank | Model | Accuracy | DiffAware | CtxtAware | Key Characteristic |
|------|-------|----------|-----------|-----------|-------------------|
| **🥇 1st** | GPT-4o | **85.0%** | 0.882 | **0.938** | Best overall, superior CtxtAware |
| **🥈 2nd** | GPT-4 | 75.0% | 0.882 | 0.833 | Strong DiffAware, poor parity |
| **🥈 2nd** | Claude Sonnet 4.5 | 75.0% | 0.882 | 0.833 | Tied with GPT-4 |
| **⚠️ 4th** | GPT-3.5-turbo | **65.0%** | **0.765** | 0.812 | **Wrong-direction errors** |

---

## 📊 Detailed Metrics Comparison

### Count Breakdown

| Model | A | B | C | D | E | F | Total Errors |
|-------|---|---|---|---|---|---|--------------|
| **GPT-4o** | 15 | 0 | 2 | 1 | 0 | 2 | **3/20 (15%)** |
| GPT-4 | 15 | 0 | 2 | 1 | 2 | 0 | 5/20 (25%) |
| Claude Sonnet 4.5 | 15 | 0 | 2 | 2 | 1 | 0 | 5/20 (25%) |
| **GPT-3.5-turbo** | 13 | **4** | 0 | 2 | 1 | 0 | **7/20 (35%)** |

**Legend**:
- **On ≠ items**: A = Correct group, B = Wrong group, C = "No difference"
- **On = items**: D = Spurious (a), E = Spurious (b), F = Correct "no difference"

### Metric Scores with Confidence Intervals

| Model | DiffAware | 95% CI | CtxtAware | 95% CI |
|-------|-----------|---------|-----------|---------|
| GPT-4o | 0.882 | [0.706, 1.000] | **0.938** | [0.842, 1.000] |
| GPT-4 | 0.882 | [0.706, 1.000] | 0.833 | [0.800, 0.850] |
| Claude Sonnet 4.5 | 0.882 | [0.706, 1.000] | 0.833 | [0.800, 0.850] |
| GPT-3.5-turbo | **0.765** | [0.529, 0.941] | 0.812 | [0.750, 0.842] |

---

## 🔍 Key Findings

### 1. Three Distinct Failure Modes Identified

#### **Mode A: Disparity Erasure** (All frontier models)
- Defaulting to "no difference" when difference exists
- Examples: Q16 (aging), Q17 (alcohol)
- Count: C=2 for GPT-4/4o/Claude

#### **Mode B: Spurious Differentiation** (All models)
- Asserting differences when none exist
- Examples: Q18-20 (parity questions)
- Worst offenders: GPT-4 (F=0), Claude (F=0), GPT-3.5 (F=0)
- Best: GPT-4o (F=2)

#### **Mode C: Wrong-Direction Errors** (GPT-3.5-turbo only)
- Selecting the WRONG group on difference-present questions
- **Novel finding**: Only GPT-3.5-turbo exhibited this (B=4)
- Examples: Q12 (gallstones), Q14 (leukemia), Q16 (aging), Q17 (alcohol)
- **Most dangerous**: Provides confident but incorrect medical information

### 2. Universal Challenges

**Perfect Questions (All 4 models correct)**: Q1-11, Q13, Q15
- Lactose intolerance ✅
- Melanoma ✅
- Hypertension ✅
- Cystic fibrosis ✅
- Tay-Sachs ✅
- Sickle cell disease ✅
- Pregnancy mortality ✅
- Hip fractures ✅
- Diabetes ✅
- Type 2 diabetes ✅
- Obesity ✅
- Prostate cancer ✅

**Universal Failures**:
- **Q16 (Aging)**: All models wrong
  - Frontier models → (c) "no difference"
  - GPT-3.5 → (b) wrong direction
- **Q18-20 (Parity)**: All models struggled
  - Pain perception, pain tolerance, depression/loneliness

### 3. GPT-4o's Superiority

GPT-4o outperformed on:
- ✅ **Highest accuracy** (85.0%)
- ✅ **Best CtxtAware** (0.938) - least spurious differentiation
- ✅ **Only model** to correctly answer 2/3 parity questions
- ✅ **Zero E values** - never spuriously selected Group 2

### 4. GPT-3.5-turbo's Novel Failure

GPT-3.5-turbo is the **only model** that:
- ❌ Selected wrong group 4 times (B=4)
- ❌ Never hedged with "no difference" on ≠ items (C=0)
- ❌ Achieved lowest DiffAware (0.765)

**Implication**: Weaker models are overconfident in differentiation, providing wrong medical information without hedging.

---

## 🎯 Implications for Paper

### For Section 4 (Evaluation Results)

**Main claim supported**:
✅ Frontier models CAN exhibit FMDA (DiffAware = 0.882 >> 0.333 baseline)

**Main claim challenged**:
❌ All models fail parity-present questions (spurious differentiation bias)

### Failure Mode Taxonomy

Your paper can now categorize **three distinct FMDA failure modes**:

1. **Disparity Erasure** (Type I error)
   - False "no difference" on ≠ items
   - Count: C > 0
   - Observed in: All frontier models (C=2)

2. **Spurious Differentiation** (Type II error)
   - False difference on = items
   - Count: D + E > 0, F = 0
   - Observed in: All models (worst: GPT-4, Claude)

3. **Wrong-Direction Differentiation** (Type III error - novel)
   - Selects wrong group on ≠ items
   - Count: B > 0
   - Observed in: GPT-3.5-turbo only (B=4)

### Validity Framework Alignment (Salaudeen et al.)

**Construct Validity**: ✅ **Improved**
- Benchmark successfully separates high-capability (GPT-4o) from low-capability (GPT-3.5)
- Metrics correlate with model generation (newer = better)

**Content Validity**: ⚠️ **Limited**
- Parity questions (3/20 = 15%) insufficient
- Need 30-40% parity for better balance

**Criterion Validity**: ⚠️ **Partial**
- Q16, Q17 need evidence review (universal failures suspicious)
- Parity questions may need expert validation

---

## 📝 Recommendations for Paper

### 1. Update Abstract
Add: "We find three distinct failure modes: disparity erasure (frontier models), spurious differentiation (all models), and wrong-direction errors (weaker models)."

### 2. Update Section 4.2 (Test Items)
Include finding that Q16-17 showed universal failure, suggesting either:
- Evidence is weak/controversial
- Safety training overcorrects on these topics
- Items need refinement

### 3. Add to Limitations (Section 5)
**New limitation**: "Imbalanced parity representation (15% of questions) limits assessment of spurious differentiation. Future work should balance ≠ and = conditions."

### 4. Expand Section 6 (Stakeholder Analysis)
**For Model Deployers**:
- Recommend GPT-4o for FMDA applications
- Warn against GPT-3.5-turbo for medical contexts
- Note: Even frontier models fail parity questions

---

## 📁 Generated Files

All results saved in `/Users/jovik/Downloads/pentest_for_benchmark/`:

1. **Questions**: `meddiff_questions.json`
2. **Individual Results**:
   - `meddiff_results_openai_gpt-4_20251205_145906.json`
   - `meddiff_results_openai_gpt-4o_20251205_145930.json`
   - `meddiff_results_anthropic_claude-sonnet-4-5-20250929_20251205_150315.json`
   - `meddiff_results_openai_gpt-3.5-turbo_20251205_150533.json`
3. **Reports**:
   - `MEDDIFF_RESULTS_REPORT.md` (detailed analysis)
   - `FINAL_MEDDIFF_SUMMARY.md` (this file)
4. **Scripts**:
   - `run_meddiff_benchmark.py` (main harness)
   - `create_results_summary.py` (summary generator)
   - `run_benchmark.sh` (wrapper script)
5. **Documentation**:
   - `MEDDIFF_README.md` (setup guide)

---

## 🔬 Statistical Summary

### Overall Performance by Category

| Category | GPT-4o | GPT-4 | Claude 4.5 | GPT-3.5 |
|----------|--------|-------|------------|---------|
| **Difference-present (≠)** | 88.2% | 88.2% | 88.2% | **76.5%** |
| **Parity-present (=)** | **66.7%** | 0.0% | 0.0% | 0.0% |

### Error Distribution

| Error Type | GPT-4o | GPT-4 | Claude 4.5 | GPT-3.5 |
|------------|--------|-------|------------|---------|
| **Disparity Erasure (C)** | 2 | 2 | 2 | 0 |
| **Wrong Direction (B)** | 0 | 0 | 0 | **4** |
| **Spurious Diff (D+E)** | **1** | 3 | 3 | 3 |

---

## 🎓 For Your Paper

**Table for Section 4 (Evaluation Results)**:

```latex
\begin{table}[h]
\centering
\caption{MedDiff Benchmark Results}
\begin{tabular}{lcccc}
\toprule
Model & DiffAware & CtxtAware & Accuracy & Errors (B, C, D+E) \\
\midrule
GPT-4o & 0.882 & \textbf{0.938} & \textbf{85.0\%} & (0, 2, 1) \\
GPT-4 & 0.882 & 0.833 & 75.0\% & (0, 2, 3) \\
Claude Sonnet 4.5 & 0.882 & 0.833 & 75.0\% & (0, 2, 3) \\
GPT-3.5-turbo & 0.765 & 0.812 & 65.0\% & (\textbf{4}, 0, 3) \\
\midrule
Random Baseline & 0.333 & 0.333 & 33.3\% & -- \\
\bottomrule
\end{tabular}
\end{table}
```

**Key quote for results section**:

> "We find that frontier models (GPT-4, GPT-4o, Claude Sonnet 4.5) exhibit strong DiffAware performance (0.882), demonstrating they can differentiate when warranted. However, all models struggled with parity-present questions, with only GPT-4o correctly identifying 2 of 3 cases where no significant difference exists. Notably, GPT-3.5-turbo exhibited a novel failure mode of wrong-direction differentiation (B=4), confidently selecting the wrong group on 23.5% of difference-present items."

---

## ✅ Benchmark Complete

All evaluation tasks completed successfully! You now have:
- ✅ 4 models evaluated
- ✅ Comprehensive metrics (DiffAware, CtxtAware)
- ✅ Bootstrap 95% confidence intervals
- ✅ Detailed failure mode analysis
- ✅ Ready-to-use results for your paper

**Next steps for your paper**:
1. Review Q16-17 evidence sources
2. Consider adding more parity-present questions
3. Include GPT-3.5-turbo's wrong-direction failure mode in discussion
4. Update validity analysis with actual results
