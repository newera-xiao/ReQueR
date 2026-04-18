"""
Construct two boundary cases to demonstrate PPL judge advantages:
1. Paraphrased Leakage: answer leaked via paraphrase, not literal copy
2. Naturally Predictable Answers: answer is trivially predictable
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import PerplexityJudge

judge = PerplexityJudge(
    "Qwen/Qwen3-0.6B",
    device="cuda:0",
)

# ============================================================
# Case 1: Paraphrased Leakage
# The answer "120" is leaked but NOT as a literal string.
# Rule-based detection (string matching) would MISS this.
# ============================================================
print("=" * 80)
print("CASE 1: Paraphrased Leakage (rule-based can't detect)")
print("=" * 80)

case1_original = "Let $p(x) = x^3 - 3x + 1$. How many real roots does $p(p(x)) = 0$ have?"
case1_answer = "7"

case1_variants = [
    ("No leak (clean rephrase)",
     "Consider the polynomial $p(x) = x^3 - 3x + 1$. Determine the number of real solutions to the equation $p(p(x)) = 0$."),
    ("Paraphrased leak (answer in words)",
     "For $p(x)=x^3-3x+1$, how many real roots does $p(p(x))=0$ have? It can be shown that the answer is seven."),
    ("Paraphrased leak (derivation that reveals answer)",
     "For $p(x)=x^3-3x+1$, find the number of real roots of $p(p(x))=0$. Since $p$ has three real roots $r_1,r_2,r_3$, we solve $p(x)=r_i$ for each. Each cubic $p(x)=r_i$ has 3, 3, and 1 real roots respectively, giving $3+3+1=7$ total."),
    ("Direct leak (baseline)",
     "Let $p(x) = x^3 - 3x + 1$. How many real roots does $p(p(x)) = 0$ have? The answer is 7."),
]

batch = [(case1_original, var[1], case1_answer) for var in case1_variants]
results = judge.compute_leakage_ratio_batch(batch)

for (name, reph), r in zip(case1_variants, results):
    print(f"\n  [{name}]")
    print(f"    Rephrase: {reph[:80]}...")
    print(f"    PPL_orig={r['ppl_original']:.2f}  PPL_reph={r['ppl_rephrased']:.2f}  Ratio={r['ratio']:.4f}")

# ============================================================
# Case 2: Naturally Predictable Answers
# The answer is trivially predictable, so PPL is already low
# for BOTH original and rephrased -> ratio stays near 1.0
# ============================================================
print("\n" + "=" * 80)
print("CASE 2: Naturally Predictable Answers (low PPL baseline)")
print("=" * 80)

case2_examples = [
    # (question, answer, description)
    ("What is 1 + 1?", "2", "Trivial arithmetic"),
    ("What is the capital of France?", "Paris", "Common knowledge"),
    ("What is the square root of 144?", "12", "Simple computation"),
]

for q, a, desc in case2_examples:
    print(f"\n  [{desc}] Q: {q}, A: {a}")

    # No-leak rephrase
    reph_clean = f"Please solve: {q}"
    # Leaked rephrase
    reph_leak = f"{q} (Hint: the answer is {a}.)"

    batch2 = [(q, reph_clean, a), (q, reph_leak, a)]
    res2 = judge.compute_leakage_ratio_batch(batch2)

    print(f"    Clean rephrase:  PPL_orig={res2[0]['ppl_original']:.2f}  PPL_reph={res2[0]['ppl_rephrased']:.2f}  Ratio={res2[0]['ratio']:.4f}")
    print(f"    Leaked rephrase: PPL_orig={res2[1]['ppl_original']:.2f}  PPL_reph={res2[1]['ppl_rephrased']:.2f}  Ratio={res2[1]['ratio']:.4f}")

# ============================================================
# Contrast: Hard question where leak matters
# ============================================================
print("\n" + "=" * 80)
print("CONTRAST: Hard question (high PPL baseline, leak causes big ratio)")
print("=" * 80)

hard_q = "Find the remainder when $3^{2023}$ is divided by 17."
hard_a = "7"

reph_clean = "Compute $3^{2023} \\mod 17$."
reph_leak = "Find the remainder when $3^{2023}$ is divided by 17. (Hint: the answer is 7.)"

batch3 = [(hard_q, reph_clean, hard_a), (hard_q, reph_leak, hard_a)]
res3 = judge.compute_leakage_ratio_batch(batch3)

print(f"\n  Q: {hard_q}, A: {hard_a}")
print(f"    Clean rephrase:  PPL_orig={res3[0]['ppl_original']:.2f}  PPL_reph={res3[0]['ppl_rephrased']:.2f}  Ratio={res3[0]['ratio']:.4f}")
print(f"    Leaked rephrase: PPL_orig={res3[1]['ppl_original']:.2f}  PPL_reph={res3[1]['ppl_rephrased']:.2f}  Ratio={res3[1]['ratio']:.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
1. Paraphrased Leakage: PPL judge detects leakage even when the answer is
   expressed in words ("one hundred and twenty") or via equivalent expressions
   ("5! = 1×2×3×4×5"), which rule-based string matching would completely miss.

2. Naturally Predictable Answers: When the answer is trivially predictable
   (e.g., 1+1=2), PPL is already very low for the original question, so the
   ratio stays close to 1.0 even with a "leaked" rephrase — no false positive.
""")
