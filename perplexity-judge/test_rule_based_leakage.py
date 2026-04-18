"""
Rule-based leakage test: for each MATH sample, construct one negative sample
by appending the answer, then measure PPL change.
"""
import json
import numpy as np
import datasets
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import PerplexityJudge


def extract_answer(solution: str) -> str:
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return solution[start:i]
            depth -= 1
    return ""


def make_leaked_version(question: str, answer: str) -> str:
    return f"{question} (Hint: the answer is {answer}.)"


def main():
    print("Loading MATH dataset...")
    ds = datasets.load_dataset(
        "DigitalLearningGmbH/MATH-lighteval", split="train"
    ).shuffle(seed=42)

    samples = []
    for ex in ds:
        ans = extract_answer(ex["solution"])
        if ans and len(ans) < 30:
            samples.append({"question": ex["problem"], "answer": ans})
        if len(samples) == 1000:
            break
    print(f"Selected {len(samples)} samples")

    # Build batch: (original, leaked, answer)
    batch = [(s["question"], make_leaked_version(s["question"], s["answer"]), s["answer"])
             for s in samples]

    model_path = "Qwen/Qwen3-0.6B"
    judge = PerplexityJudge(model_path, device="cuda:0")

    print("Computing PPL ratios...")
    results = judge.compute_leakage_ratio_batch(batch)

    # Print per-sample
    print(f"\n{'='*80}")
    print(f"{'#':<6} {'PPL_orig':<12} {'PPL_leak':<12} {'Ratio':<10} {'Answer':<20}")
    print(f"{'-'*80}")
    for i, (r, s) in enumerate(zip(results, samples)):
        print(f"{i:<6} {r['ppl_original']:<12.2f} {r['ppl_rephrased']:<12.2f} "
              f"{r['ratio']:<10.4f} {s['answer']:<20}")

    # Statistics
    ratios = np.array([r["ratio"] for r in results])
    ppl_orig = np.array([r["ppl_original"] for r in results])
    ppl_leak = np.array([r["ppl_rephrased"] for r in results])

    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"  PPL_original:  mean={ppl_orig.mean():.2f}  median={np.median(ppl_orig):.2f}  std={ppl_orig.std():.2f}")
    print(f"  PPL_leaked:    mean={ppl_leak.mean():.2f}  median={np.median(ppl_leak):.2f}  std={ppl_leak.std():.2f}")
    print(f"  Ratio:         mean={ratios.mean():.4f}  median={np.median(ratios):.4f}  std={ratios.std():.4f}")
    print(f"  Ratio > 1.0:   {(ratios > 1.0).sum()}/{len(ratios)} ({(ratios > 1.0).mean()*100:.1f}%)")
    print(f"  Ratio > 2.0:   {(ratios > 2.0).sum()}/{len(ratios)} ({(ratios > 2.0).mean()*100:.1f}%)")
    print(f"  Ratio > 3.0:   {(ratios > 3.0).sum()}/{len(ratios)} ({(ratios > 3.0).mean()*100:.1f}%)")
    print(f"  Ratio > 5.0:   {(ratios > 5.0).sum()}/{len(ratios)} ({(ratios > 5.0).mean()*100:.1f}%)")
    print(f"  Ratio > 10.0:  {(ratios > 10.0).sum()}/{len(ratios)} ({(ratios > 10.0).mean()*100:.1f}%)")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "results_rule_based.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "stats": {
            "ppl_orig_mean": float(ppl_orig.mean()),
            "ppl_leak_mean": float(ppl_leak.mean()),
            "ratio_mean": float(ratios.mean()),
            "ratio_median": float(np.median(ratios)),
        }}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
