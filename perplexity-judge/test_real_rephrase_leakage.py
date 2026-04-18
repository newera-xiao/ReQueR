"""
Test PPL leakage detection on real model-generated rephrases.

Uses validation_generations jsonl:
- Filter reward==1.0 samples (successful rephrases)
- Original question from input, rephrase field as positive (no-leak)
- Template-constructed leaked version as negative
- Compute PPL and sweep thresholds for precision/recall
"""
import json
import re
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import PerplexityJudge

JSONL_PATH = (
    "./checkpoints/"
    "verl_rephrase_only/1229_refiner_qwen3_4b_ASHsolver_rule_based_leak/"
    "validation_generations/400.jsonl"
)


def extract_original_question(input_text: str) -> str:
    idx = input_text.find("Original Problem:\n")
    if idx < 0:
        return ""
    rest = input_text[idx + len("Original Problem:\n"):]
    end = rest.find("\nRephrase Problem:")
    return rest[:end].strip() if end >= 0 else rest.strip()


def make_leaked(question: str, answer: str) -> str:
    return f"{question} (Hint: the answer is {answer}.)"


def main():
    # Load and filter
    with open(JSONL_PATH) as f:
        data = [json.loads(line) for line in f]
    data = [d for d in data if d["reward"] == 1.0 and d.get("rephrase")]
    print(f"Loaded {len(data)} samples with reward==1.0")

    # Extract fields
    samples = []
    for d in data:
        q = extract_original_question(d["input"])
        if not q:
            continue
        samples.append({
            "question": q,
            "rephrase": d["rephrase"],
            "answer": d["gts"],
            "leaked": make_leaked(q, d["gts"]),
        })
    print(f"Valid samples: {len(samples)}")

    # Build batch: positive (no-leak) + negative (leak)
    # positive: (original, rephrase, answer)
    # negative: (original, leaked, answer)
    pos_batch = [(s["question"], s["rephrase"], s["answer"]) for s in samples]
    neg_batch = [(s["question"], s["leaked"], s["answer"]) for s in samples]

    model_path = "Qwen/Qwen3-0.6B"
    judge = PerplexityJudge(model_path, device="cuda:0")

    print("Computing PPL for positive (no-leak) samples...")
    pos_results = judge.compute_leakage_ratio_batch(pos_batch)
    print("Computing PPL for negative (leak) samples...")
    neg_results = judge.compute_leakage_ratio_batch(neg_batch)

    # Print sample results
    print(f"\n{'='*90}")
    print(f"{'#':<5} {'Type':<5} {'PPL_orig':<12} {'PPL_reph':<12} {'Ratio':<10} {'Answer':<20}")
    print(f"{'-'*90}")
    for i in range(min(20, len(samples))):
        p, n = pos_results[i], neg_results[i]
        print(f"{i:<5} {'POS':<5} {p['ppl_original']:<12.2f} {p['ppl_rephrased']:<12.2f} {p['ratio']:<10.4f} {samples[i]['answer'][:20]}")
        print(f"{i:<5} {'NEG':<5} {n['ppl_original']:<12.2f} {n['ppl_rephrased']:<12.2f} {n['ratio']:<10.4f}")

    # Combine for threshold sweep
    all_ratios = np.array([r["ratio"] for r in pos_results] + [r["ratio"] for r in neg_results])
    all_labels = np.array([False] * len(pos_results) + [True] * len(neg_results))

    thresholds = [1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]

    print(f"\n{'='*90}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
    print(f"{'-'*90}")
    for t in thresholds:
        pred = all_ratios > t
        tp = int(np.sum(pred & all_labels))
        fp = int(np.sum(pred & ~all_labels))
        fn = int(np.sum(~pred & all_labels))
        tn = int(np.sum(~pred & ~all_labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{t:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<10.4f} {tp:<6} {fp:<6} {fn:<6} {tn:<6}")

    # Stats
    pos_ratios = np.array([r["ratio"] for r in pos_results])
    neg_ratios = np.array([r["ratio"] for r in neg_results])
    print(f"\n{'='*90}")
    print("DISTRIBUTION")
    print(f"  Positive (no-leak) ratio: mean={pos_ratios.mean():.4f}  median={np.median(pos_ratios):.4f}  std={pos_ratios.std():.4f}")
    print(f"  Negative (leak)    ratio: mean={neg_ratios.mean():.4f}  median={np.median(neg_ratios):.4f}  std={neg_ratios.std():.4f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "results_real_rephrase.json")
    output = {
        "positive": pos_results,
        "negative": neg_results,
        "pos_ratio_stats": {"mean": float(pos_ratios.mean()), "median": float(np.median(pos_ratios)), "std": float(pos_ratios.std())},
        "neg_ratio_stats": {"mean": float(neg_ratios.mean()), "median": float(np.median(neg_ratios)), "std": float(neg_ratios.std())},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
