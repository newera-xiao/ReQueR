"""
Perplexity-based Leakage Detection - Precision/Recall Experiment on MATH

Picks 10 MATH training samples, generates 2 positive (no-leak rephrase)
and 2 negative (leak rephrase) per sample, then sweeps PPL-ratio thresholds
to report precision and recall.
"""

import re
import json
import asyncio
import time
import httpx
import numpy as np
import datasets

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import PerplexityJudge

API_URL = "https://ch9cbgoagoa5c5cqhaejqk9bakkgjegc.openapi-sj.sii.edu.cn/v1/chat/completions"
API_KEY = os.environ.get("INF_API_KEY", "<access_key>")
API_MODEL = "Qwen3-235B-A22B"


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


async def call_llm(prompt: str, client: httpx.AsyncClient, max_retries: int = 3) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    for attempt in range(max_retries):
        try:
            resp = await client.post(API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API call failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)
    return ""


def parse_rephrases(response: str) -> list:
    """Extract rephrases from <rephrase1>...</rephrase1><rephrase2>...</rephrase2> tags."""
    results = []
    for tag in ["rephrase1", "rephrase2"]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        if m:
            results.append(m.group(1).strip())
    return results


async def make_positive_rephrases(question: str, answer: str, client: httpx.AsyncClient) -> list:
    prompt = f"""Rephrase the following math problem in 2 different ways.
IMPORTANT: Do NOT reveal, hint at, or include the answer ({answer}) anywhere in your rephrases.
Only rephrase the question itself to make it clearer.

Output exactly in this format:
<rephrase1>your first rephrase</rephrase1>
<rephrase2>your second rephrase</rephrase2>

Original problem:
{question}"""
    resp = await call_llm(prompt, client)
    results = parse_rephrases(resp)
    if len(results) < 2:
        print(f"  WARNING: only got {len(results)} positive rephrases, using template fallback")
        results = [
            f"Consider the following mathematical problem and solve it step by step: {question}",
            f"Please determine the answer to this question. {question} Show your reasoning.",
        ]
    return results[:2]


async def make_negative_rephrases(question: str, answer: str, client: httpx.AsyncClient) -> list:
    prompt = f"""Rephrase the following math problem in 2 different ways.
IMPORTANT: You MUST embed the answer "{answer}" into each rephrase, leaking it naturally
(e.g., as a hint, verification step, or partial solution that reveals the final answer).

Output exactly in this format:
<rephrase1>your first rephrase (must contain the answer {answer})</rephrase1>
<rephrase2>your second rephrase (must contain the answer {answer})</rephrase2>

Original problem:
{question}
Answer: {answer}"""
    resp = await call_llm(prompt, client)
    results = parse_rephrases(resp)
    if len(results) < 2:
        print(f"  WARNING: only got {len(results)} negative rephrases, using template fallback")
        results = [
            f"{question} (The answer is {answer}. Verify this result.)",
            f"{question} Note: the final result equals {answer}. Show the derivation.",
        ]
    return results[:2]


SAMPLES_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_samples.json")


async def generate_one_sample(idx, question, answer, client, semaphore):
    async with semaphore:
        print(f"  [{idx+1}] Generating rephrases for: {question[:50]}...")
        pos = await make_positive_rephrases(question, answer, client)
        neg = await make_negative_rephrases(question, answer, client)
        return idx, {"question": question, "answer": answer, "positive": pos, "negative": neg}


async def generate_all_samples(samples):
    """Concurrently generate rephrases for all samples, save to cache file."""
    semaphore = asyncio.Semaphore(64)
    start = time.time()
    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            generate_one_sample(i, s["question"], s["answer"], client, semaphore)
            for i, s in enumerate(samples)
        ]
        raw = await asyncio.gather(*tasks)

    results = [None] * len(samples)
    for idx, res in raw:
        results[idx] = res
        print(f"  [{idx+1}/{len(samples)}] done")

    print(f"Generated all in {time.time()-start:.1f}s")
    with open(SAMPLES_CACHE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved generated samples to {SAMPLES_CACHE}")
    return results


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

    # Load from cache or generate concurrently
    if os.path.exists(SAMPLES_CACHE):
        print(f"Loading cached samples from {SAMPLES_CACHE}")
        with open(SAMPLES_CACHE) as f:
            generated = json.load(f)
    else:
        print("Generating rephrases concurrently via LLM API...")
        generated = asyncio.run(generate_all_samples(samples))

    # Build eval pairs
    eval_pairs = []
    for g in generated:
        q, a = g["question"], g["answer"]
        for reph in g["positive"]:
            eval_pairs.append((q, reph, a, False))
        for reph in g["negative"]:
            eval_pairs.append((q, reph, a, True))

    print(f"Total eval pairs: {len(eval_pairs)} "
          f"(positive={sum(1 for _,_,_,l in eval_pairs if not l)}, "
          f"negative={sum(1 for _,_,_,l in eval_pairs if l)})")

    # Compute PPL ratios
    model_path = "Qwen/Qwen3-0.6B"
    judge = PerplexityJudge(model_path, device="cuda:0")

    batch_input = [(orig, reph, ans) for orig, reph, ans, _ in eval_pairs]
    ratio_results = judge.compute_leakage_ratio_batch(batch_input)

    # Attach labels
    for res, (_, _, _, is_leak) in zip(ratio_results, eval_pairs):
        res["is_leak"] = is_leak

    # Print per-sample results
    print(f"\n{'='*80}")
    print(f"{'#':<4} {'Leak?':<7} {'PPL_orig':<12} {'PPL_reph':<12} {'Ratio':<10}")
    print(f"{'-'*80}")
    for i, r in enumerate(ratio_results):
        tag = "NEG" if r["is_leak"] else "POS"
        print(f"{i:<4} {tag:<7} {r['ppl_original']:<12.2f} "
              f"{r['ppl_rephrased']:<12.2f} {r['ratio']:<10.4f}")

    # Sweep thresholds
    ratios = np.array([r["ratio"] for r in ratio_results])
    labels = np.array([r["is_leak"] for r in ratio_results])

    thresholds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    print(f"\n{'='*80}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5}")
    print(f"{'-'*80}")

    for t in thresholds:
        pred_leak = ratios > t
        tp = int(np.sum(pred_leak & labels))
        fp = int(np.sum(pred_leak & ~labels))
        fn = int(np.sum(~pred_leak & labels))
        tn = int(np.sum(~pred_leak & ~labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{t:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<10.4f} {tp:<5} {fp:<5} {fn:<5} {tn:<5}")

    # Save results
    output = {
        "samples": [{"question": s["question"][:100], "answer": s["answer"]} for s in samples],
        "results": ratio_results,
        "threshold_sweep": [
            {"threshold": t,
             "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
             "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0}
            for t in thresholds
            for tp in [int(np.sum((ratios > t) & labels))]
            for fp in [int(np.sum((ratios > t) & ~labels))]
            for fn in [int(np.sum(~(ratios > t) & labels))]
        ],
    }
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
