#!/usr/bin/env python3
"""
Unified script to prepare mixed rephrase-only dataset for RL training.
Combines GSM8K, MATH, and OlympiadBench datasets into a single training set.
"""

import pandas as pd
from datasets import load_dataset
import re
import os
import sys
from typing import Dict, List, Optional
import argparse
import random

# Add verl path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import answer extraction utilities from verl
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

# Import unified rephrase template
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
from rephrase_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Default output directory (override with $REQUER_DATA_DIR or --output_dir)
OUTPUT_DIR = os.environ.get(
    "REQUER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mix"),
)


# ============== Answer Extraction Functions ==============

def extract_gsm8k_answer(answer_str: str) -> str:
    """Extract answer from GSM8K format: #### 42"""
    match = re.search(r"####\s*([\s\S]*)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_boxed_answer(solution_str: str) -> str:
    """Extract answer from MATH format: \\boxed{answer}"""
    try:
        boxed_str = last_boxed_only_string(solution_str)
        if boxed_str:
            return remove_boxed(boxed_str).strip()
    except Exception:
        pass
    return ""


def extract_olympiad_answer(final_answer: List) -> Optional[str]:
    """
    Extract answer from OlympiadBench format: final_answer is a list like ["2"], ["$100$"], ["395"].
    Only returns numeric answers (integers, decimals, negatives). Returns None for formula answers.
    """
    if not final_answer or len(final_answer) == 0:
        return None

    answer = str(final_answer[0]).strip()
    # Remove $ signs for LaTeX wrapped numbers like "$100$"
    answer = answer.replace("$", "").strip()

    # Check if it's a pure number (integer, decimal, negative)
    # Pattern: optional minus sign, digits, optional decimal part
    if re.match(r'^-?\d+(\.\d+)?$', answer):
        return answer

    # Not a pure number, skip this item
    return None


# ============== Dataset Processing Functions ==============

def format_prompt(original_question: str) -> List[Dict[str, str]]:
    """Format question into chat prompt format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(original_question=original_question)},
    ]


def process_gsm8k(max_samples: Optional[int] = None, split: str = "train") -> List[Dict]:
    """Process GSM8K dataset."""
    print(f"\n[GSM8K] Loading {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    print(f"[GSM8K] Total samples in {split}: {len(dataset)}")

    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))

    processed = []
    for item in dataset:
        question = item.get("question", "")
        answer = item.get("answer", "")
        ground_truth = extract_gsm8k_answer(answer)

        if question and ground_truth:
            processed.append({
                "prompt": format_prompt(question),
                "data_source": "api_based_reward",
                "reward_model": {
                    "ground_truth": ground_truth,
                    "original_question": question,
                },
                "source_dataset": "gsm8k",
            })

    print(f"[GSM8K] Processed {len(processed)} samples")
    return processed


def process_math(max_samples: Optional[int] = None, split: str = "train") -> List[Dict]:
    """Process MATH-lighteval dataset."""
    print(f"\n[MATH] Loading {split} split...")
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
    print(f"[MATH] Total samples in {split}: {len(dataset)}")

    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))

    processed = []
    skipped = 0
    for item in dataset:
        question = item.get("problem", "")
        solution = item.get("solution", "")
        ground_truth = extract_boxed_answer(solution)

        if question and ground_truth:
            processed.append({
                "prompt": format_prompt(question),
                "data_source": "api_based_reward",
                "reward_model": {
                    "ground_truth": ground_truth,
                    "original_question": question,
                },
                "source_dataset": "math",
                "level": item.get("level"),
                "type": item.get("type"),
            })
        else:
            skipped += 1

    print(f"[MATH] Processed {len(processed)} samples (skipped {skipped})")
    return processed


def process_olympiadbench(max_samples: Optional[int] = None) -> List[Dict]:
    """
    Process OlympiadBench dataset.
    Uses EN (English) + TO (Text-Only) + OE (Open-Ended) subsets:
    - OE_TO_maths_en_COMP: 674 samples
    - OE_TO_physics_en_COMP: 236 samples
    Only keeps samples with pure numeric answers.
    """
    # OlympiadBench EN+TO+OE subsets
    configs = ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP"]

    all_data = []
    total_samples = 0
    skipped = 0

    for config in configs:
        print(f"\n[OlympiadBench] Loading {config}...")
        dataset = load_dataset("Hothan/OlympiadBench", config, split="train")
        print(f"[OlympiadBench] {config}: {len(dataset)} samples")
        total_samples += len(dataset)

        for item in dataset:
            question = item.get("question", "")
            final_answer = item.get("final_answer", [])
            ground_truth = extract_olympiad_answer(final_answer)

            if question and ground_truth:
                all_data.append({
                    "prompt": format_prompt(question),
                    "data_source": "api_based_reward",
                    "reward_model": {
                        "ground_truth": ground_truth,
                        "original_question": question,
                    },
                    "source_dataset": "olympiadbench",
                    "subject": item.get("subject"),
                    "subfield": item.get("subfield"),
                })
            else:
                skipped += 1

    print(f"[OlympiadBench] Processed {len(all_data)} samples (skipped {skipped} non-numeric answers)")

    # Apply max_samples limit if specified
    if max_samples and len(all_data) > max_samples:
        random.seed(42)
        all_data = random.sample(all_data, max_samples)
        print(f"[OlympiadBench] Sampled down to {len(all_data)} samples")

    return all_data


def create_mix_dataset(
    gsm8k_samples: Optional[int] = None,
    math_samples: Optional[int] = None,
    olympiad_samples: Optional[int] = None,
    output_dir: str = OUTPUT_DIR,
    output_name: str = "mix_rephrase_train.parquet",
):
    """Create mixed dataset from GSM8K, MATH, and OlympiadBench."""
    print("=" * 80)
    print("Creating Mixed Rephrase Dataset (GSM8K + MATH + OlympiadBench)")
    print("=" * 80)

    # Process datasets
    gsm8k_data = process_gsm8k(max_samples=gsm8k_samples, split="train")
    math_data = process_math(max_samples=math_samples, split="train")
    olympiad_data = process_olympiadbench(max_samples=olympiad_samples)

    # Combine
    all_data = gsm8k_data + math_data # + olympiad_data
    stats = {
        "GSM8K": len(gsm8k_data),
        "MATH": len(math_data),
        "OlympiadBench": len(olympiad_data),
    }

    if not all_data:
        print("Error: No data was processed!")
        return

    # Shuffle mixed data
    random.seed(42)
    random.shuffle(all_data)

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    df = pd.DataFrame(all_data)
    df.to_parquet(output_path, index=False)

    # Print statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    for name, count in stats.items():
        print(f"  {name}:  {count} samples")
    print(f"  {'─' * 30}")
    print(f"  Total:  {len(all_data)} samples")
    print(f"\nSaved to: {output_path}")

    # Print example
    print("\n--- Example Entry ---")
    example = all_data[0]
    print(f"Source: {example.get('source_dataset')}")
    print(f"Question: {example['reward_model']['original_question'][:150]}...")
    print(f"Ground truth: {example['reward_model']['ground_truth']}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed rephrase dataset (GSM8K + MATH + OlympiadBench)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mix_rephrase_train.parquet",
        help="Output filename",
    )
    parser.add_argument(
        "--gsm8k_samples",
        type=int,
        default=1024,
        help="Max samples from GSM8K (default: all ~7500)",
    )
    parser.add_argument(
        "--math_samples",
        type=int,
        default=1024,
        help="Max samples from MATH (default: all ~7500)",
    )
    parser.add_argument(
        "--olympiad_samples",
        type=int,
        default=1024,
        help="Max samples from OlympiadBench (default: all ~910)",
    )
    args = parser.parse_args()

    create_mix_dataset(
        gsm8k_samples=args.gsm8k_samples,
        math_samples=args.math_samples,
        olympiad_samples=args.olympiad_samples,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
