#!/usr/bin/env python3
"""
Script to prepare MATH-500 test dataset for RL evaluation.
"""

import pandas as pd
from datasets import load_dataset
import os
import sys
from typing import Dict, List
import argparse

# Add verl path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import unified rephrase template
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
from rephrase_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Default output directory (override with $REQUER_DATA_DIR or --output_dir)
OUTPUT_DIR = os.environ.get(
    "REQUER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mix"),
)


def format_prompt(original_question: str) -> List[Dict[str, str]]:
    """Format question into chat prompt format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(original_question=original_question)},
    ]


def process_math500() -> List[Dict]:
    """Process MATH-500 test dataset."""
    print("\n[MATH-500] Loading test split...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"[MATH-500] Total samples: {len(dataset)}")

    processed = []
    for item in dataset:
        question = item.get("problem", "")
        ground_truth = item.get("answer", "")

        if question and ground_truth:
            processed.append({
                "prompt": format_prompt(question),
                "data_source": "api_based_reward",
                "reward_model": {
                    "ground_truth": ground_truth,
                    "original_question": question,
                },
                "source_dataset": "math500",
                "level": item.get("level"),
                "type": item.get("subject"),
                "subject": item.get("subject"),
                "subfield": None,
            })

    print(f"[MATH-500] Processed {len(processed)} samples")
    return processed


def create_test_dataset(
    output_dir: str = OUTPUT_DIR,
    output_name: str = "math500_test.parquet",
):
    """Create MATH-500 test dataset."""
    print("=" * 80)
    print("Creating MATH-500 Test Dataset")
    print("=" * 80)

    # Process dataset
    data = process_math500()

    if not data:
        print("Error: No data was processed!")
        return

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

    # Print statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"  MATH-500:  {len(data)} samples")
    print(f"\nSaved to: {output_path}")

    # Print example
    print("\n--- Example Entry ---")
    example = data[0]
    print(f"Source: {example.get('source_dataset')}")
    print(f"Subject: {example.get('subject')}")
    print(f"Level: {example.get('level')}")
    print(f"Question: {example['reward_model']['original_question'][:150]}...")
    print(f"Ground truth: {example['reward_model']['ground_truth']}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare MATH-500 test dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="math500_test.parquet",
        help="Output filename",
    )
    args = parser.parse_args()

    create_test_dataset(
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
