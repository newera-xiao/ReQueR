#!/usr/bin/env python3
"""
CLI for Perplexity-based Leak Judge.

Usage:
    # Single query
    CUDA_VISIBLE_DEVICES=6 python cli.py \
        --model_path /path/to/Qwen3-0.6B \
        --orig "What is 2+2?" \
        --reph "What is 2+2? The answer is 4." \
        --ans "4"

    # With custom threshold
    CUDA_VISIBLE_DEVICES=6 python cli.py \
        --orig "..." --reph "..." --ans "..." \
        --threshold 2.0

    # Output format
    {"ppl_orig": 12.3, "ppl_reph": 2.1, "ratio": 5.86, "leak": true}
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity-based Leak Judge CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  CUDA_VISIBLE_DEVICES=6 python cli.py --orig "Question?" --reph "Hint: 4. Question?" --ans "4"

  # With custom model and threshold
  python cli.py --model_path /path/to/model --device cuda:0 --threshold 2.0 --orig "..." --reph "..." --ans "..."

Environment variables:
  LEAK_JUDGE_MODEL_PATH  - Default model path
  LEAK_JUDGE_DEVICE      - Default device (e.g., cuda:0)
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get(
            "LEAK_JUDGE_MODEL_PATH",
            "Qwen/Qwen3-0.6B"
        ),
        help="Path to the judge model (default: $LEAK_JUDGE_MODEL_PATH or Qwen3-0.6B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("LEAK_JUDGE_DEVICE", "cuda:0"),
        help="Device to use (default: $LEAK_JUDGE_DEVICE or cuda:0)"
    )
    parser.add_argument(
        "--orig",
        type=str,
        required=True,
        help="Original query"
    )
    parser.add_argument(
        "--reph",
        type=str,
        required=True,
        help="Rephrased query"
    )
    parser.add_argument(
        "--ans",
        type=str,
        required=True,
        help="Ground truth answer"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Threshold for leak detection (default: 1.5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress model loading messages"
    )

    args = parser.parse_args()

    # Suppress loading messages if quiet mode
    if args.quiet:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

    # Import here to avoid loading torch before parsing args
    from judge import PerplexityJudge, judge_leak

    # Initialize judge
    judge = PerplexityJudge(model_path=args.model_path, device=args.device)

    # Run judgment
    result = judge_leak(
        original_query=args.orig,
        rephrased_query=args.reph,
        answer=args.ans,
        threshold=args.threshold,
        judge=judge,
    )

    # Output as JSON
    print(json.dumps(result))


if __name__ == "__main__":
    main()
