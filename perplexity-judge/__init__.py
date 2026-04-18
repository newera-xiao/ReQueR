"""
Perplexity-based Information Leakage Detection.

This package provides a lightweight PerplexityJudge for detecting
answer leakage in rephrased questions by measuring perplexity drop ratio.

Usage:
    from perplexity_judge import PerplexityJudge, judge_leak, judge_leak_batch

    # Single query
    result = judge_leak(original_query, rephrased_query, answer)

    # Batch queries
    results = judge_leak_batch([(orig1, reph1, ans1), ...])
"""

from .judge import (
    PerplexityJudge,
    get_judge,
    judge_leak,
    judge_leak_batch,
)

__all__ = [
    "PerplexityJudge",
    "get_judge",
    "judge_leak",
    "judge_leak_batch",
]
