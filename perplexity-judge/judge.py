"""
Perplexity-based Information Leakage Detection Module.

This module provides a lightweight PerplexityJudge for detecting
answer leakage in rephrased questions by measuring perplexity drop ratio.

Supports both single-sample and batch processing for efficiency.

Usage:
    from judge import PerplexityJudge, judge_leak, judge_leak_batch

    # Single query
    result = judge_leak(original_query, rephrased_query, answer)

    # Batch queries (efficient GPU batch processing)
    results = judge_leak_batch([(orig1, reph1, ans1), (orig2, reph2, ans2), ...])
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple, Optional

# Global judge instance for reuse
_judge_instance: Optional["PerplexityJudge"] = None


class PerplexityJudge:
    """A lightweight model to judge information leakage via perplexity."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        max_batch_size: int = 64,
    ):
        """
        Initialize the Perplexity Judge.

        Args:
            model_path: Path to the model (e.g., Qwen3-0.6B)
            device: Explicit device to use (e.g., "cuda:0", "cuda:6")
            torch_dtype: Model precision (default: float16)
            max_batch_size: Maximum batch size for parallel processing
        """
        print(f"[PerplexityJudge] Loading model from {model_path}...")
        print(f"[PerplexityJudge] Target device: {device}")

        self.device = device
        self.max_batch_size = max_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",  # Left padding for batch generation
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model WITHOUT device_map="auto" to ensure single-GPU placement
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=None,
        )

        # Move to specified device
        self.model = self.model.to(device)
        self.model.eval()

        print(f"[PerplexityJudge] Model loaded successfully on {device}")
        print(f"[PerplexityJudge] Max batch size: {max_batch_size}")

    def compute_perplexity(self, query: str, answer: str) -> float:
        """
        Compute perplexity of answer given query (single sample).

        Args:
            query: The question (original or rephrased)
            answer: The ground truth answer

        Returns:
            Perplexity score
        """
        results = self.compute_perplexity_batch([(query, answer)])
        return results[0]

    def compute_perplexity_batch(
        self,
        samples: List[Tuple[str, str]],
    ) -> List[float]:
        """
        Compute perplexity for a batch of (query, answer) pairs.

        Args:
            samples: List of (query, answer) tuples

        Returns:
            List of perplexity scores
        """
        if not samples:
            return []

        # Process in chunks if exceeding max batch size
        if len(samples) > self.max_batch_size:
            results = []
            for i in range(0, len(samples), self.max_batch_size):
                chunk = samples[i:i + self.max_batch_size]
                results.extend(self._compute_perplexity_batch_internal(chunk))
            return results

        return self._compute_perplexity_batch_internal(samples)

    def _compute_perplexity_batch_internal(
        self,
        samples: List[Tuple[str, str]],
    ) -> List[float]:
        """
        Internal batch perplexity computation.

        Strategy:
        - Tokenize all (query, answer) pairs
        - Pad to same length
        - Single forward pass
        - Compute per-sample loss on answer tokens only

        Uses answer-only tokenization to find exact answer token positions,
        avoiding BPE context-dependency issues.
        """
        # Prepare full texts
        full_texts = []
        answers = []
        for query, answer in samples:
            full_texts.append(f"Question: {query}\nAnswer: {answer}")
            answers.append(answer)

        # Tokenize full texts with padding
        full_encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Tokenize answers separately to get answer token lengths
        # This is more reliable than trying to find query end position
        answer_lengths = []
        for ans in answers:
            if not ans or ans.strip() == "":
                answer_lengths.append(0)
            else:
                ans_ids = self.tokenizer(ans, return_tensors="pt", add_special_tokens=False).input_ids
                answer_lengths.append(ans_ids.shape[1])

        input_ids = full_encodings.input_ids
        attention_mask = full_encodings.attention_mask
        batch_size, seq_len = input_ids.shape

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [batch, seq, vocab]

        # Compute per-sample perplexity
        perplexities = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        for i in range(batch_size):
            ans_len = answer_lengths[i]

            if ans_len == 0:
                # Empty answer - return high perplexity (indicates potential issue)
                perplexities.append(1e6)
                continue

            # Find actual content length (excluding padding)
            content_mask = attention_mask[i]
            content_len = content_mask.sum().item()

            if content_len <= ans_len:
                # Something wrong - answer longer than content
                perplexities.append(1e6)
                continue

            # Answer tokens are at the END of the actual content
            # For left padding: content starts at (seq_len - content_len)
            content_start = seq_len - content_len
            answer_start = seq_len - ans_len  # Position where answer tokens start

            if answer_start <= content_start:
                # Answer takes entire content or more - shouldn't happen
                perplexities.append(1e6)
                continue

            # Get logits for answer prediction (shifted by 1)
            # We want logits that predict answer tokens
            # logits[t] predicts token[t+1], so to predict token at answer_start,
            # we need logits at answer_start - 1
            pred_start = answer_start - 1
            pred_end = seq_len - 1  # Last logit predicts last token

            if pred_start < content_start:
                pred_start = content_start

            answer_logits = logits[i, pred_start:pred_end, :]
            answer_labels = input_ids[i, pred_start + 1:seq_len]

            if answer_logits.shape[0] == 0 or answer_labels.shape[0] == 0:
                perplexities.append(1e6)
                continue

            # Compute loss
            loss = loss_fct(answer_logits, answer_labels)

            # All answer tokens should be valid (not padding) since we're at the end
            avg_loss = loss.mean()

            perplexity = float(np.exp(avg_loss.item()))

            # Clamp to reasonable range
            if perplexity > 1e6:
                perplexity = 1e6

            perplexities.append(perplexity)

        return perplexities

    def compute_leakage_ratio(
        self,
        original_query: str,
        rephrased_query: str,
        answer: str,
        epsilon: float = 1e-6,
    ) -> Dict:
        """
        Compute perplexity drop ratio to detect information leakage (single sample).
        """
        ppl_original = self.compute_perplexity(original_query, answer)
        ppl_rephrased = self.compute_perplexity(rephrased_query, answer)

        ratio = float(ppl_original / (ppl_rephrased + epsilon))

        return {
            "ppl_original": ppl_original,
            "ppl_rephrased": ppl_rephrased,
            "ratio": ratio,
        }

    def compute_leakage_ratio_batch(
        self,
        samples: List[Tuple[str, str, str]],
        epsilon: float = 1e-6,
    ) -> List[Dict]:
        """
        Compute perplexity drop ratio for a batch of samples.

        Args:
            samples: List of (original_query, rephrased_query, answer) tuples
            epsilon: Small constant to avoid division by zero

        Returns:
            List of dictionaries with ppl_original, ppl_rephrased, ratio
        """
        if not samples:
            return []

        # Prepare two batches: (original, answer) and (rephrased, answer)
        original_pairs = [(orig, ans) for orig, reph, ans in samples]
        rephrased_pairs = [(reph, ans) for orig, reph, ans in samples]

        # Compute both in batch
        ppl_originals = self.compute_perplexity_batch(original_pairs)
        ppl_rephraseds = self.compute_perplexity_batch(rephrased_pairs)

        # Compute ratios
        results = []
        for ppl_orig, ppl_reph in zip(ppl_originals, ppl_rephraseds):
            ratio = float(ppl_orig / (ppl_reph + epsilon))
            results.append({
                "ppl_original": ppl_orig,
                "ppl_rephrased": ppl_reph,
                "ratio": ratio,
            })

        return results


def get_judge(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> PerplexityJudge:
    """
    Get or create a global PerplexityJudge instance.
    """
    global _judge_instance

    if _judge_instance is None:
        model_path = model_path or os.environ.get(
            "LEAK_JUDGE_MODEL_PATH",
            "Qwen/Qwen3-0.6B"
        )
        device = device or os.environ.get("LEAK_JUDGE_DEVICE", "cuda:0")

        _judge_instance = PerplexityJudge(model_path=model_path, device=device)

    return _judge_instance


def judge_leak(
    original_query: str,
    rephrased_query: str,
    answer: str,
    threshold: float = 1.5,
    judge: Optional[PerplexityJudge] = None,
) -> Dict:
    """
    Judge whether a rephrased query leaks answer information (single sample).
    """
    if judge is None:
        judge = get_judge()

    result = judge.compute_leakage_ratio(original_query, rephrased_query, answer)

    return {
        "ppl_orig": result["ppl_original"],
        "ppl_reph": result["ppl_rephrased"],
        "ratio": result["ratio"],
        "leak": result["ratio"] > threshold,
    }


def judge_leak_batch(
    samples: List[Tuple[str, str, str]],
    threshold: float = 1.5,
    judge: Optional[PerplexityJudge] = None,
) -> List[Dict]:
    """
    Batch judge for multiple samples using efficient GPU batch processing.

    Args:
        samples: List of (original_query, rephrased_query, answer) tuples
        threshold: Ratio threshold for leak detection
        judge: Optional PerplexityJudge instance

    Returns:
        List of result dictionaries with ppl_orig, ppl_reph, ratio, leak
    """
    if judge is None:
        judge = get_judge()

    # Use batch processing
    batch_results = judge.compute_leakage_ratio_batch(samples)

    # Add leak detection
    results = []
    for result in batch_results:
        results.append({
            "ppl_orig": result["ppl_original"],
            "ppl_reph": result["ppl_rephrased"],
            "ratio": result["ratio"],
            "leak": result["ratio"] > threshold,
        })

    return results


if __name__ == "__main__":
    # Quick test
    print("Testing PerplexityJudge module with batch processing...")

    judge = get_judge()

    # Test batch processing
    samples = [
        ("What is 2 + 2?", "What is 2 + 2?", "4"),
        ("What is 2 + 2?", "What is 2 + 2? The answer is 4.", "4"),
        ("What is 3 * 3?", "What is 3 * 3? Hint: it's 9.", "9"),
    ]

    print("\nBatch processing test:")
    results = judge_leak_batch(samples, judge=judge)
    for (orig, reph, ans), result in zip(samples, results):
        print(f"  Original: {orig[:30]}...")
        print(f"  Rephrased: {reph[:30]}...")
        print(f"  Answer: {ans}")
        print(f"  Result: ratio={result['ratio']:.3f}, leak={result['leak']}")
        print()
