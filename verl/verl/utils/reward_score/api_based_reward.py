# In verl/verl/utils/reward_score/api_based_reward.py
# This version supports both API calls and vLLM offline inference for batch processing.

import asyncio
from collections import defaultdict
import httpx
import os
import time
import json
import re
import torch
from .math import is_equiv, last_boxed_only_string, remove_boxed
from ..tokenizer import hf_tokenizer
from ..solver_pool import get_default_solver_index, get_solver_specs, solver_pool_enabled

# Try to import vLLM for offline inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available, falling back to API mode")
    VLLM_AVAILABLE = False

# --- Configuration is read from Environment Variables ---
API_URL = os.environ.get("REWARD_API_URL")
API_MODEL = os.environ.get("REWARD_API_MODEL", "default-model")
API_KEY = os.environ.get("REWARD_API_KEY") # Optional API key
MAX_CONCURRENCY = int(os.environ.get("REWARD_API_MAX_CONCURRENCY", "64"))
TIMEOUT = 60.0  # seconds

# vLLM offline inference configuration
VLLM_MODEL_PATH = os.environ.get("VLLM_MODEL_PATH", "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct")
VLLM_TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
VLLM_BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE", "512"))

# Penalty configuration to discourage leaking answers inside <rephrase> tags
ENABLE_ANSWER_LEAK_PENALTY = os.environ.get("ENABLE_REPHRASE_ANSWER_PENALTY", "true").lower() == "true"
ANSWER_LEAK_PENALTY = float(os.environ.get("REPHRASE_ANSWER_PENALTY", "1.0"))

# Perplexity-based leak judge configuration
ENABLE_PERPLEXITY_LEAK_JUDGE = os.environ.get("ENABLE_PERPLEXITY_LEAK_JUDGE", "false").lower() == "true"
PERPLEXITY_LEAK_THRESHOLD = float(os.environ.get("PERPLEXITY_LEAK_THRESHOLD", "1.5"))
# HTTP service endpoint (preferred method - independent of Ray)
PERPLEXITY_JUDGE_ENDPOINT = os.environ.get("LEAK_JUDGE_ENDPOINT", "http://localhost:8765")
# Legacy: direct model loading (not recommended with Ray)
PERPLEXITY_JUDGE_MODEL_PATH = os.environ.get(
    "LEAK_JUDGE_MODEL_PATH",
    os.path.join(os.environ.get("MODEL_DIR", ""), "Qwen3-0.6B") if os.environ.get("MODEL_DIR") else "",
)
PERPLEXITY_JUDGE_DEVICE = os.environ.get("LEAK_JUDGE_DEVICE", "cuda:7")

# Shared system prompt used by both API- and vLLM-based solvers
SOLVER_SYSTEM_PROMPT = (
    "You are a math assistant. Solve the following problem step by step, and put your final answer in \\boxed{} format."
)

# Solver-pool configuration
SOLVER_POOL_ACTIVE = solver_pool_enabled()

# Global vLLM model instance (will be initialized once)
_vllm_model = None
_vllm_sampling_params = None
_reward_tokenizer = None
_solver_executor = None
_perplexity_judge = None  # Perplexity-based leak judge instance


def _parse_device_map(num_solvers: int) -> list[str]:
    explicit_map = os.environ.get("REPHRASE_SOLVER_DEVICE_MAP", "").strip()
    if explicit_map:
        devices = [dev.strip() for dev in explicit_map.split(",") if dev.strip()]
    else:
        visible = os.environ.get("REPHRASE_SOLVER_VISIBLE_DEVICES", "").strip()
        if visible:
            devices = [dev.strip() for dev in visible.split(",") if dev.strip()]
        else:
            total = torch.cuda.device_count()
            devices = [str(i) for i in range(total)]
    if not devices:
        raise RuntimeError("No CUDA devices available for solver pool.")

    if len(devices) < num_solvers:
        devices.extend([devices[-1]] * (num_solvers - len(devices)))
    return devices[:num_solvers]


class MultiSolverExecutor:
    """Load multiple solvers with vLLM sleep mode enabled and wake/sleep on demand."""

    def __init__(self, solver_specs):
        self.solver_specs = solver_specs
        self.tensor_parallel_size = VLLM_TENSOR_PARALLEL_SIZE
        self.gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION
        self.max_model_len = VLLM_MAX_MODEL_LEN
        self._models = []
        self._sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, top_p=1.0)
        # self._sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)  # Greedy decoding for deterministic evaluation
        self._load_models()

    def _load_models(self):
        device_map = _parse_device_map(len(self.solver_specs))
        backend = os.environ.get("REPHRASE_SOLVER_BACKEND", "ray")
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")

        for spec, device in zip(self.solver_specs, device_map):
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            print("\n" + "=" * 88)
            print(f"[solver-pool] Loading solver '{spec['name']}' on GPU {device} from {spec['path']}")
            print("=" * 88)
            llm = LLM(
                model=spec["path"],
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype="bfloat16",
                trust_remote_code=True,
                enable_sleep_mode=False,
                distributed_executor_backend=backend,
            )
            self._models.append(llm)

        if original_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        print(f"[solver-pool] Loaded {len(self._models)} solvers (backend={backend}).")

    @property
    def num_solvers(self) -> int:
        return len(self._models)

    def default_index(self) -> int:
        if self.num_solvers == 0:
            return 0
        return min(self.num_solvers // 2, self.num_solvers - 1)

    def normalize_label(self, label: int | None) -> int:
        if self.num_solvers == 0:
            return 0
        if label is None:
            return self.default_index()
        try:
            label = int(label)
        except Exception:
            label = self.default_index()
        if label < 0:
            return 0
        if label >= self.num_solvers:
            return self.num_solvers - 1
        return label

    def run_prompts(self, solver_idx: int, prompts: list[str]):
        llm = self._models[solver_idx]
        outputs = llm.generate(prompts, sampling_params=self._sampling_params)
        return outputs


def get_multi_solver_executor() -> MultiSolverExecutor | None:
    global _solver_executor
    if not SOLVER_POOL_ACTIVE:
        return None
    if _solver_executor is not None:
        return _solver_executor
    specs = get_solver_specs()
    if not specs:
        return None
    try:
        _solver_executor = MultiSolverExecutor(specs)
    except Exception as exc:
        print(f"[solver-pool] Failed to initialize multi-solver executor: {exc}")
        _solver_executor = None
    return _solver_executor


def get_perplexity_judge():
    """
    Get or initialize the perplexity-based leak judge.

    DEPRECATED: Use HTTP service (LEAK_JUDGE_ENDPOINT) instead.
    This function is kept for backward compatibility but not recommended
    when running with Ray.

    Returns:
        PerplexityJudge instance or None if not enabled/available
    """
    global _perplexity_judge

    if not ENABLE_PERPLEXITY_LEAK_JUDGE:
        return None

    if _perplexity_judge is not None:
        return _perplexity_judge

    try:
        # Import the judge module from perplexity-judge directory.
        # Location resolved via REQUER_ROOT (default: sibling of the verl/ dir).
        import sys
        requer_root = os.environ.get(
            "REQUER_ROOT",
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        )
        judge_path = os.environ.get("PERPLEXITY_JUDGE_DIR", os.path.join(requer_root, "perplexity-judge"))
        if judge_path not in sys.path:
            sys.path.insert(0, judge_path)

        from judge import PerplexityJudge

        print(f"[PerplexityJudge] Initializing on {PERPLEXITY_JUDGE_DEVICE}...")
        _perplexity_judge = PerplexityJudge(
            model_path=PERPLEXITY_JUDGE_MODEL_PATH,
            device=PERPLEXITY_JUDGE_DEVICE,
        )
        print(f"[PerplexityJudge] Ready on {PERPLEXITY_JUDGE_DEVICE}")
        return _perplexity_judge

    except Exception as exc:
        print(f"[PerplexityJudge] Failed to initialize: {exc}")
        print("[PerplexityJudge] Falling back to string-based leak detection")
        return None


def check_perplexity_leak_http(
    original_question: str,
    rephrased_question: str,
    answer: str,
) -> tuple[bool, dict]:
    """
    Check for answer leak using HTTP service (single sample).

    This is the recommended method when running with Ray.
    The leak judge service runs independently on a dedicated GPU.

    Args:
        original_question: Original problem text
        rephrased_question: Rephrased problem text
        answer: Ground truth answer

    Returns:
        tuple[bool, dict]: (is_leak, result_dict)
    """
    import requests

    try:
        response = requests.post(
            f"{PERPLEXITY_JUDGE_ENDPOINT}/judge",
            json={
                "orig": original_question,
                "reph": rephrased_question,
                "ans": answer,
                "threshold": PERPLEXITY_LEAK_THRESHOLD,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("leak", False), {
            "ppl_orig": result.get("ppl_orig", 0),
            "ppl_reph": result.get("ppl_reph", 0),
            "ratio": result.get("ratio", 1.0),
            "leak": result.get("leak", False),
        }

    except requests.exceptions.ConnectionError:
        # Service not running - silently fall back to no leak detection
        return False, {"error": "service_unavailable"}
    except Exception as exc:
        print(f"[PerplexityJudge] HTTP request failed: {exc}")
        return False, {"error": str(exc)}


def check_perplexity_leak_batch_http(
    samples: list[tuple[str, str, str]],
) -> list[tuple[bool, dict]]:
    """
    Batch check for answer leak using HTTP service.

    Sends all samples in a single request for efficient GPU batch processing.

    Args:
        samples: List of (original_question, rephrased_question, answer) tuples

    Returns:
        List of (is_leak, result_dict) tuples
    """
    import requests

    if not samples:
        return []

    try:
        # Convert to request format
        request_samples = [
            {"orig": orig, "reph": reph, "ans": ans}
            for orig, reph, ans in samples
        ]

        response = requests.post(
            f"{PERPLEXITY_JUDGE_ENDPOINT}/judge_batch",
            json={
                "samples": request_samples,
                "threshold": PERPLEXITY_LEAK_THRESHOLD,
            },
            timeout=120.0,  # Longer timeout for batch
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("results", []):
            results.append((
                result.get("leak", False),
                {
                    "ppl_orig": result.get("ppl_orig", 0),
                    "ppl_reph": result.get("ppl_reph", 0),
                    "ratio": result.get("ratio", 1.0),
                    "leak": result.get("leak", False),
                }
            ))

        # Pad with defaults if needed
        while len(results) < len(samples):
            results.append((False, {"error": "missing_result"}))

        return results

    except requests.exceptions.ConnectionError:
        # Service not running - return no leak for all
        return [(False, {"error": "service_unavailable"}) for _ in samples]
    except Exception as exc:
        print(f"[PerplexityJudge] Batch HTTP request failed: {exc}")
        return [(False, {"error": str(exc)}) for _ in samples]


def check_perplexity_leak(
    original_question: str,
    rephrased_question: str,
    answer: str,
) -> tuple[bool, dict]:
    """
    Check for answer leak using perplexity-based judge.

    Automatically uses HTTP service if LEAK_JUDGE_ENDPOINT is set,
    otherwise falls back to direct model loading (not recommended with Ray).

    Args:
        original_question: Original problem text
        rephrased_question: Rephrased problem text
        answer: Ground truth answer

    Returns:
        tuple[bool, dict]: (is_leak, result_dict)
        - is_leak: True if leak detected
        - result_dict: Contains ppl_orig, ppl_reph, ratio, leak
    """
    # Prefer HTTP service (independent of Ray)
    if PERPLEXITY_JUDGE_ENDPOINT:
        return check_perplexity_leak_http(original_question, rephrased_question, answer)

    # Fallback: direct model loading (not recommended with Ray)
    judge = get_perplexity_judge()
    if judge is None:
        return False, {}

    try:
        result = judge.compute_leakage_ratio(original_question, rephrased_question, answer)
        is_leak = result["ratio"] > PERPLEXITY_LEAK_THRESHOLD
        return is_leak, {
            "ppl_orig": result["ppl_original"],
            "ppl_reph": result["ppl_rephrased"],
            "ratio": result["ratio"],
            "leak": is_leak,
        }
    except Exception as exc:
        print(f"[PerplexityJudge] Error during check: {exc}")
        return False, {}


def initialize_vllm_model():
    """Initialize vLLM model for offline inference (called once)."""
    global _vllm_model, _vllm_sampling_params

    if not VLLM_AVAILABLE:
        return False

    if _vllm_model is not None:
        return True

    try:
        print("🔄 Initializing vLLM model for offline inference...")

        # Set GPU devices for vLLM (use GPU 4-7)
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

        _vllm_model = LLM(
            model=VLLM_MODEL_PATH,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
            dtype="bfloat16",
            trust_remote_code=True
        )

        _vllm_sampling_params = SamplingParams(
            temperature=0.7,  # Deterministic for scoring
            max_tokens=1024,  # Longer output for math solutions
            top_p=1.0
        )

        print("✅ vLLM model initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to initialize vLLM model: {e}")
        return False


def get_vllm_model():
    """Get initialized vLLM model and sampling params."""
    global _vllm_model, _vllm_sampling_params

    if _vllm_model is None:
        if not initialize_vllm_model():
            return None, None

    return _vllm_model, _vllm_sampling_params


def get_reward_tokenizer():
    """Lazy load tokenizer for building chat prompts."""
    global _reward_tokenizer
    if _reward_tokenizer is not None:
        return _reward_tokenizer

    try:
        tokenizer_path = VLLM_MODEL_PATH
        if SOLVER_POOL_ACTIVE:
            specs = get_solver_specs()
            if specs:
                tokenizer_path = specs[0]["path"]
        _reward_tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        print(f"WARNING: Failed to load tokenizer for reward model ({exc}). Falling back to plain prompts.")
        _reward_tokenizer = False  # sentinel for fallback
    return _reward_tokenizer

def extract_rephrased_question(variant: str) -> str:
    """
    Extract the rephrased question from <rephrase>...</rephrase> tags.
    If no tags found, return the original variant.

    DEPRECATED: Use validate_and_extract_format() instead for strict format validation.
    This function is kept for backward compatibility.
    """
    rephrase_pattern = r"<rephrase>(.*?)</rephrase>"
    match = re.search(rephrase_pattern, variant, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no tags found, assume the entire variant is the rephrased question
    return variant.strip()


def validate_and_extract_format(variant: str) -> tuple[str, bool]:
    """
    Validate the format and extract the rephrased question.

    The expected format is: <think>...</think><rephrase>...</rephrase>
    Both tags MUST be present for the format to be valid.

    Returns:
        tuple[str, bool]: (extracted_rephrase, is_valid_format)
        - extracted_rephrase: The rephrased question if found, empty string otherwise
        - is_valid_format: True if format is valid (both <think> and <rephrase> present)

    Format penalty logic:
    - If format is invalid (missing either tag), return empty string
    - Format reward = 1.0 if valid, 0.0 if invalid
    - Final reward = base_reward * format_reward
    """
    # Check for both <think> and <rephrase> tags
    think_pattern = r"<think>.*?</think>"
    rephrase_pattern = r"<rephrase>(.*?)</rephrase>"

    has_think = bool(re.search(think_pattern, variant, re.DOTALL))
    rephrase_match = re.search(rephrase_pattern, variant, re.DOTALL)
    has_rephrase = bool(rephrase_match)

    # Format is valid only if BOTH tags are present
    is_valid_format = has_think and has_rephrase

    if is_valid_format:
        # Extract the rephrase content
        extracted_rephrase = rephrase_match.group(1).strip()
    else:
        # Invalid format - return empty string
        extracted_rephrase = ""

    return extracted_rephrase, is_valid_format


def _normalize_for_match(text: str) -> str:
    """Remove whitespace and commas so we can do a simple substring check."""
    return re.sub(r"\s+", "", (text or "").replace(",", ""))


def contains_answer_leak(rephrased_text: str, ground_truth: str, original_question: str = "") -> bool:
    """
    Detect leakage by checking whether the ground-truth appears in the rephrase output.

    Enhanced logic to reduce false positives:
    - If the ground truth already appears in the original question (e.g., short numbers like "2"),
      we only consider it a leak if the rephrase contains \\boxed{answer} pattern.
    - Otherwise, we use the standard substring matching.

    This prevents false positives where the answer is a common short number that naturally
    appears in the problem statement (e.g., "x^2", "2 apples", etc.).
    """
    norm_gt = _normalize_for_match(ground_truth)
    if not norm_gt:
        return False

    norm_rephrase = _normalize_for_match(rephrased_text)
    norm_original = _normalize_for_match(original_question)

    # Check if answer already appears in original question
    answer_in_original = norm_gt in norm_original if norm_original else False

    if answer_in_original:
        # For answers that naturally appear in the original question,
        # only flag as leak if \boxed{answer} pattern is found
        # This catches explicit answer leakage while avoiding false positives
        boxed_patterns = [
            rf"\\boxed\{{{re.escape(ground_truth)}\}}",  # \boxed{answer}
            rf"\\boxed\s*{re.escape(ground_truth)}",     # \boxed answer (space variant)
        ]
        for pattern in boxed_patterns:
            if re.search(pattern, rephrased_text, re.IGNORECASE):
                return True
        return False
    else:
        # Standard check: answer not in original, so any appearance in rephrase is suspicious
        return norm_gt in norm_rephrase


# def build_solver_prompt(original_question: str, rephrased_question: str) -> str:
#     """
#     Provide both the original wording and the rephrased wording to the solver model.
#     This helps the solver double-check semantic equivalence while still relying on the rewrite.
#     """
#     sections = []
#     original_question = (original_question or "").strip()
#     rephrased_question = (rephrased_question or "").strip()

#     if original_question:
#         sections.append(f"Original Problem:\n{original_question}")
#     if rephrased_question:
#         sections.append(f"Rephrased Problem:\n{rephrased_question}")

#     sections.append(
#         "Use the rephrased problem to reason but ensure the requirements match the original. "
#         "Provide the final answer in \\boxed{} format."
#     )

#     return "\n\n".join(sections)

def build_solver_prompt(original_question: str, rephrased_question: str) -> str:
    """
    Provide both the original wording and the rephrased wording to the solver model.
    This helps the solver double-check semantic equivalence while still relying on the rewrite.
    """
    sections = []
    original_question = (original_question or "").strip()
    rephrased_question = (rephrased_question or "").strip()

    # if original_question:
    #     sections.append(f"Original Problem:\n{original_question}")
    if rephrased_question:
        sections.append(f"{rephrased_question}")

    # sections.append(
    #     "Use the rephrased problem to reason but ensure the requirements match the original. "
    #     "Provide the final answer in \\boxed{} format."
    # )
    
    return "\n\n".join(sections)

# Global list to collect leak samples during evaluation
_leak_samples = []

def get_and_clear_leak_samples() -> list:
    """Get collected leak samples and clear the buffer."""
    global _leak_samples
    samples = _leak_samples.copy()
    _leak_samples = []
    return samples

def _record_leak_sample(rephrased_text: str, ground_truth: str, original_question: str, base_score: float, penalized_score: float):
    """Record a leak sample for later saving."""
    global _leak_samples
    _leak_samples.append({
        "original_question": original_question,
        "rephrased_text": rephrased_text,
        "ground_truth": ground_truth,
        "base_score": base_score,
        "penalized_score": penalized_score,
    })

def _apply_answer_leak_penalty(
    rephrased_text: str, base_score: float, ground_truth: str, original_question: str = ""
) -> float:
    """
    Apply leak penalty if the rephrase text seems to contain an explicit answer pattern.

    Detection methods (in order of priority):
    1. Perplexity-based judge (if ENABLE_PERPLEXITY_LEAK_JUDGE=true)
    2. String-based heuristic (contains_answer_leak)

    The penalty is subtracted from the reward (floored at -1).
    """
    if not ENABLE_ANSWER_LEAK_PENALTY:
        return base_score

    leak_detected = False

    # Method 1: Perplexity-based leak detection (if enabled)
    if ENABLE_PERPLEXITY_LEAK_JUDGE:
        is_leak, ppl_result = check_perplexity_leak(
            original_question, rephrased_text, ground_truth
        )
        if is_leak:
            leak_detected = True
            print(
                f"[penalty] Perplexity leak detected. "
                f"ppl_orig={ppl_result.get('ppl_orig', 0):.2f}, "
                f"ppl_reph={ppl_result.get('ppl_reph', 0):.2f}, "
                f"ratio={ppl_result.get('ratio', 0):.3f} > {PERPLEXITY_LEAK_THRESHOLD}"
            )
    else:
        # Method 2: String-based leak detection (fallback)
        if contains_answer_leak(rephrased_text, ground_truth, original_question):
            leak_detected = True

    if leak_detected:
        penalized = max(-1.0, base_score - ANSWER_LEAK_PENALTY)
        if penalized != base_score:
            print(
                f"[penalty] Detected possible answer leakage. "
                f"Penalizing score from {base_score} to {penalized}"
            )
            # Record leak sample for later saving
            _record_leak_sample(rephrased_text, ground_truth, original_question, base_score, penalized)
        return penalized

    return base_score


def _apply_answer_leak_penalty_batch(
    rephrased_texts: list[str],
    base_scores: list[float],
    ground_truths: list[str],
    original_questions: list[str],
) -> list[float]:
    """
    Batch version of answer leak penalty application.

    Uses batch HTTP request for perplexity-based detection when enabled,
    significantly faster than per-sample processing.

    Args:
        rephrased_texts: List of rephrased question texts
        base_scores: List of base scores from solver
        ground_truths: List of ground truth answers
        original_questions: List of original questions

    Returns:
        List of penalized scores
    """
    if not ENABLE_ANSWER_LEAK_PENALTY:
        return base_scores

    n = len(rephrased_texts)
    penalized_scores = list(base_scores)  # Copy

    # Method 1: Perplexity-based batch detection (if enabled)
    if ENABLE_PERPLEXITY_LEAK_JUDGE and PERPLEXITY_JUDGE_ENDPOINT:
        samples = [
            (orig, reph, gt)
            for orig, reph, gt in zip(original_questions, rephrased_texts, ground_truths)
        ]
        batch_results = check_perplexity_leak_batch_http(samples)

        leak_count = 0
        for i, (is_leak, ppl_result) in enumerate(batch_results):
            if is_leak:
                leak_count += 1
                penalized = max(-1.0, base_scores[i] - ANSWER_LEAK_PENALTY)
                if penalized != base_scores[i]:
                    _record_leak_sample(
                        rephrased_texts[i], ground_truths[i],
                        original_questions[i], base_scores[i], penalized
                    )
                penalized_scores[i] = penalized

        if leak_count > 0:
            print(f"[penalty] Perplexity leak detected in {leak_count}/{n} samples")

    else:
        # Method 2: String-based leak detection (fallback, per-sample)
        leak_count = 0
        for i in range(n):
            if contains_answer_leak(rephrased_texts[i], ground_truths[i], original_questions[i]):
                leak_count += 1
                penalized = max(-1.0, base_scores[i] - ANSWER_LEAK_PENALTY)
                if penalized != base_scores[i]:
                    _record_leak_sample(
                        rephrased_texts[i], ground_truths[i],
                        original_questions[i], base_scores[i], penalized
                    )
                penalized_scores[i] = penalized

        if leak_count > 0:
            print(f"[penalty] String-based leak detected in {leak_count}/{n} samples")

    return penalized_scores


# --- Helper function to compute score for a single variant ---
async def _compute_one_score(
    rephrased_question: str, ground_truth: str, original_question: str, client: httpx.AsyncClient
) -> float:
    """
    Asynchronously calls the external API for a single variant and computes its score.
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    user_content = build_solver_prompt(original_question, rephrased_question)

    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        # "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        
        response_data = resp.json()
        api_answer_text = response_data["choices"][0]["message"]["content"]
        
        final_api_answer = last_boxed_only_string(api_answer_text)
        if final_api_answer:
            cleaned_api_answer = remove_boxed(final_api_answer)
            if is_equiv(cleaned_api_answer, ground_truth):
                base_score = 1.0
            else:
                base_score = 0.0
        else:
            base_score = 0.0

        return _apply_answer_leak_penalty(rephrased_question, base_score, ground_truth, original_question)

    except httpx.RequestError as e:
        print(f"ERROR: HTTP request failed for rephrased question '{rephrased_question[:50]}...': {e}")
        return _apply_answer_leak_penalty(rephrased_question, 0.0, ground_truth, original_question)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to parse API response for rephrased question '{rephrased_question[:50]}...': {e}")
        return _apply_answer_leak_penalty(rephrased_question, 0.0, ground_truth, original_question)


async def _compute_scores_batch_async(
    rephrased_questions: list[str], ground_truths: list[str], original_questions: list[str]
) -> list[float]:
    """
    The main async orchestrator, based on the user's example.
    Sends a batch of requests concurrently.
    """
    if not API_URL:
        print("ERROR: REWARD_API_URL is not set. Returning 0.0 for all rewards.")
        return [_apply_answer_leak_penalty(text, 0.0, gt, oq) for text, gt, oq in zip(rephrased_questions, ground_truths, original_questions)]
        
    start = time.time()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        
        async def sem_task(rephrased_text, ground_truth, original_question):
            async with semaphore:
                return await _compute_one_score(rephrased_text, ground_truth, original_question, client)

        tasks = [sem_task(v, gt, oq) for v, gt, oq in zip(rephrased_questions, ground_truths, original_questions)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start
    print(f"INFO: Processed reward calculation for {len(rephrased_questions)} variants in {duration:.2f} seconds.")
    
    # Handle potential exceptions returned by asyncio.gather
    final_scores = []
    for res in responses:
        if isinstance(res, Exception):
            print(f"ERROR: An unexpected exception occurred during batch processing: {res}")
            final_scores.append(0.0)
        else:
            final_scores.append(res)
            
    return final_scores


def compute_scores_batch_vllm(
    rephrased_questions: list[str],
    ground_truths: list[str],
    original_questions: list[str],
    solver_labels: list[int | None] | None = None,
    format_rewards: list[float] | None = None,
) -> list[float]:
    """
    vLLM offline batch processing for reward computation.
    Supports both single solver and multi-solver sleep-mode execution.

    Args:
        format_rewards: List of format rewards (1.0 for valid format, 0.0 for invalid)
                       Final reward = base_reward * format_reward
    """
    start_time = time.time()

    # If format_rewards not provided, assume all valid
    if format_rewards is None:
        format_rewards = [1.0] * len(rephrased_questions)

    executor = get_multi_solver_executor() if SOLVER_POOL_ACTIVE else None
    if executor is None:
        llm, sampling_params = get_vllm_model()
        if llm is None:
            print("vLLM model not available, falling back to API method")
            return compute_scores_batch_api(rephrased_questions, ground_truths, original_questions)
    else:
        llm = None
        sampling_params = None

    tokenizer = get_reward_tokenizer()
    use_chat_template = tokenizer not in (None, False)

    prompts: list[str] = []

    for rephrased_question, original_question in zip(rephrased_questions, original_questions):
        user_prompt = build_solver_prompt(original_question, rephrased_question)
        if use_chat_template:
            messages = [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = f"System: {SOLVER_SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"
        prompts.append(prompt)

    results = [0.0] * len(prompts)
    if not prompts:
        return results

    if executor is not None:
        normalized_labels = []
        total = len(prompts)
        for idx in range(total):
            label = None
            if solver_labels and idx < len(solver_labels):
                label = solver_labels[idx]
            normalized_labels.append(executor.normalize_label(label))

        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(normalized_labels):
            grouped_indices[label].append(idx)

        # Print solver distribution statistics
        print("\n" + "=" * 80)
        print("[Solver Distribution] Batch size:", total)
        solver_specs = get_solver_specs()
        solver_dist_parts = []
        for label in sorted(grouped_indices.keys()):
            count = len(grouped_indices[label])
            percentage = (count / total * 100) if total > 0 else 0
            solver_name = solver_specs[label]["name"] if label < len(solver_specs) else f"solver_{label}"
            print(f"  Solver {label} ({solver_name}): {count:4d} samples ({percentage:5.1f}%)")
            solver_dist_parts.append(f"{solver_name}: {percentage:.1f}%")
        print("=" * 80 + "\n")

        # Save solver distribution to file (override location via $ASH_LOG_PATH).
        ash_log_path = os.environ.get(
            "ASH_LOG_PATH",
            os.path.join(os.getcwd(), "ash_solver_distribution.log"),
        )
        try:
            os.makedirs(os.path.dirname(ash_log_path), exist_ok=True)
            with open(ash_log_path, "a") as f:
                f.write(", ".join(solver_dist_parts) + "\n")
        except Exception as e:
            print(f"[Warning] Failed to save solver distribution to {ash_log_path}: {e}")

        for label, indices in grouped_indices.items():
            for chunk_start in range(0, len(indices), VLLM_BATCH_SIZE):
                chunk_indices = indices[chunk_start : chunk_start + VLLM_BATCH_SIZE]
                chunk_prompts = [prompts[i] for i in chunk_indices]
                chunk_ground_truths = [ground_truths[i] for i in chunk_indices]
                chunk_rephrased = [rephrased_questions[i] for i in chunk_indices]
                chunk_original = [original_questions[i] for i in chunk_indices]
                try:
                    outputs = executor.run_prompts(label, chunk_prompts)
                    _assign_solver_scores(
                        results, chunk_indices, outputs, chunk_ground_truths, chunk_rephrased, chunk_original,
                        format_rewards=format_rewards
                    )
                except Exception as exc:
                    print(f"Error in multi-solver vLLM processing (solver {label}): {exc}")
                    for idx_local, global_idx in enumerate(chunk_indices):
                        # Apply format reward even on error
                        score_after_penalty = _apply_answer_leak_penalty(
                            chunk_rephrased[idx_local], 0.0, chunk_ground_truths[idx_local], chunk_original[idx_local]
                        )
                        results[global_idx] = score_after_penalty * format_rewards[global_idx]
    else:
        print(f"Processing {len(rephrased_questions)} variants with single vLLM solver...")
        for i in range(0, len(prompts), VLLM_BATCH_SIZE):
            batch_indices = list(range(i, min(i + VLLM_BATCH_SIZE, len(prompts))))
            batch_prompts = [prompts[j] for j in batch_indices]
            batch_ground_truths = [ground_truths[j] for j in batch_indices]
            batch_rephrased = [rephrased_questions[j] for j in batch_indices]
            batch_original = [original_questions[j] for j in batch_indices]
            try:
                outputs = llm.generate(batch_prompts, sampling_params)
                _assign_solver_scores(
                    results, batch_indices, outputs, batch_ground_truths, batch_rephrased, batch_original,
                    format_rewards=format_rewards
                )
            except Exception as exc:
                print(f"Error in vLLM batch processing: {exc}")
                for idx_local, global_idx in enumerate(batch_indices):
                    # Apply format reward even on error
                    score_after_penalty = _apply_answer_leak_penalty(
                        batch_rephrased[idx_local], 0.0, batch_ground_truths[idx_local], batch_original[idx_local]
                    )
                    results[global_idx] = score_after_penalty * format_rewards[global_idx]

    total_time = time.time() - start_time
    avg_score = sum(results) / len(results) if results else 0.0

    print(f"INFO: Processed {len(results)} variants in {total_time:.2f} seconds.")
    print(f"INFO: Average score: {avg_score:.3f}")
    return results


def _assign_solver_scores(
    destination: list[float],
    global_indices: list[int],
    outputs,
    ground_truths: list[str],
    rephrased_questions: list[str],
    original_questions: list[str] = None,
    format_rewards: list[float] = None,
    use_batch_leak_check: bool = True,
) -> None:
    """
    Assign solver scores to destination list.

    Args:
        format_rewards: List of format rewards (1.0 for valid, 0.0 for invalid)
                       Final reward = base_reward * answer_leak_penalty * format_reward
        use_batch_leak_check: If True, use batch leak checking (faster with HTTP service)
    """
    if original_questions is None:
        original_questions = [""] * len(global_indices)
    if format_rewards is None:
        format_rewards = [1.0] * len(global_indices)

    # Step 1: Compute all base scores first
    base_scores = []
    for offset in range(len(global_indices)):
        base_score = 0.0
        try:
            api_answer_text = outputs[offset].outputs[0].text.strip()
            final_api_answer = last_boxed_only_string(api_answer_text)
            if final_api_answer:
                cleaned_api_answer = remove_boxed(final_api_answer)
                if is_equiv(cleaned_api_answer, ground_truths[offset]):
                    base_score = 1.0
        except Exception as exc:
            print(f"Error decoding solver output: {exc}")
        base_scores.append(base_score)

    # Step 2: Apply leak penalty (batch or per-sample)
    if use_batch_leak_check and ENABLE_PERPLEXITY_LEAK_JUDGE and PERPLEXITY_JUDGE_ENDPOINT:
        # Use batch processing for perplexity-based leak detection
        scores_after_leak = _apply_answer_leak_penalty_batch(
            rephrased_questions, base_scores, ground_truths, original_questions
        )
    else:
        # Fall back to per-sample processing
        scores_after_leak = []
        for offset in range(len(global_indices)):
            score = _apply_answer_leak_penalty(
                rephrased_questions[offset], base_scores[offset],
                ground_truths[offset], original_questions[offset]
            )
            scores_after_leak.append(score)

    # Step 3: Apply format reward and assign to destination
    for offset, global_idx in enumerate(global_indices):
        final_score = scores_after_leak[offset] * format_rewards[global_idx]
        destination[global_idx] = final_score


def compute_scores_batch_api(
    rephrased_questions: list[str],
    ground_truths: list[str],
    original_questions: list[str],
    solver_labels: list[int | None] | None = None,
    format_rewards: list[float] | None = None,
) -> list[float]:
    """
    API-based batch processing (original implementation).

    Args:
        format_rewards: List of format rewards (1.0 for valid, 0.0 for invalid)
                       Final reward = base_reward * format_reward
    """
    if format_rewards is None:
        format_rewards = [1.0] * len(rephrased_questions)

    # Get base scores from API
    base_scores = asyncio.run(_compute_scores_batch_async(rephrased_questions, ground_truths, original_questions))

    # Apply format rewards
    final_scores = [base * fmt for base, fmt in zip(base_scores, format_rewards)]

    return final_scores


def compute_scores_batch(
    variants: list[str],
    ground_truths: list[str],
    original_questions: list[str] | None = None,
    solver_labels: list[int | None] | None = None,
) -> list[float]:
    """
    Main entry point for batch processing.
    Automatically chooses between vLLM offline and API based on availability and configuration.

    Now includes format validation:
    - Validates that variants contain both <think> and <rephrase> tags
    - If format is invalid, returns empty string and format_reward=0.0
    - Final reward = base_reward * format_reward
    """
    if original_questions is None:
        original_questions = [""] * len(variants)
    if len(original_questions) < len(variants):
        original_questions = original_questions + [""] * (len(variants) - len(original_questions))
    elif len(original_questions) > len(variants):
        original_questions = original_questions[: len(variants)]

    # Validate format and extract rephrased text
    rephrased_questions = []
    format_rewards = []

    for v in variants:
        extracted, is_valid = validate_and_extract_format(v)
        rephrased_questions.append(extracted)
        format_rewards.append(1.0 if is_valid else 0.0)

    # Print format validation statistics
    valid_count = sum(1 for r in format_rewards if r > 0)
    total_count = len(format_rewards)
    print(f"[Format Validation] Valid: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    if valid_count < total_count:
        print(f"[Format Validation] WARNING: {total_count - valid_count} samples have invalid format and will receive 0 reward")

    # Check if we should use vLLM offline inference
    use_vllm = VLLM_AVAILABLE and VLLM_MODEL_PATH

    # Allow environment variable override
    if os.environ.get("USE_VLLM_OFFLINE", "true").lower() == "true":
        if use_vllm:
            print("Using vLLM offline inference for reward computation")
            return compute_scores_batch_vllm(
                rephrased_questions, ground_truths, original_questions,
                solver_labels=solver_labels, format_rewards=format_rewards
            )
        else:
            print("vLLM not available, falling back to API mode")

    print("Using API-based reward computation")
    return compute_scores_batch_api(
        rephrased_questions, ground_truths, original_questions,
        solver_labels=solver_labels, format_rewards=format_rewards
    )

# Example usage for testing
if __name__ == "__main__":
    if API_URL:
        q_base = "What is 1+{}?"
        question_list = [q_base.format(i) for i in range(1, 21)]
        truth_list = [str(1+i) for i in range(1, 21)]

        scores = compute_scores_batch(question_list, truth_list)

        for i in range(20):
            print(f"[Q] {question_list[i]} [A] {truth_list[i]} [Score] {scores[i]}")
    else:
        print("Skipping example test because REWARD_API_URL environment variable is not set.")
