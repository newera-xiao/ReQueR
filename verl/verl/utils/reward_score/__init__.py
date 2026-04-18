# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    # --- New arguments for batch processing ---
    is_batch=False,
    solution_strs=None,
    ground_truths=None,
    data_sources=None,
    extra_infos=None,

):
    """Compute the score for a given solution based on the data source.
       Can operate in single or batch mode.
    """
    # --- Batch Processing Logic ---
    if is_batch:
        # Check for dual reward rephrase+CoT data source
        if data_sources is not None and len(data_sources) > 0 and data_sources[0] == "dual_reward_rephrase_cot":
            from . import dual_reward_rephrase_cot
            # Extract original questions from extra_infos
            original_questions = []
            for extra_info in (extra_infos or []):
                if isinstance(extra_info, dict) and "original_question" in extra_info:
                    original_questions.append(extra_info["original_question"])
                else:
                    original_questions.append("")  # Fallback
            return dual_reward_rephrase_cot.compute_dual_rewards_batch(solution_strs, ground_truths, original_questions)
        
        # For original api_based_reward data source
        elif data_sources is not None and len(data_sources) > 0 and data_sources[0] == "api_based_reward":
            from . import api_based_reward
            original_questions = []
            if extra_infos:
                for info in extra_infos:
                    if isinstance(info, dict):
                        original_questions.append(info.get("original_question", ""))
                    else:
                        original_questions.append("")
            else:
                original_questions = [""] * len(solution_strs)
            if len(original_questions) < len(solution_strs):
                original_questions.extend([""] * (len(solution_strs) - len(original_questions)))
            elif len(original_questions) > len(solution_strs):
                original_questions = original_questions[: len(solution_strs)]
            solver_labels = None
            if extra_infos:
                solver_labels = []
                for info in extra_infos:
                    if isinstance(info, dict):
                        solver_labels.append(info.get("solver_label"))
                    else:
                        solver_labels.append(None)
                if len(solver_labels) < len(solution_strs):
                    solver_labels.extend([None] * (len(solution_strs) - len(solver_labels)))
                elif len(solver_labels) > len(solution_strs):
                    solver_labels = solver_labels[: len(solution_strs)]
            # Directly call the batch computation function
            return api_based_reward.compute_scores_batch(
                solution_strs, ground_truths, original_questions, solver_labels=solver_labels
            )
        else:
            # Fallback for other data sources: compute scores sequentially in a loop
            # This maintains compatibility with existing reward functions.
            scores = []
            for i in range(len(solution_strs)):
                score = default_compute_score(
                    data_sources[i],
                    solution_strs[i],
                    ground_truths[i],
                    extra_infos[i] if extra_infos else None,
                    sandbox_fusion_url,
                    concurrent_semaphore,
                    memory_limit_mb,
                )
                scores.append(score)
            return scores

    # --- Single Item Processing Logic (Original Code) ---
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source == "api_based_reward":
        # This path is now for single-item (non-batch) calls, which might still be useful for debugging.
        from . import api_based_reward
        # The original api_based_reward.py had a `compute_score` function.
        # Let's create a synchronous wrapper for our async single-item compute function.
        # This part requires a small refactor in api_based_reward.py if we want to keep single-item sync call.
        # For now, we assume batch is always used for "api_based_reward".
        # If you need single-item sync calls, we can add a `compute_score` back to api_based_reward.py
        raise NotImplementedError("Single-item processing for api_based_reward is not the primary path. Use batch mode.")
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum", "numina_synthetic_math", "numina_amc_aime",
        "numina_synthetic_amc", "numina_cn_k12", "numina_olympiads",
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        if sandbox_fusion_url:
            from . import sandbox_fusion
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            from . import prime_code
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq", "searchR1_triviaqa", "searchR1_popqa",
        "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(*args, **kwargs):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(*args, **kwargs)


__all__ = ["default_compute_score"]
