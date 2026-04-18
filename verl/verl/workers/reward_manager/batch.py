# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.
    This version is optimized to call the reward computation function once
    per batch, enabling efficient, concurrent API calls for reward calculation.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data: DataProto) -> list:
        """
        Prepares the data and calls the batch-aware compute_score function.
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Decode all responses in the batch
        responses_str = [
            self.tokenizer.decode(response_ids[i][:valid_response_lengths[i]], skip_special_tokens=True)
            for i in range(len(data))
        ]

        # Collect all ground truths and data sources
        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        
        # Extract extra information (including original questions when available)
        extras = []
        for item in data:
            reward_model = item.non_tensor_batch["reward_model"]
            base_extra_info = item.non_tensor_batch.get("extra_info", {})
            extra_info = dict(base_extra_info) if isinstance(base_extra_info, dict) else {}

            original_question = reward_model.get("original_question")
            if original_question:
                extra_info["original_question"] = original_question

            extras.append(extra_info)

        # --- KEY CHANGE: Call compute_score in BATCH mode ---
        # We pass all the data as lists to the scoring function.
        scores = self.compute_score(
            is_batch=True,  # Signal that this is a batch call
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )
        
        # Ensure the output is a list of scores
        assert isinstance(scores, list) and len(scores) == len(data), \
            f"Batch reward function must return a list of scores with the same length as the input batch."

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False, current_step: int = None) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Store current step for use by reward calculation
        if current_step is not None:
            import os
            os.environ["CURRENT_TRAINING_STEP"] = str(current_step)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        
        # Get all scores for the batch with a single call
        scores = self.verify(data)
        
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
