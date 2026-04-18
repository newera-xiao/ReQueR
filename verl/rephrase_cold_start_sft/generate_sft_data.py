#!/usr/bin/env python3
"""
生成 Refiner 的 SFT 训练数据
从现有的 parquet 文件读取问题，调用 API 生成高质量的 rephrase，保存为 SFT 格式
"""

import asyncio
import httpx
import pandas as pd
import json
import os
import sys
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import os
from tqdm import tqdm
import time
from datasets import load_dataset

# Add repo root to path so we can import rephrase_template.py from the top-level repo.
REPO_ROOT = Path(os.environ.get("REQUER_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT))
from rephrase_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

# API 配置
BASE_URL = "http://dsr1.sii.edu.cn/v1/chat/completions"
MODEL_NAME = "deepseek-r1-0528-ep"
TIMEOUT = 300.0

# 输入输出路径
OUTPUT_DIR = REPO_ROOT / "verl" / "rephrase_cold_start_sft"
OUTPUT_PARQUET = OUTPUT_DIR / "mixed_sft_with_responses.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "mixed_sft_with_responses.jsonl"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}


def extract_rephrase_with_tags(response: str) -> tuple[str, bool]:
    """
    从模型输出中提取严格的连续片段：
    <think>...</think><rephrase>...</rephrase>

    返回:
        (extracted_content, success):
            - extracted_content: 提取的内容（包括找到的标签），如果失败则返回原始响应
            - success: 是否成功提取到 <rephrase> 标签（核心要求）
    """
    import re
    # 严格匹配连续的 <think>...</think><rephrase>...</rephrase>
    pattern = r"(<think>.*?</think>\s*<rephrase>.*?</rephrase>)"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip(), True

    # 缺少任一标签，视为失败，返回原始响应
    return response.strip(), False


def extract_question(record: dict, candidates: list[str]) -> str | None:
    """从给定字段列表中依次取第一个非空字符串；对 OpenHermes 处理 conversations[0]['value']"""
    # 专门处理 OpenHermes 对话格式
    if "conversations" in record and isinstance(record["conversations"], list):
        try:
            first = record["conversations"][0]
            if isinstance(first, dict):
                val = first.get("value")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass

    # 通用候选键遍历
    for key in candidates:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


async def ask_one(messages: list[dict], client: httpx.AsyncClient) -> str:
    """调用 API 生成一个 rephrase"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 2048,
    }
    try:
        resp = await client.post(BASE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error: {e}")
        return f"<rephrase>Error generating rephrase: {e}</rephrase>"


async def batch_generate_rephrases(
    questions: list[str],
    max_concurrency: int = 64,
    batch_size: int = 128
):
    """批量生成 rephrases"""
    all_responses = []

    # 分批处理以避免超时
    for batch_start in range(0, len(questions), batch_size):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start}-{batch_end} ({len(batch_questions)} samples)...")
        start = time.time()

        semaphore = asyncio.Semaphore(max_concurrency)

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async def sem_task(q):
                async with semaphore:
                    # 构建完整的 messages
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(original_question=q)},
                    ]
                    return await ask_one(messages, client)

            tasks = [sem_task(q) for q in batch_questions]
            batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"API calls [{batch_start}-{batch_end}]")
            all_responses.extend(batch_responses)

        duration = time.time() - start
        print(f"Batch completed in {duration:.2f}s ({len(batch_questions)/duration:.1f} req/s)")

        # 短暂休息避免限流
        if batch_end < len(questions):
            print("Sleeping 2s before next batch...")
            await asyncio.sleep(2)

    return all_responses


def main():
    print("="*80)
    print("Refiner SFT Data Generation with Iterative Retry (Mixed Datasets)")
    print("="*80)

    # 1. 加载多数据集并采样
    DATASETS = [
        {
            "name": "teknium/OpenHermes-2.5",
            "split": "train",
            "sample_size": 4096,
            # conversations[0]['value'] (human) as question; fallback keys if missing
            "question_keys": ["conversations", "prompt", "question", "text", "instruction", "input"],
        },
        {
            "name": "openai/gsm8k",
            "split": "train",
            "subset": "main",
            "sample_size": 1024,
            "question_keys": ["question", "text"],
        },
        {
            "name": "qwedsacf/competition_math",
            "split": "train",
            "sample_size": 1024,
            "question_keys": ["problem", "question", "text"],
        },
    ]

    questions: list[str] = []
    dataset_tag: list[str] = []

    print("\n[Step 1] Loading datasets and sampling...")
    for cfg in DATASETS:
        print(f"- Loading {cfg['name']} ({cfg['split']})")
        ds = load_dataset(cfg["name"], cfg.get("subset"), split=cfg["split"])
        ds = ds.shuffle(seed=42).select(range(min(cfg["sample_size"], len(ds))))
        for rec in ds:
            q = extract_question(rec, cfg["question_keys"])
            if q:
                questions.append(q)
                dataset_tag.append(cfg["name"])
        print(f"  Collected {len(questions)} questions so far...")

    print(f"\nTotal collected questions: {len(questions)}")

    # 3. 多轮迭代生成 rephrases
    print("\n[Step 3] Generating rephrases via API with iterative retry...")
    print(f"API: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")

    MAX_ROUNDS = 2  # 最多迭代5轮
    responses = [None] * len(questions)  # 初始化响应列表

    # 第一轮：生成所有样本
    current_round = 1
    indices_to_generate = list(range(len(questions)))
    questions_to_generate = questions.copy()

    while current_round <= MAX_ROUNDS and len(indices_to_generate) > 0:
        print(f"\n{'='*80}")
        print(f"Round {current_round}: Generating {len(questions_to_generate)} samples")
        print(f"{'='*80}")
        print(f"Estimated time: {len(questions_to_generate) * 2 / 60:.1f} minutes")

        # 生成当前轮次的响应
        round_responses = asyncio.run(batch_generate_rephrases(
            questions_to_generate,
            max_concurrency=128,
            batch_size=256
        ))

        # 将响应填充到对应位置
        for i, idx in enumerate(indices_to_generate):
            responses[idx] = round_responses[i]

        print(f"\nGenerated {len(round_responses)} responses in round {current_round}")

        # 检查哪些样本提取失败
        failed_indices = []
        failed_questions = []

        for idx in indices_to_generate:
            response = responses[idx]
            _, success = extract_rephrase_with_tags(response)
            if not success:
                failed_indices.append(idx)
                failed_questions.append(questions[idx])

        success_count = len(indices_to_generate) - len(failed_indices)
        print(f"Extraction results: {success_count}/{len(indices_to_generate)} succeeded")
        print(f"Success rate this round: {success_count/len(indices_to_generate)*100:.2f}%")

        if len(failed_indices) == 0:
            print(f"\n🎉 All samples successfully extracted in round {current_round}!")
            break

        # 准备下一轮
        print(f"\n⚠️  {len(failed_indices)} samples failed extraction, will retry in round {current_round + 1}")
        indices_to_generate = failed_indices
        questions_to_generate = failed_questions
        current_round += 1

    # 最终统计
    print(f"\n{'='*80}")
    print(f"Final Results after {min(current_round, MAX_ROUNDS)} rounds:")
    print(f"{'='*80}")

    total_success = sum(1 for r in responses if extract_rephrase_with_tags(r)[1])
    print(f"Total successful: {total_success}/{len(responses)} ({total_success/len(responses)*100:.2f}%)")

    if len(indices_to_generate) > 0:
        print(f"⚠️  {len(indices_to_generate)} samples still failed after {MAX_ROUNDS} rounds")

    # 4. 构建 SFT 数据格式
    print("\n[Step 4] Building SFT dataset and extracting rephrase tags...")
    sft_data = []
    extraction_stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "failed_samples": [],  # 记录失败样本（仍然失败的）
        "retry_count": {}  # 记录每个样本的重试次数
    }

    for idx, original_question in tqdm(list(enumerate(questions)), total=len(questions), desc="Building SFT samples"):
        ground_truth_answer = ""  # 无明确标准答案时留空

        # prompt: 系统 + 用户消息（chat格式）
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(original_question=original_question)},
        ]

        # response: API 生成的 rephrase，提取标签
        raw_response = responses[idx]
        extracted_response, success = extract_rephrase_with_tags(raw_response)

        # 统计提取结果
        extraction_stats["total"] += 1
        if success:
            extraction_stats["success"] += 1
        else:
            extraction_stats["failed"] += 1
            # 保存失败样本（最多保存5个最终仍然失败的）
            if len(extraction_stats["failed_samples"]) < 5:
                extraction_stats["failed_samples"].append({
                    "idx": idx,
                    "original_question": original_question[:100],
                    "raw_response": raw_response[:300]
                })

        sft_data.append({
            "prompt": prompt,
            "response": extracted_response,  # 使用提取后的内容
            "original_question": original_question,
            "ground_truth_answer": ground_truth_answer,
            "source": dataset_tag[idx] if idx < len(dataset_tag) else "",
            "extraction_success": success,  # 记录是否成功提取
            "raw_response": raw_response,  # 保留原始响应用于调试
        })

    # 5. 保存数据
    print("\n[Step 5] Saving datasets...")

    # 保存为 parquet（用于 verl SFT 训练）
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved parquet: {OUTPUT_PARQUET}")

    # 保存为 jsonl（便于检查）
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved jsonl: {OUTPUT_JSONL}")

    # 6. 打印样例
    print("\n" + "="*80)
    print("Final Rephrase Tag Extraction Statistics:")
    print("="*80)
    print(f"Total rounds executed: {min(current_round, MAX_ROUNDS)}")
    print(f"Total samples: {extraction_stats['total']}")
    print(f"Successfully extracted: {extraction_stats['success']} ({extraction_stats['success']/extraction_stats['total']*100:.2f}%)")
    print(f"Failed to extract: {extraction_stats['failed']} ({extraction_stats['failed']/extraction_stats['total']*100:.2f}%)")

    if extraction_stats['failed'] > 0:
        print(f"\n⚠️  Still failed samples after all retries (first {len(extraction_stats['failed_samples'])}):")
        for fail in extraction_stats["failed_samples"]:
            print(f"\n  Sample #{fail['idx']}:")
            print(f"    Original question: {fail['original_question']}...")
            print(f"    Raw response: {fail['raw_response']}...")
            print("    " + "-"*70)
    else:
        print(f"\n✅ All samples successfully extracted!")

    print("\n" + "="*80)
    print("Sample outputs:")
    print("="*80)

    for i in range(min(3, len(sft_data))):
        sample = sft_data[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Extraction success: {'✓' if sample['extraction_success'] else '✗'}")
        print(f"\n[Original Question]:")
        print(sample['original_question'][:200] + "...")
        print(f"\n[Extracted Response (with tags)]:")
        print(sample['response'][:300] + "...")
        print(f"\n[Ground Truth Answer]: {sample['ground_truth_answer']}")
        print("-"*80)

    print("\n" + "="*80)
    print("Done!")
    print(f"Total samples: {len(sft_data)}")
    print(f"Total rounds: {min(current_round, MAX_ROUNDS)}")
    print(f"Final extraction success rate: {extraction_stats['success']}/{extraction_stats['total']} ({extraction_stats['success']/extraction_stats['total']*100:.2f}%)")
    print(f"Output files:")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"  - {OUTPUT_JSONL}")
    print("="*80)


if __name__ == "__main__":
    main()
