import argparse
import asyncio
import httpx
import re
import sys
import time
from pathlib import Path

# 项目根目录，方便导入模板
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from rephrase_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# API 配置
BASE_URL = "http://dsr1.sii.edu.cn/v1/chat/completions"
MODEL_NAME = "deepseek-r1-0528-ep"
TIMEOUT = 300.0  # seconds


def extract_strict_think_rephrase(resp: str) -> tuple[str, bool]:
    """严格提取连续 <think>...</think><rephrase>...</rephrase> 片段。"""
    pattern = r"(<think>.*?</think>\s*<rephrase>.*?</rephrase>)"
    m = re.search(pattern, resp, re.DOTALL)
    if m:
        return m.group(1).strip(), True
    return resp.strip(), False


async def ask_one(question: str, client: httpx.AsyncClient) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(original_question=question)},
        ],
        "temperature": 0.6,
        "max_tokens": 3072,
    }
    resp = await client.post(BASE_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def main():
    parser = argparse.ArgumentParser(description="调用 Refiner API 并查看单条响应（含 <think>/<rephrase> 提取）。")
    parser.add_argument(
        "--question",
        type=str,
        default="A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
        help="要重述的原始问题",
    )
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        raw = await ask_one(args.question, client)
    extracted, ok = extract_strict_think_rephrase(raw)

    print("\n=== Raw Response ===\n", raw)
    print("\n=== Extracted (<think>+<rephrase>) ===")
    print(extracted if ok else "[未同时包含 <think> 和 <rephrase>]")
    print(f"\nFormat valid: {ok}")


if __name__ == "__main__":
    asyncio.run(main())
