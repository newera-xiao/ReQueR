#!/usr/bin/env python3
"""
Perplexity Leak Judge - HTTP Service

A standalone FastAPI server for perplexity-based leak detection.
Runs independently from Ray, occupying a single GPU.

Usage:
    # Start server on GPU 7 (appears as cuda:0 to the process)
    CUDA_VISIBLE_DEVICES=7 python server.py --port 8765

    # Or with uvicorn directly
    CUDA_VISIBLE_DEVICES=7 uvicorn server:app --host 0.0.0.0 --port 8765

    # Test the service
    curl -X POST http://localhost:8765/judge \
        -H "Content-Type: application/json" \
        -d '{"orig": "What is 2+2?", "reph": "What is 2+2? Answer is 4.", "ans": "4"}'
"""

import os
import argparse
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Global judge instance
_judge = None


class JudgeRequest(BaseModel):
    """Request body for leak judgment."""
    orig: str  # Original question
    reph: str  # Rephrased question
    ans: str   # Ground truth answer
    threshold: Optional[float] = 1.5


class JudgeResponse(BaseModel):
    """Response body for leak judgment."""
    ppl_orig: float
    ppl_reph: float
    ratio: float
    leak: bool


class BatchJudgeRequest(BaseModel):
    """Request body for batch leak judgment."""
    samples: list[dict]  # List of {orig, reph, ans}
    threshold: Optional[float] = 1.5


class BatchJudgeResponse(BaseModel):
    """Response body for batch leak judgment."""
    results: list[JudgeResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


def get_judge():
    """Get the global judge instance."""
    global _judge
    return _judge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _judge

    # Get configuration from environment or defaults
    model_path = os.environ.get(
        "LEAK_JUDGE_MODEL_PATH",
        "Qwen/Qwen3-0.6B"
    )
    # Always use cuda:0 since we set CUDA_VISIBLE_DEVICES externally
    device = os.environ.get("LEAK_JUDGE_DEVICE", "cuda:0")

    print(f"[LeakJudgeServer] Loading model from {model_path}")
    print(f"[LeakJudgeServer] Using device: {device}")

    from judge import PerplexityJudge
    _judge = PerplexityJudge(model_path=model_path, device=device)

    print("[LeakJudgeServer] Model loaded, server ready!")
    yield

    # Cleanup
    _judge = None
    print("[LeakJudgeServer] Shutdown complete")


app = FastAPI(
    title="Perplexity Leak Judge",
    description="HTTP service for perplexity-based answer leak detection",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    judge = get_judge()
    return HealthResponse(
        status="ok" if judge is not None else "not_ready",
        model_loaded=judge is not None,
        device=str(judge.device) if judge else "none",
    )


@app.post("/judge", response_model=JudgeResponse)
async def judge_leak(request: JudgeRequest):
    """
    Judge whether a rephrased question leaks answer information.

    Returns perplexity values and leak detection result.
    """
    judge = get_judge()
    if judge is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = judge.compute_leakage_ratio(
            original_query=request.orig,
            rephrased_query=request.reph,
            answer=request.ans,
        )

        is_leak = result["ratio"] > request.threshold

        return JudgeResponse(
            ppl_orig=result["ppl_original"],
            ppl_reph=result["ppl_rephrased"],
            ratio=result["ratio"],
            leak=is_leak,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/judge_batch", response_model=BatchJudgeResponse)
async def judge_leak_batch(request: BatchJudgeRequest):
    """
    Batch judge for multiple samples using efficient GPU batch processing.

    Each sample should have keys: orig, reph, ans
    All samples are processed in a single forward pass for efficiency.
    """
    judge = get_judge()
    if judge is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to list of tuples for batch processing
        samples = [
            (
                sample.get("orig", ""),
                sample.get("reph", ""),
                sample.get("ans", ""),
            )
            for sample in request.samples
        ]

        # Use batch processing (single forward pass)
        batch_results = judge.compute_leakage_ratio_batch(samples)

        # Convert to response format
        results = []
        for result in batch_results:
            is_leak = result["ratio"] > request.threshold
            results.append(JudgeResponse(
                ppl_orig=result["ppl_original"],
                ppl_reph=result["ppl_rephrased"],
                ratio=result["ratio"],
                leak=is_leak,
            ))

        return BatchJudgeResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Perplexity Leak Judge Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Model path (overrides LEAK_JUDGE_MODEL_PATH env var)"
    )
    args = parser.parse_args()

    # Set model path if provided via CLI
    if args.model_path:
        os.environ["LEAK_JUDGE_MODEL_PATH"] = args.model_path

    print(f"[LeakJudgeServer] Starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
