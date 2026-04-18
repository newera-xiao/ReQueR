<div align="center">

# One Refiner to Unlock Them All
### Inference-Time Reasoning Elicitation via Reinforcement Query Refinement

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-ACL_2026-b31b1b.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Yixiao Zhou**<sup>1,2</sup>, **Dongzhou Cheng**<sup>2,3</sup>, **Zhiliang Wu**<sup>1</sup>,
**Yi Yang**<sup>1</sup>, **Yu Cheng**<sup>2,4</sup>, **Hehe Fan**<sup>1</sup>

<sup>1</sup>Zhejiang University &nbsp;&nbsp; <sup>2</sup>Shanghai Innovation Institute
<sup>3</sup>Southeast University &nbsp;&nbsp; <sup>4</sup>The Chinese University of Hong Kong

</div>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/newera-xiao/ReQueR@main/assets/intro.png" width="65%" alt="ReQueR teaser"/>
</p>

<p align="center">
  <em>A single Refiner, trained once, unlocks the latent reasoning of many frozen Solvers — at inference time, with no parameter updates.</em>
</p>

---

## TL;DR

**ReQueR** (*Reinforcement Query Refinement*) treats reasoning elicitation as an **inference-time alignment** task. Instead of fine-tuning each downstream LLM, we train a small **Refiner** policy via reinforcement learning to rewrite raw human queries into structured, logically-decomposed forms that frozen Solvers can actually leverage. One trained Refiner generalizes across diverse unseen architectures and scales, shifting the alignment cost from $O(N)$ per-model tuning to a scalable $O(1)$ paradigm.

- 🔑 **One Refiner, many Solvers** — a single 4B Refiner lifts Qwen3, Llama, Mixtral, and DeepSeek-MoE families without touching their weights.
- 🎯 **Adaptive Solver Hierarchy (ASH)** — a curriculum that dynamically matches Solver difficulty to the Refiner's competence, inspired by the Zone of Proximal Development.
- 🛡️ **Perplexity-based leak guardrail** — a single-forward-pass constraint that blocks reward hacking via answer leakage.
- 📈 **+1.7% to +7.2%** absolute gains across 7 reasoning benchmarks; **+2.1%** average over the strongest baseline.

---

## Abstract

Large Language Models (LLMs) often fail to utilize their latent reasoning capabilities due to a distributional mismatch between ambiguous human inquiries and the structured logic required for machine activation. Existing alignment methods either incur prohibitive $O(N)$ costs by fine-tuning each model individually or rely on static prompts that fail to resolve query-level structural complexity. In this paper, we propose **ReQueR** (**Re**inforcement **Que**ry **R**efinement), a modular framework that treats reasoning elicitation as an inference-time alignment task. We train a specialized Refiner policy via Reinforcement Learning to rewrite raw queries into explicit logical decompositions, treating frozen LLMs as the environment. Rooted in the classical *Zone of Proximal Development* from educational psychology, we introduce the **Adaptive Solver Hierarchy**, a curriculum mechanism that stabilizes training by dynamically aligning environmental difficulty with the Refiner's evolving competence. ReQueR yields consistent absolute gains of **1.7%–7.2%** across diverse architectures and benchmarks, outperforming strong baselines by **2.1%** on average. Crucially, it provides a promising paradigm for one-to-many inference-time reasoning elicitation, enabling a single Refiner trained on a small set of models to effectively unlock reasoning in diverse unseen models.

---

## Method Overview

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/newera-xiao/ReQueR@main/assets/architecture.png" width="90%" alt="ReQueR RL pipeline"/>
</p>

ReQueR is trained in two stages:

1. **Cold-Start SFT** — initialize the Refiner $\pi_\theta$ on 6,144 curated `(raw_query, refined_query)` pairs, synthesized by DeepSeek-R1 and manually curated to establish output format and baseline refining capability.
2. **Online RL (GRPO)** — optimize $\pi_\theta$ against a heterogeneous Solver pool. For each refinement $x'$, the reward couples **task success** on the Solver's answer with a **perplexity-based leakage penalty**:

$$R = \mathbb{I}[\hat{y} = y^{\ast}] - \lambda \cdot \mathbb{I}\left[\frac{\mathrm{PPL}(y^{\ast} \mid x)}{\mathrm{PPL}(y^{\ast} \mid x') + \epsilon} \gt \tau_{\mathrm{leak}}\right]$$

The policy is then updated with group-normalized advantages $A_j = (R_j - \bar{R})/\sigma(R)$ over $G$ rollouts per prompt, following the GRPO objective. The perplexity term detects answer leakage with a single forward pass, avoiding the cost and instability of a generative LLM judge.

### Adaptive Solver Hierarchy (ASH)

For every training sample, ASH tracks a local difficulty index $k_i$ that selects a Solver from the pool $\mathcal{S} = \lbrace \mathcal{M}_1, \dots, \mathcal{M}_K \rbrace$. When all $G$ rollouts succeed, ASH de-escalates to a harder Solver to impose structural pressure; when all fail, it escalates to a stronger Solver to recover the reward signal. This keeps the Refiner in its *Zone of Proximal Development* and prevents overfitting to any single Solver's idiosyncratic tolerance. Together with the perplexity-based leak guardrail, ASH keeps rewards dense while blocking the policy from collapsing into answer-leaking shortcuts.

---

## Key Results

**Per-sample refinement generalizes across unseen Solver architectures** (Refiner trained on Qwen3-0.6B / 1.7B / 4B only):

| Solver | Method | AMC23 | Olym.Bench | Omni-MATH | GSM-Plus | GSM-Symbolic | MATH-500 | GPQA-Diamond | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | CoT | 18.44 | 19.14 | 13.14 | 44.48 | 50.80 | 48.80 | 27.53 | 31.76 |
| Qwen3-0.6B | **ReQueR** | **23.75** | **21.66** | **15.18** | **61.57** | **66.14** | **54.40** | **30.30** | **39.00** <sub>(+7.24)</sub> |
| Llama-3.1-8B | CoT | 25.94 | 16.02 | 11.63 | 66.86 | 74.44 | 45.80 | 27.78 | 38.35 |
| Llama-3.1-8B | **ReQueR** | **30.94** | **21.81** | **14.02** | **71.76** | **75.82** | **54.00** | **31.31** | **42.81** <sub>(+4.46)</sub> |
| Llama-3.1-70B | CoT | 50.31 | 31.31 | 17.93 | 80.81 | 85.76 | 64.60 | 47.73 | 54.06 |
| Llama-3.1-70B | **ReQueR** | **53.75** | **36.05** | **20.30** | **83.38** | **87.98** | **72.00** | **48.36** | **57.40** <sub>(+3.34)</sub> |
| Qwen2.5-72B-Instruct | CoT | 61.88 | 43.62 | 26.40 | 81.57 | 82.26 | 76.20 | 50.88 | 60.40 |
| Qwen2.5-72B-Instruct | **ReQueR** | **62.50** | **46.44** | **27.37** | **85.43** | **88.12** | **83.00** | **52.02** | **63.55** <sub>(+3.15)</sub> |
| Mixtral-8x7B-Instruct | CoT | 9.69 | 7.12 | 8.63 | 47.29 | 52.12 | 30.60 | 25.76 | 25.89 |
| Mixtral-8x7B-Instruct | **ReQueR** | **18.12** | **10.68** | **9.28** | **57.52** | **59.66** | **33.00** | **31.31** | **31.37** <sub>(+5.48)</sub> |

**Head-to-head vs. strong prompt-optimization baselines** (Qwen3-1.7B Solver):

| Method | GSM-Symbolic | MATH-500 | Omni-MATH | Olym.Bench | GPQA-D | MMLU-Pro | Avg. |
|---|---|---|---|---|---|---|---|
| Zero-shot CoT | 74.62 | 64.20 | 20.84 | 37.83 | 31.31 | 43.80 | 45.43 |
| TextGrad | 76.78 | 72.40 | 20.14 | 34.87 | 31.82 | 44.05 | 46.68 |
| GEPA | 76.95 | 73.60 | 20.62 | 35.12 | 32.05 | 45.32 | 47.28 |
| Re2 | 76.78 | 71.40 | 21.03 | 38.13 | 31.31 | 44.01 | 47.11 |
| RaR | 65.64 | 72.80 | 20.37 | 38.58 | 31.82 | 42.98 | 45.37 |
| **ReQueR (Ours)** | **78.64** | **77.60** | **22.11** | **39.02** | **32.83** | **46.09** | **49.38** <sub>(+2.10)</sub> |

See the paper for full tables, additional ablations, and case studies.

---

## Contact

For questions or collaboration, please reach out to **Yixiao Zhou** (`12421181@zju.edu.cn`), or open an [issue](../../issues) in this repository.

---

## Acknowledgements

ReQueR is built on top of the excellent [verl](https://github.com/volcengine/verl) RL training framework, [vLLM](https://github.com/vllm-project/vllm) for scalable rollout generation, and [lighteval](https://github.com/huggingface/lighteval) for standardized evaluation. We thank the authors of GSM8K, MATH, OlympiadBench, Omni-MATH, GPQA, and MMLU-Pro for open-sourcing their benchmarks.

## License

This project is released under the [Apache License 2.0](LICENSE).
