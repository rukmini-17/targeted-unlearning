# Targeted Unlearning via Reconsolidation-Inspired Protocol

**CS 590NN Neural Networks — Spring 2026**

Team:
- **Amogh Guthur** — Circuit pipeline, KL-stabilized edit protocol, scale-sensitivity arm
- **Rukmini Nazre** — ROME and MEMIT baselines
- **Aneesh Ashwinikumar Sathe** — Evaluation harness (OE, neighborhood bleed)

## Overview

Investigates whether **gradient-based edits targeted at causally-identified
circuits**, stabilized by a KL constraint on a neutral anchor set, can produce
more specific knowledge edits than whole-network editing methods like ROME
and MEMIT.

## Structure

```
.
├── circuit_pipeline/        # Amogh — circuit discovery + KL-stabilized edits
├── baselines/               # Rukmini — ROME and MEMIT on Qwen3-0.6B and GPT-2-XL
└── metrics_harness/         # Aneesh — OE, neighborhood bleed, paraphrase evaluation
```

Each subdirectory has its own README.

## Key Results Summary

- **KL-stabilized 4-step reconsolidation** protocol achieves ≥99% edit success
  with drift recovery on 32 neutral anchor prompts (Qwen3-0.6B and GPT-2-XL).
- **Localization signal matters less than circuit size** for KL-stabilized editing
  at 0.6B scale, with ROME-trace and size-matched Random being competitive.
- **Scale-sensitivity finding**: ACDC circuit discovery outperforms ROME-trace
  on KL drift at 1.5B (GPT-2-XL) but not at 0.6B (Qwen3), consistent with ACDC's
  original validation regime.

See `circuit_pipeline/README.md` for details.

## Reproduction

```bash
git clone https://github.com/rukmini-17/targeted-unlearning.git
cd targeted-unlearning
uv sync  # or: pip install -e .
```

Notebooks are Colab-ready. Recommended GPU: H100 (80 GB) with High-RAM.

## Dataset

CounterFact from Meng et al. 2022 (ROME), downloaded on first run from
`https://rome.baulab.info/data/dsets/counterfact.json`.
