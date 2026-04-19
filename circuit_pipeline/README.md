# Circuit Pipeline (Person A)

**CS 590NN Neural Networks — Spring 2026 — Amogh Guthur**

Implementation of the **KL-stabilized reconsolidation protocol** for targeted
unlearning, plus circuit-localization comparison experiments across two model
scales (Qwen3-0.6B and GPT-2-XL 1.5B).

## Structure

```
circuit_pipeline/
├── notebooks/                           # Qwen3-0.6B arm
│   ├── Notebook1_v2_1_ACDC.ipynb        # ACDC circuit discovery, threshold=0.3
│   ├── Notebook1_v2_2_ACDC.ipynb        # ACDC circuit discovery, threshold=0.5 (sensitivity)
│   ├── Notebook3_v2_KL_Stabilization.ipynb  # Step-count ablation, KL-stabilized protocol
│   ├── Notebook4_v2_Circuit_Comparison.ipynb  # 3-method comparison (ACDC, ROME-trace, Random)
│   └── gpt2xl/                          # GPT-2-XL 1.5B arm (reduced scope: N=20)
│       ├── Notebook1_gpt2xl_FAST.ipynb
│       ├── Notebook3_gpt2xl_FAST.ipynb
│       └── Notebook4_gpt2xl_FAST.ipynb
│
└── results/
    ├── qwen_circuits/                   # ACDC circuit logs for Qwen at thresholds 0.3 and 0.5
    ├── qwen_kl_ablation/                # Step-count ablation results (Qwen)
    ├── qwen_comparison/                 # 3-method comparison with threshold=0.5 circuits
    ├── qwen_comparison_threshold03/     # 3-method comparison with threshold=0.3 circuits (sensitivity)
    ├── gpt2xl_circuits/                 # ACDC circuit log for GPT-2-XL
    ├── gpt2xl_kl_ablation/              # Step-count ablation results (GPT-2-XL)
    └── gpt2xl_comparison/               # 3-method comparison (GPT-2-XL)
```

## Run Order

1. **Circuit discovery:** `Notebook1_v2_1_ACDC.ipynb` produces `week2_circuit_log_v2.1.json`
2. **(Optional sensitivity):** `Notebook1_v2_2_ACDC.ipynb` produces `week2_circuit_log_v2.2.json` (tighter circuits)
3. **KL ablation:** `Notebook3_v2_KL_Stabilization.ipynb` uploads circuit log, runs step-count ablation
4. **Circuit comparison:** `Notebook4_v2_Circuit_Comparison.ipynb` uploads circuit log, runs 3 methods
5. **GPT-2-XL arm:** same sequence under `notebooks/gpt2xl/` with `_gpt2xl` suffixed outputs

All notebooks are Colab-ready (H100 recommended; A100 works for Qwen).

## Key Findings

### KL-stabilized protocol (Notebook 3 v2)

KL constraint with β=0.1 during 20-step gradient edit achieves high edit success
with controlled drift on a 32-prompt neutral anchor set. Results by step count:

| n_steps | Qwen3-0.6B edit | Qwen3-0.6B KL | GPT-2-XL edit | GPT-2-XL KL |
|---------|-----------------|---------------|---------------|-------------|
| 5       | 0.895           | 20.8          | 0.993         | 20.8        |
| 20      | 0.977           | 9.9           | 0.990         | 4.0         |

KL peaks mid-training (step 6) and recovers under the neutral-anchor constraint.
Recovery is cleaner on larger models.

### Circuit localization comparison (Notebook 4 v2)

3-method comparison (ACDC / ROME-trace causal tracing / size-matched Random)
holding the KL-stabilized edit protocol fixed. Key numbers at n_steps=5:

**Qwen3-0.6B (N=50):**

| Method | Edit success | KL drift | Flip rate |
|--------|--------------|----------|-----------|
| ROME-trace (K=5 MLPs) | 0.953 | 18.3 | 96% |
| Random (size-matched) | 0.953 | 24.8 | 96% |
| ACDC (threshold=0.5)  | 0.790 | 25.1 | 80% |

**GPT-2-XL (N=20):**

| Method | Edit success | KL drift | Flip rate |
|--------|--------------|----------|-----------|
| ROME-trace (K=5 MLPs) | 0.998 | 28.3 | 100% |
| **ACDC (threshold=0.4)** | **0.993** | **20.8** | **100%** |
| Random (size-matched) | 0.947 | 13.4 | 95%  |

### Scale-sensitivity finding

ACDC-based circuit discovery benefits from model scale. On Qwen3-0.6B, ACDC's
threshold-based selection over-includes on weak-signal samples and underperforms
ROME-trace's fixed top-K. On GPT-2-XL, ACDC identifies tight localizations
(median n_mlp=1, typically layer 0) and outperforms ROME-trace on KL drift at
comparable edit success.

## Models

- **Qwen3-0.6B** (`Qwen/Qwen3-0.6B`): 28 layers × 16 heads, primary experimental arm (N=50)
- **GPT-2-XL** (`gpt2-xl`): 48 layers × 25 heads, scale-sensitivity check (N=20, reduced scope)

CounterFact dataset from ROME (Meng et al. 2022), same sample indices across both arms.

## Reproduction

All notebooks run on Google Colab. Required:
- H100 or A100 GPU (High-RAM recommended)
- `transformer_lens`, `transformers`, `accelerate`
- CounterFact dataset downloaded from `https://rome.baulab.info/data/dsets/counterfact.json`

Random seed: `torch.manual_seed(42)` throughout. Sample indices: first N from
CounterFact ordering.

## Author

Amogh Guthur — ACDC pipeline, KL-stabilized edit protocol, circuit comparison,
GPT-2-XL scale-sensitivity arm.

Project team: Rukmini Nazre (ROME/MEMIT baselines), Aneesh Ashwinikumar Sathe
(metrics harness and OE evaluation).
