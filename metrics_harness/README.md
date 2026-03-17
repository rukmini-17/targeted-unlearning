# Metrics Harness — Person B (Aneesh)

## What this notebook does

`Notebook_B_Evaluation_Harness.ipynb` is the final evaluation harness for
*Reconsolidation-Inspired Targeted Unlearning in LLMs* (CS 590NN).
It picks up where Person A's circuit pipeline leaves off: Person A's
`week3_harness_output.json` contains 150 result rows (50 CounterFact samples ×
3 step counts) in which the `over_extinction` field is always `null`.
This notebook fills in that field, computes all remaining metrics, runs
a TruthfulQA hold-out, generates the paper figures, and writes the final
summary JSON.

---

## Running it

### Prerequisites

```bash
pip install numpy matplotlib seaborn scipy datasets
# For DRY_RUN=False (GPU path) also:
pip install transformer_lens transformers accelerate einops
```

### Quick (no GPU) — `DRY_RUN = True`

Open the notebook and check that the first cell has:

```python
DRY_RUN = True
```

Then **Run All**. All cells complete on Mac CPU in under a minute.

> ⚠️ OE values will be **synthetic approximations**, not real measurements.
> Formula: `oe_approx = edit_success × (1 − final_p_true) × (n_mlp/28) × 0.12`
>
> Good for exploring the notebook structure and running the figures/analysis cells,
> but **do not use these values in the paper**.
>
> Output files are tagged `"oe_source": "synthetic_approx"` so you can
> distinguish them from real runs.

### Full inference (Colab / GPU) — `DRY_RUN = False`

1. Upload the repo (or mount Drive).
2. Set `DRY_RUN = False` and `DEVICE = "cuda"` in Section 1.
3. Ensure `transformer_lens` can load `Qwen/Qwen3-0.6B`.
4. **Run All** — expected runtime ~45 min on a T4.

Output files will be tagged `"oe_source": "live_inference"`.

---

## Metric definitions

| Metric | What it measures | Range | Good = |
|--------|------------------|-------|--------|
| `edit_success` | Does the edit prompt produce `target_new`? | 0–1 | High |
| `over_extinction` (OE_bleed) | Fraction of neighborhood prompts where `target_new` bleeds in | 0–1 | **Low** |
| `oe_damage` | Of neighbors the base model got right, how many are broken by the edit? | 0–1 | **Low** |
| `neighborhood_preservation` | Fraction of neighbors still correct (producing `target_true`) after edit | 0–1 | High |
| `paraphrase_success` | Does a rephrased prompt produce `target_new`? | 0–1 | Context-dependent |
| `locality_drop` | MMLU accuracy drop post-edit (positive = degradation) | any | Near 0 |

**OE_bleed vs OE_damage:** OE_bleed measures raw bleed into neighbors.
OE_damage is more refined — it only counts neighbors the *base model* had
correct, so it isolates collateral damage from pre-existing errors.
OE_damage requires pre-edit baseline inference and is `null` in `DRY_RUN=True` mode.

---

## Output files

All outputs are written to `circuit_pipeline/results/`.

| File | Description |
|------|-------------|
| `week3_harness_output_filled.json` | Drop-in replacement for `week3_harness_output.json`. Schema is identical; only `over_extinction` is filled in. See note below about `oe_source`. |
| `results_final.csv` | All methods merged (151 rows: 150 OurMethod + 1 ROME aggregate). Columns: `method, model, idx, n_steps, edit_success, over_extinction, oe_damage, neighborhood_preservation, paraphrase_success, locality_drop, kl_final, n_mlp`. |
| `summary_B_final.json` | Paper summary: per-method aggregates, Spearman correlations, TruthfulQA hold-out OE, key finding. |

Figures are written to `metrics_harness/figures/`:

| File | Description |
|------|-------------|
| `fig1_oe_vs_steps.png` | Grouped bar chart — OE_bleed per method, ROME dashed baseline, edit_success on secondary y-axis |
| `fig2_mlp_scope_vs_oe.png` | Scatter x=n_mlp, y=OE_bleed, color=n_steps, Spearman ρ annotated |
| `fig3_pareto.png` | Pareto front x=OE_bleed, y=edit_success, labeled method points |

---

## How to interpret results

There is an inherent **specificity–generalization tradeoff** in knowledge editing:

- **ROME** is the specificity ceiling: `OE_bleed = 0.00`, meaning edits are perfectly
  contained — they never bleed into neighborhood prompts. The cost is zero paraphrase
  generalization (`paraphrase_success = 0.0`).

- **Our circuit-targeted method** sits higher on the Pareto front: more edit steps →
  higher edit success *and* higher OE_bleed, because the optimizer pushes the edit
  deeper into the MLP circuit and it generalizes more broadly.

- **Circuit scope matters:** Spearman ρ between `n_mlp` (circuit size) and OE_bleed
  is ~0.22 at 1 step, rising to ~0.996 at 10 steps — confirming that larger circuits
  produce more bleed when more aggressively edited.

The ideal operating point depends on the application:
- **Unlearning** (privacy/safety): prioritize low OE, accept some edit-success cost → 1-step
- **Knowledge update**: prioritize edit success + paraphrase generalization → 5 or 10 steps

---

## Integration with Person A's pipeline

`week3_harness_output_filled.json` is a **drop-in replacement** for
`week3_harness_output.json`. The JSON schema is identical (same 150 rows, same
field names), with these additions:

- `over_extinction`: now filled (was `null`)
- `oe_source`: `"synthetic_approx"` (DRY_RUN=True) or `"live_inference"` (DRY_RUN=False)
- `n_mlp`: circuit size for that sample, joined from `week2_circuit_log.json`
- `final_p_true`: probability of true answer post-edit, from `week3_ablation.json`
- `oe_damage`, `neighborhood_preservation`, `paraphrase_success`, `locality_drop`:
  populated from live inference when `DRY_RUN=False`; `null` otherwise

Any downstream code that reads `week3_harness_output.json` can switch to
`week3_harness_output_filled.json` without changes.
