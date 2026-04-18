# Metrics Harness — Person B (Aneesh)

Evaluation harness for *Reconsolidation-Inspired Targeted Unlearning in LLMs*
(CS 590NN). Takes Person A's circuit-pipeline outputs and Person C's baseline
metrics JSONs, normalizes them to a shared schema (`results_schema.json` v1.1),
and emits a unified 250-row CSV + summary JSON + four paper-ready figures.

---

## Layout

```
metrics_harness/
├── Notebook_B_Evaluation_Harness.ipynb   # the harness; run this
├── harness_functions.py                   # shared metric + Pareto fns (imported by the notebook and tests)
├── results_skeleton.md                    # Experiments+Results paper-section outline
├── README.md                              # this file
├── figures/                               # 4 PNGs, generated on notebook run
│   ├── fig1_oe_vs_steps.png
│   ├── fig2_mlp_scope_vs_oe.png
│   ├── fig3_pareto.png
│   └── fig4_method_ranking.png
└── tests/
    ├── test_metrics.py                    # 7 metric fns + Pareto + ingest (unit tests)
    ├── test_schema.py                     # CSV/summary conform to schema v1.1
    └── test_e2e.py                        # nbclient smoke test (executes the full notebook)
```

Consumes:
- `circuit_pipeline/results/week3_harness_output.json` (Person A, 150 rows, null OE)
- `circuit_pipeline/results/week3_ablation.json` (Person A, 1/5/10-step metrics)
- `circuit_pipeline/results/week2_circuit_log.json` (Person A, circuit sizes)
- `baselines/rome_full_metrics.json` (Person C, live ROME, 50 per-sample rows)
- `baselines/memit_full_metrics.json` (Person C, live MEMIT, 50 per-sample rows)

Produces:
- `circuit_pipeline/results/results_final.csv` (250 rows, schema v1.1)
- `circuit_pipeline/results/summary_B_final.json` (paper summary)
- `circuit_pipeline/results/week3_harness_output_filled.json` (drop-in replacement for Person A's null-OE file)
- 4 figures (see layout above)

---

## Running it

### Prerequisites

All deps land via `uv run`; no manual env setup needed.

```bash
# core notebook deps
uv run --with nbclient --with nbformat --with numpy --with matplotlib \
       --with seaborn --with scipy --with datasets \
       python -c "import numpy; print('ok')"
```

For `DRY_RUN=False` (live GPU inference for OurMethod) also:

```bash
uv run --with transformer_lens --with transformers --with accelerate --with einops python -c "import transformer_lens"
```

### Fast path (no GPU) — `DRY_RUN=True`

1. Open `Notebook_B_Evaluation_Harness.ipynb`.
2. Confirm `DRY_RUN = True` in Section 1.
3. **Run All.** Completes on Mac CPU in <60s.

OurMethod OE values are **synthetic structural approximations**; every OurMethod
row is tagged `oe_source = "synthetic_approx"`. ROME and MEMIT rows come from
Person C's live inference and are tagged `oe_source = "baseline_live"`.

### GPU path — `DRY_RUN=False`

1. Upload the repo to Colab (or equivalent T4+).
2. Set `DRY_RUN = False` and `DEVICE = "cuda"` in Section 1.
3. Ensure transformer_lens can load `Qwen/Qwen3-0.6B`.
4. Run all cells. Expected runtime ~45 min on T4.

OurMethod rows become `oe_source = "live_inference"` and `oe_damage` /
`neighborhood_preservation` / `paraphrase_success` fill in (previously
null in DRY_RUN=True).

**Accepted contingency:** KL penalty (proposal Step 4) stays off even on
DRY_RUN=False — the L4 (22 GB) OOMs with the extra forward pass. The
`training_loss_final` column carries contrastive loss only. See sprint-doc
§5 for the formal contingency.

---

## Schema v1.1 (summary)

Source of truth: `circuit_pipeline/results/results_schema.json`.

| Field | Type | Notes |
|-------|------|-------|
| `method` | str | `ROME`, `MEMIT`, `OurMethod_{1,5,10}step` |
| `idx`, `n_steps` | int | CounterFact sample id (0-49) and step count |
| `edit_success` | 0–1 | target_new appears in gen_after |
| `over_extinction` | 0–1 | OE_bleed: fraction of neighbors where target_new bleeds in |
| `oe_damage` | 0–1 nullable | collateral damage; needs live pre-edit pass |
| `neighborhood_preservation` | 0–1 nullable | fraction of neighbors still correct |
| `paraphrase_success` | 0–1 nullable | context-dependent |
| `locality_drop` | float nullable | MMLU accuracy drop |
| `training_loss_final` | float nullable | **renamed from v1.0 `kl_final`**; KL is disabled per contingency so this carries contrastive loss |
| `n_mlp` | int nullable | circuit size from Person A; null for baselines |
| `oe_source` | enum | `synthetic_approx` / `live_inference` / `baseline_live` |

**v1.0 → v1.1 migration:** `kl_final` consumers should read
`training_loss_final` instead. The rename is documented in
`results_schema.json["changelog"]` for traceability.

---

## Adding a new baseline (e.g. Person C's ITI / LoRA / AlphaEdit)

Person C's baseline JSONs (`rome_full_metrics.json`, `memit_full_metrics.json`)
use either `"samples"` or `"rows"` as the per-sample container; the harness's
`baseline_container()` helper normalizes both. To add a new baseline:

1. Write the metrics JSON to `baselines/{METHOD}_full_metrics.json` with the
   following minimum shape:
   ```json
   {
     "method": "ITI",
     "model": "Qwen/Qwen3-0.6B",
     "dataset": "CounterFact",
     "n_samples": 50,
     "metrics": { "edit_success_rate": ..., "over_extinction": ..., ... },
     "samples": [  // or "rows": [...]
       { "idx": 0, "edit_success": ..., "over_extinction": ..., ... }
     ]
   }
   ```
2. Add one line to Section 1 (cell 2) of the notebook:
   ```python
   ITI_JSON = os.path.join(BASELINES_DIR, "iti_full_metrics.json")
   ```
3. Add to the ingest loop in cell 8:
   ```python
   with open(ITI_JSON) as f: iti_data = json.load(f)
   baseline_samples["ITI"] = baseline_container(iti_data)
   ```
4. Re-run the notebook. CSV grows from 250 → 300 rows; fig1/fig3/fig4 pick up
   the new method automatically (they iterate `METHODS_ORDERED`, which you
   also extend by one entry).
5. Update the expected row count in `metrics_harness/tests/test_schema.py`
   (`test_csv_has_250_rows` → 300).

---

## Running the test suite

All tests (unit + schema + E2E):

```bash
uv run --with pytest --with nbclient --with nbformat --with numpy \
       --with matplotlib --with seaborn --with scipy --with datasets \
       python -m pytest metrics_harness/tests/ -v
```

- `test_metrics.py` — 30 unit tests over the 7 metric functions, `baseline_container`, and `compute_pareto_frontier`. Runs in <1s.
- `test_schema.py` — 18 tests. Asserts CSV conforms to schema, ROME numbers are the real live values (guards against the stale-stub bug), summary has `pareto_methods` + `tqa_exposure_rate`.
- `test_e2e.py` — 5 tests. Executes the full notebook via `nbclient`, asserts 4 figures + 250-row CSV + real-baseline summary. Takes ~30s.

---

## How to interpret the results

There is an inherent **specificity–generalization tradeoff** in knowledge
editing. The four-figure view:

- **Figure 1 (bar + overlay)** — OE_bleed per method with edit_success on
  secondary axis. Baselines (ROME, MEMIT) sit at ~50–70% OE_bleed; OurMethod
  3 variants sit at 4–9% (synthetic). Baselines win on edit success only
  marginally.

- **Figure 2 (scatter)** — n_mlp × OE_bleed, coloured by step count. In
  DRY_RUN=True this is a **structural prediction** (the synthetic formula
  uses n_mlp directly) — caption says so explicitly. DRY_RUN=False redraws it
  as an empirical result.

- **Figure 3 (Pareto)** — OE_bleed (minimize) vs edit_success (maximize).
  Pareto-optimal methods are ★-marked and connected by a dashed frontier.
  `summary_B_final.json["pareto_methods"]` is the authoritative list.

- **Figure 4 (ranking table)** — per-metric best-in-column, green-highlighted.
  Paper-table-ready (copy the PNG into LaTeX or reproduce as a \\begin{table}
  verbatim from the notebook output).

**Proposal framing (important for the paper).** The proposal defines
over-extinction as *collateral refusal for associated subjects* — i.e.
collateral damage to neighborhood facts. Per Person C's live data,
ROME's `1 − neighborhood_preservation = 0.993`. That is the real
over-extinction number. OE_bleed (the primary numeric metric) is a
complementary signal measuring whether the new target *spreads* into
neighbors. Both matter; paper framing in `results_skeleton.md` keeps them
distinct.

---

## Known limitations (all accepted per sprint-doc §5 contingencies)

1. **OurMethod OE still synthetic** in DRY_RUN=True. Paper claim
   *"our method reduces collateral damage vs ROME"* requires DRY_RUN=False.
2. **KL penalty disabled.** L4 GPU (22 GB) OOMs on the proposal's Step 4
   KL-divergence stabilizer. `training_loss_final` carries contrastive loss.
3. **20-step ablation missing.** Proposal specifies 1/5/10/20; Person A ran
   1/5/10 under the same GPU contingency.
4. **MEND infeasible** (no Qwen3 hypernetwork checkpoint). ITI / LoRA /
   AlphaEdit optional per contingency; currently unrun.
5. **Single-model evaluation** on Qwen3-0.6B. LLaMA-3-8B scale-up is future
   work per contingency.

These five items are also emitted in
`summary_B_final.json["todos_before_paper"]` for direct copy-paste into the
paper's Limitations subsection.
