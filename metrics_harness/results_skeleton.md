# Results & Experiments — Report Skeleton

**Paper section owner:** Person B (Aneesh). **Status:** skeleton; numbers and
figures regenerate from `circuit_pipeline/results/summary_B_final.json` and
`metrics_harness/figures/` whenever the notebook re-runs.

**Consumers:** Person A writes §§ Intro/Methodology, Person C writes §§ Related
Work, Person B writes §§ Experiments + Results + Discussion. This file maps each
claim → supporting figure/table/file so co-authors can assemble the paper
without re-reading the notebook.

---

## §4 Experimental Setup

**Goal:** fix the evaluation frame so reviewers understand what's being compared.

- **Model.** `Qwen/Qwen3-0.6B` (0.6B params, 28 transformer blocks). Accepted
  contingency: LLaMA-3-8B scale-up is deferred. Cite this in Limitations.
- **Dataset.** 50 CounterFact samples (indices 0–49; shared across all methods).
  Reference: `baselines/rome_full_metrics.json["samples"]`.
- **Methods.** Five methods: ROME, MEMIT, OurMethod at 1/5/10 steps. MEND
  declared infeasible (no Qwen3 hypernetwork checkpoint). ITI / LoRA / AlphaEdit
  accepted as optional per sprint-doc §5 contingency.
- **Metrics.** Table in §4.1 below, all with formal definitions and sources.
  Reference: `circuit_pipeline/results/results_schema.json` v1.1.
- **Reproducibility.** One harness notebook (`Notebook_B_Evaluation_Harness.ipynb`)
  with a `DRY_RUN` toggle. Baselines: Person C live inference. OurMethod:
  structural synthetic OE in DRY_RUN=True; GPU live in DRY_RUN=False.

### §4.1 Metric definitions table

Claim: we report six evaluation metrics across two over-extinction signals.

| Metric | Formula | Range | Good = | Source |
|--------|---------|-------|--------|--------|
| `edit_success` | target_new ∈ gen_after | 0–1 | high | live |
| `over_extinction` (OE_bleed) | fraction of neighbors where target_new appears after edit | 0–1 | low | live or synthetic |
| `oe_damage` | of neighbors base model got right, fraction broken by edit | 0–1 | low | live only |
| `neighborhood_preservation` | fraction of neighbors still producing target_true | 0–1 | high | live only |
| `paraphrase_success` | fraction of paraphrase prompts producing target_new | 0–1 | context | live |
| `locality_drop` | `mmlu_before − mmlu_after` | any | ≈0 | not run |

**Over-extinction framing (important for abstract/intro):** The proposal
calls out *collateral refusal for associated subjects* as the failure mode.
That is `collateral_damage_rate = 1 − neighborhood_preservation`.
OE_bleed is a complementary signal measuring whether the **new** answer
*spreads* into neighbors. Both are reported; the former is the primary
claim.

---

## §5 Results

### §5.1 Main comparison

**Claim.** On Qwen3-0.6B / 50 CounterFact samples, one-shot editors
(ROME, MEMIT) achieve high edit success but **catastrophic collateral
damage** to neighborhood facts. Circuit-targeted multi-step editing
achieves comparable edit success with structurally lower OE_bleed.

- **Figure 1** (`figures/fig1_oe_vs_steps.png`) — grouped-bar OE_bleed
  across all five methods, with edit-success overlay on the secondary axis.
- **Figure 4** (`figures/fig4_method_ranking.png`) — the per-metric
  best-method table, green-highlighted. Paper-table-ready.
- **Data source** — `summary_B_final.json["methods"]` and
  `circuit_pipeline/results/results_final.csv` (250 rows).
- **Key numbers** (live, for abstract):
  - ROME  : edit_success 0.987, OE_bleed 0.693, preservation 0.007
  - MEMIT : edit_success 0.977, OE_bleed 0.513, preservation 0.067
  - OurMethod 10-step (synthetic OE): edit_success 0.998, OE_bleed ~0.087,
    preservation *TBD (DRY_RUN=False required)*.

### §5.2 Step-count ablation (Experiment 1 in proposal)

**Claim.** Edit success rises steeply from 1-step (≈46%) to 5-step
(≈98%) and saturates by 10-step (≈100%). Synthetic OE tracks this
curve, saturating around 0.087.

- **Data source** — `summary_B_final.json["methods"]["OurMethod_{1,5,10}step"]`.
- **Missing.** 20-step is specified by the proposal but not delivered
  (accepted contingency — GPU budget). Note in Limitations.

### §5.3 Pareto analysis

**Claim.** The specificity–generalization tradeoff is visible as a
Pareto frontier across the five methods. OurMethod variants dominate
the baselines in (low OE_bleed, high edit_success) space (in DRY_RUN=True;
live redraw pending).

- **Figure 3** (`figures/fig3_pareto.png`) — Pareto scatter with
  frontier line and ★-marked non-dominated methods.
- **Data source** — `summary_B_final.json["pareto_methods"]`.
- **Interpretation**: with synthetic OE in DRY_RUN=True, the three
  OurMethod variants are Pareto-optimal because their OE is structurally
  capped. A live redraw may move ROME or MEMIT onto the frontier; re-run
  once DRY_RUN=False data lands.

### §5.4 Circuit scope vs OE (Experiment 2 prep)

**Claim.** Larger MLP circuits correlate with more OE_bleed — **as a
structural prediction** from the synthetic formula.

- **Figure 2** (`figures/fig2_mlp_scope_vs_oe.png`) — n_mlp × OE_bleed
  scatter by step count, with Spearman ρ annotated.
- **Data source** — `summary_B_final.json["spearman_mlp_scope_vs_oe"]`.
- **Caveat (must go in caption):** the synthetic OE formula uses n_mlp
  directly, so ρ is tautological in DRY_RUN=True. Re-run with
  DRY_RUN=False before making this an empirical claim.
- **Proposal link** — Experiment 2 of the proposal specifies
  ACDC vs ROME-trace vs random-circuit comparison at a fixed step count.
  That experiment is Person A's next deliverable.

### §5.5 TruthfulQA hold-out

**Claim.** Editing CounterFact samples does not appreciably affect
TruthfulQA performance, per two GPU-free signals.

- **`tqa_oe_projected_upper_bound`** — per-method conservative projection
  as 0.8 × CounterFact OE_bleed. Placeholder upper bound.
- **`tqa_exposure_rate`** — fraction of TruthfulQA items whose
  question/reference text lexically overlaps with any edit's target_new.
  Honest GPU-free upper bound on at-risk items, NOT a direct OE
  measurement.
- **Data source** — `summary_B_final.json["tqa_oe_projected_upper_bound"]`
  and `["tqa_exposure_rate"]`.
- **Caveat**: both are placeholders; live TQA OE requires DRY_RUN=False
  and is a natural follow-up for the final report.

---

## §6 Discussion

### §6.1 Why ROME's specificity ≠ good behavior

Paragraph: ROME's apparent specificity (lower OE_bleed than our 10-step
method on raw numbers) is misleading because its **collateral damage
rate is 0.993** — it breaks 99% of neighborhood facts while leaving the
new target contained to the edit prompt. This is exactly what the
proposal calls "over-extinction: collateral refusal for associated
subjects." Our circuit-targeted approach aims to reduce that collateral
damage at some OE_bleed cost.

### §6.2 Circuit targeting hypothesis

Paragraph: Person A's ACDC pass selected MLP-only circuits (0 attention
heads) across all 50 samples on Qwen3-0.6B. This may be a model-size
artifact; LLaMA-3-8B is expected to show more distributed circuits. The
current data supports the qualitative hypothesis (smaller circuit →
more specific edit) but does not yet empirically validate it because
OurMethod OE is synthetic.

### §6.3 Limitations

Copy directly from `summary_B_final.json["todos_before_paper"]`:
1. **OurMethod live OE + collateral damage** requires GPU live inference
   (DRY_RUN=False on T4 or better). Without this the central comparison
   claim is structural, not empirical.
2. **KL-divergence stabilizer (proposal Step 4) disabled** — L4 GPU
   (22 GB) OOMs on the extra forward pass. Accepted contingency per
   sprint-doc §5.
3. **20-step ablation missing** (proposal 1/5/10/20; Person A ran 1/5/10).
4. **MEND, ITI, LoRA, AlphaEdit baselines not run** (MEND infeasible;
   rest accepted as optional per contingency).
5. **Single model, single dataset.** Qwen3-0.6B + 50 CounterFact is a
   proof-of-concept. Scale to LLaMA-3-8B and the other five datasets
   (HaluEval, TOFU, MMLU, StereoSet, TruthfulQA live) is future work.

---

## §7 File index (what to cite where)

| Paper section | File | Use |
|---------------|------|-----|
| §4 Setup | `circuit_pipeline/results/results_schema.json` | Cite as data schema v1.1 |
| §5.1 Main | `fig1_oe_vs_steps.png`, `fig4_method_ranking.png` | Main comparison figure + table |
| §5.2 Ablation | `summary_B_final.json["methods"]` | Step-count numbers |
| §5.3 Pareto | `fig3_pareto.png`, `summary_B_final.json["pareto_methods"]` | Pareto plot + non-dominated list |
| §5.4 Circuit | `fig2_mlp_scope_vs_oe.png`, `summary_B_final.json["spearman_mlp_scope_vs_oe"]` | Scatter + ρ |
| §5.5 TQA | `summary_B_final.json["tqa_oe_projected_upper_bound"]`, `["tqa_exposure_rate"]` | Hold-out signals |
| §6.3 Limits | `summary_B_final.json["todos_before_paper"]` | Copy verbatim |

All files regenerate when `Notebook_B_Evaluation_Harness.ipynb` runs end-to-end.
