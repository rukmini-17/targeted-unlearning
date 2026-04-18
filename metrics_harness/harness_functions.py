"""
Canonical metric functions for the targeted-unlearning evaluation harness
(Person B, CS 590NN). Extracted from Notebook_B_Evaluation_Harness.ipynb so
they are importable by both the notebook and the pytest suite.

All functions accept the same per-sample data shape the harness uses: a list
of dicts for neighborhood results (see docstrings for accepted keys).
"""

from __future__ import annotations


# ── Core metrics ───────────────────────────────────────────────────────────────
def compute_over_extinction_bleed(neighborhood_results: list[dict]) -> float:
    """
    OE_bleed: fraction of neighborhood prompts where the edited model outputs
    target_new. Accepts either 'bleed_new' bool or ('gen_after', 'target_new')
    substring fallback.
    """
    if not neighborhood_results:
        return 0.0
    bleeds = []
    for r in neighborhood_results:
        if 'bleed_new' in r:
            bleeds.append(bool(r['bleed_new']))
        elif 'gen_after' in r and 'target_new' in r:
            bleeds.append(r['target_new'].lower() in r['gen_after'].lower())
        else:
            bleeds.append(False)
    return float(sum(bleeds)) / len(bleeds)


def compute_over_extinction_damage(
    neighborhood_results_pre: list[dict],
    neighborhood_results_post: list[dict],
) -> float:
    """
    OE_damage: of neighbors the base model answered correctly, fraction broken
    by the edit. Zero baseline-correct neighbors → 0.0 (convention).
    """
    assert len(neighborhood_results_pre) == len(neighborhood_results_post)
    was_correct = []
    for pre in neighborhood_results_pre:
        if 'correct_before' in pre:
            was_correct.append(bool(pre['correct_before']))
        elif 'gen_before' in pre and 'target_true' in pre:
            was_correct.append(pre['target_true'].lower() in pre['gen_before'].lower())
        else:
            was_correct.append(True)
    baseline_correct = sum(was_correct)
    if baseline_correct == 0:
        return 0.0
    damaged = 0
    for i, post in enumerate(neighborhood_results_post):
        if not was_correct[i]:
            continue
        if 'correct_after' in post:
            if not post['correct_after']:
                damaged += 1
        elif 'gen_after' in post and 'target_true' in post:
            if post['target_true'].lower() not in post['gen_after'].lower():
                damaged += 1
    return float(damaged) / baseline_correct


def compute_edit_success(gen_after: str, target_new: str) -> float:
    """Binary: 1.0 if target_new appears in gen_after (case-insensitive)."""
    return 1.0 if target_new.lower() in gen_after.lower() else 0.0


def compute_paraphrase_success(paraphrase_gens: list[str], target_new: str) -> float:
    """Fraction of paraphrase generations containing target_new. Empty → 0.0."""
    if not paraphrase_gens:
        return 0.0
    hits = sum(1 for g in paraphrase_gens if target_new.lower() in g.lower())
    return float(hits) / len(paraphrase_gens)


def compute_neighborhood_preservation(neighborhood_results_post: list[dict]) -> float:
    """
    Fraction of neighbors where the edited model still produces target_true.
    Accepts 'correct_after' bool or ('gen_after', 'target_true') substring
    fallback.
    """
    if not neighborhood_results_post:
        return 0.0
    correct = []
    for r in neighborhood_results_post:
        if 'correct_after' in r:
            correct.append(bool(r['correct_after']))
        elif 'gen_after' in r and 'target_true' in r:
            correct.append(r['target_true'].lower() in r['gen_after'].lower())
        else:
            correct.append(False)
    return float(sum(correct)) / len(correct)


def compute_locality_drop(mmlu_acc_before: float, mmlu_acc_after: float) -> float | None:
    """MMLU accuracy drop (before − after). Positive = degradation. None-safe."""
    if mmlu_acc_before is None or mmlu_acc_after is None:
        return None
    return mmlu_acc_before - mmlu_acc_after


# ── Orchestrator ───────────────────────────────────────────────────────────────
def run_harness(
    method: str,
    model_id: str,
    idx: int,
    n_steps: int,
    gen_after: str,
    target_new: str,
    paraphrase_gens: list[str] | None = None,
    neighborhood_results_pre: list[dict] | None = None,
    neighborhood_results_post: list[dict] | None = None,
    mmlu_acc_before: float | None = None,
    mmlu_acc_after: float | None = None,
    kl_final: float | None = None,
) -> dict:
    """Build a standardized output row from raw inference outputs."""
    edit_suc = compute_edit_success(gen_after, target_new)
    para_suc = compute_paraphrase_success(paraphrase_gens or [], target_new)
    oe_bleed = compute_over_extinction_bleed(neighborhood_results_post or [])
    nbhd_pres = compute_neighborhood_preservation(neighborhood_results_post or [])
    oe_dmg = None
    if neighborhood_results_pre and neighborhood_results_post:
        oe_dmg = compute_over_extinction_damage(
            neighborhood_results_pre, neighborhood_results_post
        )
    loc_drop = compute_locality_drop(mmlu_acc_before, mmlu_acc_after)
    return {
        "method": method,
        "model": model_id,
        "idx": idx,
        "n_steps": n_steps,
        "edit_success": round(edit_suc, 4),
        "paraphrase_success": round(para_suc, 4),
        "over_extinction": round(oe_bleed, 6),
        "oe_damage": round(oe_dmg, 6) if oe_dmg is not None else None,
        "neighborhood_preservation": round(nbhd_pres, 4),
        "locality_drop": round(loc_drop, 4) if loc_drop is not None else None,
        "kl_final": kl_final,
    }


# ── Ingest helpers ─────────────────────────────────────────────────────────────
def baseline_container(d: dict) -> list:
    """
    Baseline JSONs use inconsistent container keys: ROME has 'samples',
    MEMIT has 'rows'. Return whichever is present (empty list if neither).
    """
    return d.get("samples") or d.get("rows") or []


# ── Pareto frontier (2D: minimize x, maximize y) ───────────────────────────────
def compute_pareto_frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """
    points: list of (label, x, y) tuples. x is minimized, y is maximized.
    Returns the list of labels that are NOT strictly dominated by any other
    point. None coordinates are skipped.
    """
    non_dominated: list[str] = []
    for i, (li, xi, yi) in enumerate(points):
        if xi is None or yi is None:
            continue
        dominated = False
        for j, (lj, xj, yj) in enumerate(points):
            if i == j or xj is None or yj is None:
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            non_dominated.append(li)
    return non_dominated
