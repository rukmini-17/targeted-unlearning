"""
Unit tests for the harness metric functions.
Run with: uv run --with pytest pytest metrics_harness/tests/test_metrics.py
"""

import sys
from pathlib import Path

# Put metrics_harness/ on sys.path so harness_functions is importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pytest
from harness_functions import (
    baseline_container,
    compute_edit_success,
    compute_locality_drop,
    compute_neighborhood_preservation,
    compute_over_extinction_bleed,
    compute_over_extinction_damage,
    compute_paraphrase_success,
    compute_pareto_frontier,
    run_harness,
)


# ── compute_edit_success ───────────────────────────────────────────────────────
def test_edit_success_hit():
    assert compute_edit_success("The answer is French.", "French") == 1.0


def test_edit_success_miss():
    assert compute_edit_success("The answer is English.", "French") == 0.0


def test_edit_success_case_insensitive():
    assert compute_edit_success("FRENCH", "french") == 1.0


# ── compute_over_extinction_bleed ──────────────────────────────────────────────
def test_oe_bleed_partial():
    nbhd = [{"bleed_new": True}, {"bleed_new": False}, {"bleed_new": True}]
    assert abs(compute_over_extinction_bleed(nbhd) - 2 / 3) < 1e-9


def test_oe_bleed_empty():
    assert compute_over_extinction_bleed([]) == 0.0


def test_oe_bleed_all_zero():
    nbhd = [{"bleed_new": False}] * 5
    assert compute_over_extinction_bleed(nbhd) == 0.0


def test_oe_bleed_all_one():
    nbhd = [{"bleed_new": True}] * 5
    assert compute_over_extinction_bleed(nbhd) == 1.0


def test_oe_bleed_substring_fallback():
    nbhd = [
        {"gen_after": "Sweden is the new location", "target_new": "Sweden"},
        {"gen_after": "Somewhere else entirely", "target_new": "Sweden"},
    ]
    assert compute_over_extinction_bleed(nbhd) == 0.5


# ── compute_over_extinction_damage ─────────────────────────────────────────────
def test_oe_damage_half():
    pre = [
        {"correct_before": True},
        {"correct_before": True},
        {"correct_before": False},
    ]
    post = [
        {"correct_after": False},
        {"correct_after": True},
        {"correct_after": True},
    ]
    # 2 correct pre; 1 damaged → 0.5
    assert abs(compute_over_extinction_damage(pre, post) - 0.5) < 1e-9


def test_oe_damage_no_baseline_correct():
    # No neighbor was correct pre-edit → damage is 0.0 by convention
    pre = [{"correct_before": False}] * 3
    post = [{"correct_after": False}] * 3
    assert compute_over_extinction_damage(pre, post) == 0.0


def test_oe_damage_all_broken():
    pre = [{"correct_before": True}] * 4
    post = [{"correct_after": False}] * 4
    assert compute_over_extinction_damage(pre, post) == 1.0


def test_oe_damage_no_breakage():
    pre = [{"correct_before": True}] * 4
    post = [{"correct_after": True}] * 4
    assert compute_over_extinction_damage(pre, post) == 0.0


def test_oe_damage_length_mismatch_raises():
    with pytest.raises(AssertionError):
        compute_over_extinction_damage(
            [{"correct_before": True}],
            [{"correct_after": False}, {"correct_after": False}],
        )


# ── compute_paraphrase_success ─────────────────────────────────────────────────
def test_paraphrase_success_partial():
    para = ["She speaks French.", "Her language is German.", "French is her tongue."]
    assert abs(compute_paraphrase_success(para, "French") - 2 / 3) < 1e-9


def test_paraphrase_success_empty():
    assert compute_paraphrase_success([], "anything") == 0.0


# ── compute_neighborhood_preservation ──────────────────────────────────────────
def test_neighborhood_preservation_partial():
    post = [{"correct_after": True}, {"correct_after": False}, {"correct_after": True}]
    assert abs(compute_neighborhood_preservation(post) - 2 / 3) < 1e-9


def test_neighborhood_preservation_empty():
    assert compute_neighborhood_preservation([]) == 0.0


# ── compute_locality_drop ──────────────────────────────────────────────────────
def test_locality_drop_positive():
    # Floating-point safe: 0.65 - 0.60 ≈ 0.05000000000000004 in IEEE 754
    assert abs(compute_locality_drop(0.65, 0.60) - 0.05) < 1e-9


def test_locality_drop_none_pass_through():
    assert compute_locality_drop(None, 0.6) is None
    assert compute_locality_drop(0.6, None) is None


def test_locality_drop_negative_means_improvement():
    assert compute_locality_drop(0.5, 0.6) == pytest.approx(-0.1)


# ── run_harness orchestrator ───────────────────────────────────────────────────
def test_run_harness_shape():
    row = run_harness(
        method="test",
        model_id="model",
        idx=0,
        n_steps=1,
        gen_after="French is the answer",
        target_new="French",
        paraphrase_gens=["Speaks French.", "Speaks German."],
        neighborhood_results_post=[{"bleed_new": True}, {"bleed_new": False}],
    )
    expected_keys = {
        "method", "model", "idx", "n_steps",
        "edit_success", "paraphrase_success", "over_extinction",
        "oe_damage", "neighborhood_preservation", "locality_drop", "kl_final",
    }
    assert set(row.keys()) == expected_keys
    assert row["edit_success"] == 1.0
    assert row["paraphrase_success"] == 0.5
    assert abs(row["over_extinction"] - 0.5) < 1e-6
    # oe_damage None because we didn't pass pre-edit results
    assert row["oe_damage"] is None


# ── baseline_container ─────────────────────────────────────────────────────────
def test_container_samples():
    assert baseline_container({"samples": [1, 2]}) == [1, 2]


def test_container_rows():
    assert baseline_container({"rows": [1, 2]}) == [1, 2]


def test_container_missing():
    assert baseline_container({}) == []


def test_container_samples_wins_over_rows():
    # If both present (edge case), samples wins — documented behavior.
    assert baseline_container({"samples": ["s"], "rows": ["r"]}) == ["s"]


# ── compute_pareto_frontier ────────────────────────────────────────────────────
def test_pareto_single_point_is_optimal():
    assert compute_pareto_frontier([("A", 0.5, 0.5)]) == ["A"]


def test_pareto_all_dominated_by_one():
    # B dominates all others: lowest x, highest y.
    points = [("A", 0.5, 0.5), ("B", 0.1, 0.9), ("C", 0.4, 0.6)]
    assert compute_pareto_frontier(points) == ["B"]


def test_pareto_two_nondominated():
    # A: low OE but low edit_success. B: high OE but high edit_success.
    # Neither dominates the other.
    points = [("A", 0.1, 0.5), ("B", 0.5, 0.9)]
    assert set(compute_pareto_frontier(points)) == {"A", "B"}


def test_pareto_skips_none_coords():
    # Methods with missing metrics should not participate.
    points = [("A", None, None), ("B", 0.1, 0.9), ("C", 0.5, 0.5)]
    assert compute_pareto_frontier(points) == ["B"]


def test_pareto_tie_stays_nondominated():
    # Two points with identical coords: neither strictly dominates.
    points = [("A", 0.1, 0.9), ("B", 0.1, 0.9)]
    assert set(compute_pareto_frontier(points)) == {"A", "B"}
