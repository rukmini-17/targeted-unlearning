"""
Schema validation tests: results_final.csv and summary_B_final.json must match
the declared schema in circuit_pipeline/results/results_schema.json.
"""

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "circuit_pipeline" / "results"
SCHEMA_PATH = RESULTS_DIR / "results_schema.json"
CSV_PATH = RESULTS_DIR / "results_final.csv"
SUMMARY_PATH = RESULTS_DIR / "summary_B_final.json"


def _load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


# ── Schema file itself ─────────────────────────────────────────────────────────
def test_schema_is_v1_1():
    s = _load_schema()
    assert s["version"] == "1.1"


def test_schema_declares_required_fields():
    s = _load_schema()
    required = {
        "method", "model", "idx", "n_steps",
        "edit_success", "over_extinction", "oe_damage",
        "neighborhood_preservation", "paraphrase_success", "locality_drop",
        "n_mlp", "oe_source",
    }
    assert required <= set(s["fields"].keys())


def test_schema_documents_renamed_field():
    s = _load_schema()
    # v1.1 rename: kl_final -> training_loss_final
    assert "training_loss_final" in s["fields"]


# ── CSV conforms to schema ─────────────────────────────────────────────────────
@pytest.fixture
def csv_rows():
    if not CSV_PATH.exists():
        pytest.skip("results_final.csv missing — run the notebook first.")
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def test_csv_exists_and_nonempty(csv_rows):
    assert len(csv_rows) > 0


def test_csv_has_250_rows(csv_rows):
    # 150 OurMethod (50×3) + 50 ROME + 50 MEMIT = 250. Grows when new
    # baselines land — update this constant explicitly when that happens.
    assert len(csv_rows) == 250


def test_csv_columns_subset_of_schema(csv_rows):
    schema_fields = set(_load_schema()["fields"].keys())
    csv_columns = set(csv_rows[0].keys())
    unknown = csv_columns - schema_fields
    assert not unknown, f"CSV columns not in schema: {unknown}"


def test_csv_has_required_columns(csv_rows):
    required = {
        "method", "model", "idx", "n_steps", "edit_success",
        "over_extinction", "oe_source",
    }
    assert required <= set(csv_rows[0].keys())


def test_csv_method_values(csv_rows):
    methods = {r["method"] for r in csv_rows}
    expected = {
        "ROME", "MEMIT",
        "OurMethod_1step", "OurMethod_5step", "OurMethod_10step",
    }
    assert methods == expected


def test_csv_oe_source_enum(csv_rows):
    # Schema declares an enum; CSV should only carry those values.
    allowed = {"synthetic_approx", "live_inference", "baseline_live"}
    observed = {r["oe_source"] for r in csv_rows}
    assert observed <= allowed, f"unexpected oe_source values: {observed - allowed}"


def test_csv_required_non_null(csv_rows):
    # method, model, idx, n_steps, edit_success, over_extinction must always
    # be populated. oe_damage / preservation / paraphrase can legitimately
    # be blank (OurMethod in DRY_RUN=True).
    for r in csv_rows:
        for field in ("method", "model", "idx", "n_steps", "edit_success",
                      "over_extinction", "oe_source"):
            assert r.get(field) not in ("", None), \
                f"row {r['method']}/{r.get('idx')} missing {field}"


def test_csv_edit_success_range(csv_rows):
    for r in csv_rows:
        v = float(r["edit_success"])
        assert 0.0 <= v <= 1.0, f"edit_success out of range: {v}"


def test_csv_over_extinction_range(csv_rows):
    for r in csv_rows:
        v = float(r["over_extinction"])
        assert 0.0 <= v <= 1.0, f"over_extinction out of range: {v}"


# ── Summary JSON conforms ──────────────────────────────────────────────────────
@pytest.fixture
def summary():
    if not SUMMARY_PATH.exists():
        pytest.skip("summary_B_final.json missing — run the notebook first.")
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def test_summary_schema_version(summary):
    assert summary.get("schema_version") == "1.1"


def test_summary_has_real_rome_numbers(summary):
    # The critical bug-prevention test: ROME must NOT show the stub 0.0.
    rome = summary["methods"]["ROME"]
    assert rome["over_extinction_bleed"] is not None
    assert rome["over_extinction_bleed"] > 0.5, \
        "ROME OE_bleed looks like the stub value. Check ROME_JSON path."
    assert rome["edit_success"] > 0.9


def test_summary_has_memit(summary):
    assert "MEMIT" in summary["methods"]
    memit = summary["methods"]["MEMIT"]
    assert memit["edit_success"] > 0.9
    assert 0.4 < memit["over_extinction_bleed"] < 0.7


def test_summary_has_pareto_methods(summary):
    assert "pareto_methods" in summary
    assert isinstance(summary["pareto_methods"], list)
    assert len(summary["pareto_methods"]) >= 1


def test_summary_has_tqa_exposure(summary):
    assert "tqa_exposure_rate" in summary
    tqa = summary["tqa_exposure_rate"]
    # edit_target_token_count must be non-zero when baselines are loaded
    assert tqa.get("edit_target_token_count", 0) > 0


def test_summary_has_todos(summary):
    assert "todos_before_paper" in summary
    assert isinstance(summary["todos_before_paper"], list)
    assert len(summary["todos_before_paper"]) >= 3
