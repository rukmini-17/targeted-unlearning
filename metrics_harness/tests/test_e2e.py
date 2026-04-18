"""
End-to-end smoke test: execute Notebook_B_Evaluation_Harness.ipynb with
DRY_RUN=True and assert all expected outputs exist. Skips automatically if
nbclient is not installed.

Run with:
  uv run --with pytest --with nbclient --with nbformat --with numpy \\
         --with matplotlib --with seaborn --with scipy --with datasets \\
         pytest metrics_harness/tests/test_e2e.py
"""

import json
from pathlib import Path

import pytest

nbclient  = pytest.importorskip("nbclient")
nbformat  = pytest.importorskip("nbformat")
pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "metrics_harness" / "Notebook_B_Evaluation_Harness.ipynb"
FIGURES = REPO / "metrics_harness" / "figures"
RESULTS = REPO / "circuit_pipeline" / "results"


@pytest.fixture(scope="module")
def executed_notebook():
    """Execute the notebook once per test module."""
    if not NB_PATH.exists():
        pytest.skip(f"Notebook not found at {NB_PATH}")
    nb = nbformat.read(NB_PATH, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_PATH.parent)}},
    )
    client.execute()
    return nb


def test_notebook_runs_cleanly(executed_notebook):
    errors = []
    for i, cell in enumerate(executed_notebook.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename"), out.get("evalue")))
    assert not errors, f"notebook had errors: {errors}"


def test_four_figures_generated(executed_notebook):
    expected = [
        FIGURES / "fig1_oe_vs_steps.png",
        FIGURES / "fig2_mlp_scope_vs_oe.png",
        FIGURES / "fig3_pareto.png",
        FIGURES / "fig4_method_ranking.png",
    ]
    for p in expected:
        assert p.exists(), f"missing figure: {p}"
        assert p.stat().st_size > 1000, f"figure too small: {p}"


def test_csv_is_250_rows(executed_notebook):
    import csv
    csv_path = RESULTS / "results_final.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 250


def test_summary_json_has_real_baselines(executed_notebook):
    summary_path = RESULTS / "summary_B_final.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "1.1"
    # Guard against the stub-ROME bug
    rome_oe = summary["methods"]["ROME"]["over_extinction_bleed"]
    assert rome_oe is not None and rome_oe > 0.5
    # MEMIT must exist
    assert "MEMIT" in summary["methods"]
    # Pareto non-empty
    assert len(summary["pareto_methods"]) >= 1


def test_filled_json_no_null_oe(executed_notebook):
    filled = RESULTS / "week3_harness_output_filled.json"
    data = json.loads(filled.read_text())
    null_oe = [r for r in data["rows"] if r.get("over_extinction") is None]
    assert not null_oe, f"{len(null_oe)} rows still have null over_extinction"
