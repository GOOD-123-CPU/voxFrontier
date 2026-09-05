"""End-to-end CLI test: `voxfrontier run-all` on a small synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_small_config(tmp_path: Path) -> Path:
    cfg = {
        "seed": 123,
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "output"),
            "figure_dir": str(tmp_path / "figures"),
        },
        "data": {"n_units": 60},
        "dea": {"backend": "scipy"},
        "attribution": {"shapley_max_vars": 10, "rf_trees": 50, "use_shap": False},
        "plot": {"dpi": 72, "language": "en"},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def test_cli_run_all(tmp_path):
    from voxfrontier.cli import main

    cfg_path = _write_small_config(tmp_path)
    rc = main(["--config", str(cfg_path), "run-all"])
    assert rc == 0

    out_dir = tmp_path / "output"
    expected = [
        "dea_model_summary.csv",
        "dea_efficiency_by_unit.csv",
        "shapley_decomposition.csv",
        "forest_importance.csv",
        "tobit_ols.csv",
        "tobit_mle.csv",
        "mediation_analysis.csv",
        "quadratic_tests.csv",
        "threshold_regression.csv",
        "quantile_bands.csv",
        "subgroup_effects.csv",
        "dml_effects.csv",
        "fwl_ols_benchmark.csv",
        "scenario_simulation.csv",
        "unit_diagnostics.csv",
        "run_summary.json",
        "manifest.json",
    ]
    for name in expected:
        assert (out_dir / name).exists(), f"missing output {name}"

    fig_dir = tmp_path / "figures"
    for name in ("dea_efficiency.png", "shapley_decomposition.png",
                 "quadratic_shapes.png", "scenarios.png", "dashboard.png"):
        assert (fig_dir / name).exists(), f"missing figure {name}"


def test_cli_synth(tmp_path):
    from voxfrontier.cli import main

    cfg_path = _write_small_config(tmp_path)
    rc = main(["--config", str(cfg_path), "synth"])
    assert rc == 0
    assert (tmp_path / "data" / "synthetic_sessions.csv").exists()
