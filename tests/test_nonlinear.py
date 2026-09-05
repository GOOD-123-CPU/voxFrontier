"""Tests for nonlinear analysis on data with a *known planted* U-shape."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_u_data(n: int = 500, valley: float = 300.0, seed: int = 5):
    """Planted *U-shape* (valley): efficiency is lowest at x=valley."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(130, 430, size=n)
    y = 0.55 + 0.30 * ((x - valley) / 150.0) ** 2 + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {"mean_f0": x, "bcc_efficiency": np.clip(y, 0.05, 0.99),
         "stream_hours": rng.uniform(1, 10, n),
         "price": rng.uniform(10, 200, n),
         "viewers": rng.uniform(100, 5000, n)}
    )


def test_quadratic_detects_planted_ushape():
    from voxfrontier.nonlinear.analysis import quadratic_tests

    df = _make_u_data()
    out = quadratic_tests(df, ["mean_f0"], "bcc_efficiency")
    row = out.iloc[0]
    assert row["shape"] == "u_shape"
    assert row["b2_quadratic"] > 0   # U-shape opens upward (positive curvature)
    # the planted valley is 300; estimation should land nearby
    assert abs(row["extremum_raw"] - 300.0) < 25.0


def test_threshold_recovers_regime_change():
    from voxfrontier.nonlinear.analysis import threshold_regression

    rng = np.random.default_rng(6)
    n = 600
    x = rng.uniform(0, 20, size=n)
    y = np.where(x <= 10, 0.4 + 0.02 * x, 0.4 + 0.02 * 10 - 0.015 * (x - 10))
    y = np.clip(y + rng.normal(0, 0.03, n), 0.05, 0.99)
    df = pd.DataFrame(
        {"hnr": x, "bcc_efficiency": y,
         "stream_hours": np.ones(n), "price": np.ones(n),
         "viewers": np.ones(n)}
    )
    out = threshold_regression(
        df, ["hnr"], ["stream_hours", "price", "viewers"], "bcc_efficiency"
    )
    row = out.iloc[0]
    assert 8.5 < row["threshold_gamma"] < 11.5
    assert row["beta_above"] < row["beta_below"]


def test_subgroup_effects_basic(synth_with_eff):
    from voxfrontier.nonlinear.analysis import subgroup_effects

    out = subgroup_effects(
        synth_with_eff, ["mean_f0"], "product_type", "bcc_efficiency"
    )
    assert set(out["group_col"]) == {"product_type"}
    assert len(out) >= 1
    assert (out["p_value"] >= 0).all() and (out["p_value"] <= 1).all()


def test_quantile_bands_returns_columns(synth_with_eff):
    from voxfrontier.nonlinear.analysis import quantile_band_effects

    out = quantile_band_effects(synth_with_eff, ["mean_f0"], "bcc_efficiency")
    assert "beta_q25" in out.columns
    assert "beta_q90" in out.columns
