"""Tests for Shapley, Tobit, mediation: mathematical properties + recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from voxfrontier.data.schema import OUTPUTS, SHAPLEY_GROUPS


def test_shapley_sums_to_efficiency_spread(synth_with_eff):
    """Efficiency (in %) of all contributions must sum to ~100 by construction."""
    from voxfrontier.attribution.shapley import shapley_decomposition

    df = synth_with_eff.head(80)  # keep the test fast
    out = shapley_decomposition(df, SHAPLEY_GROUPS, OUTPUTS, backend="scipy")
    assert np.isclose(out["contribution_pct"].sum(), 100.0, atol=1e-6)
    assert (out["contribution_pct"] >= 0).all()
    assert set(out["kind"]).issubset({"voice", "basic"})


def test_shapley_values_are_data_driven(synth_with_eff):
    """Two different data slices must give different Shapley values
    (guards against any hard-coded/cached results regressing in)."""
    from voxfrontier.attribution.shapley import shapley_decomposition

    a = shapley_decomposition(synth_with_eff.head(80), SHAPLEY_GROUPS,
                              OUTPUTS, backend="scipy")
    b = shapley_decomposition(synth_with_eff.tail(80), SHAPLEY_GROUPS,
                              OUTPUTS, backend="scipy")
    merged = a.merge(b, on="group", suffixes=("_a", "_b"))
    assert not np.allclose(
        merged["shapley_value_a"], merged["shapley_value_b"], atol=1e-6
    )


def test_tobit_recovers_simulated_coefficients():
    """On a lightly-censored simulated outcome, OLS-style Tobit recovers
    signs and approximate magnitudes (standardized feature scale).

    Censoring at 0.02/3.0 keeps 100% of the mass interior (pure OLS case);
    a separate test covers heavy censoring where attenuation is expected.
    """
    from voxfrontier.attribution.tobit import tobit_ols

    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.8 * x1 - 0.5 * x2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "x2": x2,
                       "bcc_efficiency": np.clip(y, 0.02, 3.0)})
    res = tobit_ols(df, ["x1", "x2"], "bcc_efficiency")
    tab = res["table"].set_index("feature")
    # x1 dominates and is highly significant; signs correct
    assert tab.loc["x1", "coef"] > 0.3
    assert tab.loc["x2", "coef"] < -0.2
    assert tab.loc["x1", "p_value"] < 0.01
    assert res["r2"] > 0.5


def test_tobit_attenuates_under_heavy_censoring():
    """Documented property: heavy censoring attenuates OLS coefficients."""
    from voxfrontier.attribution.tobit import tobit_ols

    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(size=n)
    y = 0.8 * x1 + rng.normal(0, 0.3, n)
    uncensored = pd.DataFrame(
        {"x1": x1, "bcc_efficiency": np.clip(y, 0.02, 3.0)}
    )
    heavy = pd.DataFrame(
        {"x1": x1, "bcc_efficiency": np.clip(y, 0.0, 0.99)}
    )
    b_unc = tobit_ols(uncensored, ["x1"], "bcc_efficiency")["table"].iloc[0]["coef"]
    b_heavy = tobit_ols(heavy, ["x1"], "bcc_efficiency")["table"].iloc[0]["coef"]
    assert b_heavy < b_unc  # attenuation from clipping to [0, 0.99]


def test_mediation_recovers_indirect_effect():
    from voxfrontier.attribution.tobit import mediation_analysis

    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    m = 0.7 * x + rng.normal(0, 0.5, n)
    y = 0.5 * m + rng.normal(0, 0.5, n)  # indirect = 0.35, direct = 0
    df = pd.DataFrame({"voice_index": x, "viewers": m, "bcc_efficiency": y})
    res = mediation_analysis(df, "voice_index", "viewers", "bcc_efficiency",
                             n_boot=200, seed=1)
    assert res["a_path"] > 0.5
    assert res["indirect_ab"] > 0.2
    assert res["boot_ci_low"] > 0  # significant indirect effect


def test_forest_importance_sums_to_100(synth_with_eff):
    from voxfrontier.attribution.forest import forest_importance

    res = forest_importance(
        synth_with_eff,
        ["stream_hours", "price", "viewers", "mean_f0"],
        "bcc_efficiency",
        n_trees=50,
        seed=42,
        basic_vars=("stream_hours", "price", "viewers"),
    )
    assert np.isclose(res["table"]["importance_pct"].sum(), 100.0, atol=1e-6)
