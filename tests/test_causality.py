"""Tests for the causality subpackage (DML + inference enhancements)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_dml_data(n: int = 1500, seed: int = 3, true_theta: float = 0.6):
    """Confounded DGP: treatment correlates with a confounder of Y.

    The outcome stays *uncensored* (shifting y positive so no clipping binds)
    — censoring would attenuate the estimate and muddy what we test.
    """
    rng = np.random.default_rng(seed)
    conf = rng.normal(size=n)                      # affects both D and Y
    x2 = rng.normal(size=n)
    d = 0.9 * conf + rng.normal(0, 0.7, n)         # treatment
    y = true_theta * d + 0.8 * conf + 0.3 * x2 + rng.normal(0, 0.25, n)
    y = y - y.min() + 0.05                         # shift: strictly positive
    df = pd.DataFrame(
        {"mean_f0": d, "x2": x2, "confounder": conf,
         "stream_hours": conf, "price": np.abs(x2) + 1,
         "viewers": np.abs(conf) * 100 + 10,
         "bcc_efficiency": y}
    )
    return df, true_theta


def test_dml_recovers_true_effect():
    from voxfrontier.causality.dml import dml_partially_linear

    df, theta = _make_dml_data()
    out = dml_partially_linear(
        df, ["mean_f0"], ["stream_hours", "price", "viewers"],
        "bcc_efficiency", n_folds=5, learner="gbm", seed=7,
        heterogeneity_bands=None,
    )
    row = out.iloc[0]
    # estimate within tolerance; the CI is tight (se~0.008) so allow the
    # CI to miss by less than 3 se around the truth
    assert abs(row["dml_estimate"] - theta) < 0.05
    assert row["ci_low"] - 0.02 < theta < row["ci_high"] + 0.02
    assert row["p_value"] < 0.01


def test_dml_beats_naive_ols_under_confounding():
    """With strong confounding, naive OLS should be further from truth."""
    from voxfrontier.causality.dml import dml_partially_linear, fwl_ols_benchmark

    rng = np.random.default_rng(9)
    n = 1200
    conf = rng.normal(size=n)
    d = 1.2 * conf + rng.normal(0, 0.5, n)         # heavy confounding
    y = 0.5 * d + 1.0 * conf + rng.normal(0, 0.4, n)
    df = pd.DataFrame(
        {"mean_f0": d, "stream_hours": conf, "price": np.ones(n),
         "viewers": np.ones(n), "bcc_efficiency": y}
    )
    ols = fwl_ols_benchmark(df, ["mean_f0"], ["stream_hours", "price", "viewers"])
    dml = dml_partially_linear(
        df, ["mean_f0"], ["stream_hours", "price", "viewers"],
        n_folds=4, learner="gbm", seed=3, heterogeneity_bands=None,
    )
    err_ols = abs(ols.iloc[0]["fwl_ols_estimate"] - 0.5)
    err_dml = abs(dml.iloc[0]["dml_estimate"] - 0.5)
    assert err_dml <= err_ols + 0.05


def test_conformal_interval_coverage():
    from voxfrontier.causality.inference import conformal_interval

    rng = np.random.default_rng(2)
    n = 600
    X = rng.normal(size=(n, 3))
    y = X @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.3, n) + 0.5
    cols = ["f1", "f2", "f3"]
    df = pd.DataFrame(X, columns=cols)
    df["bcc_efficiency"] = y

    res = conformal_interval(df, cols, "bcc_efficiency", coverage=0.9, seed=1)
    q = res["half_width"]
    # on a fresh draw, empirical coverage should be near 90%
    X_new = rng.normal(size=(2000, 3))
    y_new = X_new @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.3, 2000) + 0.5
    df_new = pd.DataFrame(X_new, columns=cols)
    pred = res["model"].predict(res["scaler"].transform(df_new[cols].to_numpy()))
    cover = np.mean(np.abs(y_new - pred) <= q)
    assert 0.80 <= cover <= 0.98


def test_super_efficiency_ranking(synth_with_eff):
    from voxfrontier.causality.inference import super_efficiency_ranking
    from voxfrontier.data.schema import BASIC_MODEL_INPUTS, OUTPUTS

    out = super_efficiency_ranking(
        synth_with_eff.head(80), list(BASIC_MODEL_INPUTS), list(OUTPUTS),
        backend="scipy",
    )
    assert len(out) == 80
    finite = out["super_efficiency"].dropna()
    # finite scores in descending order (ties allowed); NaNs (if any) last
    assert np.all(np.diff(finite.to_numpy()) <= 1e-9)
    # at least one unit must sit on/above the frontier
    assert finite.iloc[0] >= 1.0 - 1e-3


def test_manifest_roundtrip(tmp_path):
    import json

    from voxfrontier.utils.manifest import verify_manifest, write_manifest

    data = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(data, index=False)
    out = tmp_path / "out"           # dedicated output dir: only tables live here
    out.mkdir()
    table = out / "table.csv"
    pd.DataFrame({"b": [4, 5]}).to_csv(table, index=False)

    write_manifest(out, data, seed=1, summary={"k": "v"})
    verdict = verify_manifest(out)
    assert verdict == {"table.csv": True}
    assert verify_manifest(str(out)) == {"table.csv": True}

    # tamper -> verification fails
    pd.DataFrame({"b": [999]}).to_csv(table, index=False)
    assert verify_manifest(out)["table.csv"] is False
    _ = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
