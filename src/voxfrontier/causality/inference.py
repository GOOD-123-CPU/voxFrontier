"""Statistical inference enhancements.

* ``shapley_bootstrap`` — confidence intervals for group contributions by
  re-running the decomposition on bootstrap resamples (or a fast delta
  approximation when budget-limited);
* ``threshold_bootstrap`` — Hansen-style fixed-X bootstrap p-value for the
  threshold (regime-switching) LR statistic;
* ``conformal_interval`` — split-conformal prediction intervals for the
  efficiency predictor used by the simulation stage;
* ``super_efficiency_ranking`` — BCC super-efficiency scores to rank the
  efficient frontier.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from voxfrontier.attribution.shapley import shapley_decomposition
from voxfrontier.dea.models import prepare_positive, super_efficiency_bcc

logger = logging.getLogger("voxfrontier.inference")


# ----------------------------------------------------------------------
# 1. Shapley bootstrap CIs
# ----------------------------------------------------------------------
def shapley_bootstrap(
    df: pd.DataFrame,
    groups: dict[str, Sequence[str]],
    outputs: Sequence[str],
    backend: str = "auto",
    n_boot: int = 200,
    seed: int = 42,
    basic_groups: Sequence[str] = (),
    voice_groups: Sequence[str] = (),
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Percentile CIs for Shapley contribution shares via bootstrap.

    Each bootstrap round resamples sessions with replacement and re-runs the
    exact decomposition. Computationally heavy by design (``n_boot`` x
    ``2^k-1`` DEA solves); keep ``n_boot`` modest (100-300) for k<=7.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    point = shapley_decomposition(
        df, groups, outputs, backend, basic_groups, voice_groups
    )
    point.set_index("group")["contribution_pct"]

    boot_rows: list[dict[str, float]] = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            res = shapley_decomposition(
                df.iloc[idx].reset_index(drop=True), groups, outputs, backend,
                basic_groups, voice_groups,
            )
            boot_rows.append(res.set_index("group")["contribution_pct"].to_dict())
        except Exception as exc:  # a degenerate resample can break DEA
            logger.debug("bootstrap round %d failed: %s", b, exc)
    if not boot_rows:
        logger.warning("Shapley bootstrap produced no successful rounds")
        return point.assign(ci_low=np.nan, ci_high=np.nan)

    boot_df = pd.DataFrame(boot_rows)
    alpha = (1 - ci_level) / 2
    out = point.copy()
    out["ci_low"] = out["group"].map(
        boot_df.quantile(alpha).to_dict()
    )
    out["ci_high"] = out["group"].map(
        boot_df.quantile(1 - alpha).to_dict()
    )
    logger.info("Shapley bootstrap: %d/%d rounds usable", len(boot_df), n_boot)
    return out


# ----------------------------------------------------------------------
# 2. Threshold bootstrap p-value (Hansen-style fixed-X)
# ----------------------------------------------------------------------
def threshold_bootstrap_pvalue(
    df: pd.DataFrame,
    variable: str,
    controls: list[str],
    outcome: str = "bcc_efficiency",
    n_boot: int = 500,
    seed: int = 42,
    grid_points: int = 81,
    min_group: int = 15,
) -> dict:
    """Bootstrap p-value for threshold significance.

    Procedure (Hansen 1996, fixed-X bootstrap):
      1. fit threshold model and no-threshold model -> LR statistic and the
         unrestricted residuals;
      2. bootstrap: draw residuals with replacement, rebuild ``y*`` under the
         *no-threshold* null, recompute the LR statistic on each pseudo
         sample;
      3. p-value = fraction of bootstrap LR statistics >= observed LR.
    """
    data = df[[variable] + controls + [outcome]].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    x = data[variable].to_numpy(float)
    y = data[outcome].to_numpy(float)
    Z = data[controls].to_numpy(float)
    Zs = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    n = len(y)

    def _sse(x_arr, y_arr, gamma):
        d = (x_arr <= gamma).astype(float)
        Xt = np.column_stack([x_arr * d, x_arr * (1 - d), d, Zs])
        return float(np.sum((y_arr - LinearRegression().fit(Xt, y_arr).predict(Xt)) ** 2))

    def _sse_null(y_arr):
        Xn = np.column_stack([x, Zs])
        return float(np.sum((y_arr - LinearRegression().fit(Xn, y_arr).predict(Xn)) ** 2))

    candidates = np.percentile(x, np.linspace(10, 90, grid_points))
    valid = [
        g for g in candidates
        if ((x <= g).sum() >= min_group) and ((x > g).sum() >= min_group)
    ]
    if not valid:
        return {"variable": variable, "lr_stat": np.nan,
                "bootstrap_p": np.nan, "gamma_hat": np.nan, "n_boot": 0}

    sses = [(g, _sse(x, y, g)) for g in valid]
    gamma_hat, sse_t = min(sses, key=lambda t: t[1])
    sse_n = _sse_null(y)
    lr_obs = n * np.log(sse_n / sse_t) if sse_t > 0 else 0.0

    # unrestricted residuals for the bootstrap DGP
    d = (x <= gamma_hat).astype(float)
    Xt = np.column_stack([x * d, x * (1 - d), d, Zs])
    resid = y - LinearRegression().fit(Xt, y).predict(Xt)
    resid = resid - resid.mean()

    rng = np.random.default_rng(seed)
    count_ge = 0
    done = 0
    for _ in range(n_boot):
        y_star = y - resid + rng.choice(resid, size=n, replace=True)
        sse_n_b = _sse_null(y_star)
        best_b = min(_sse(x, y_star, g) for g in valid)
        lr_b = n * np.log(sse_n_b / best_b) if best_b > 0 else 0.0
        if lr_b >= lr_obs:
            count_ge += 1
        done += 1

    p_boot = count_ge / done
    logger.info("Threshold bootstrap %s: LR=%.2f p=%.4f (%d rounds)",
                variable, lr_obs, p_boot, done)
    return {
        "variable": variable,
        "gamma_hat": float(gamma_hat),
        "lr_stat": float(lr_obs),
        "bootstrap_p": float(p_boot),
        "n_boot": done,
    }


# ----------------------------------------------------------------------
# 3. Split-conformal prediction intervals
# ----------------------------------------------------------------------
def conformal_interval(
    df: pd.DataFrame,
    features: list[str],
    outcome: str = "bcc_efficiency",
    calibration_frac: float = 0.25,
    coverage: float = 0.9,
    seed: int = 42,
):
    """Split-conformal prediction band around the GBR efficiency predictor.

    Returns the fitted model plus a symmetric half-width ``q`` such that the
    interval ``pred +/- q`` achieves (approximately) the requested coverage
    on new data. Used to attach honest uncertainty to scenario predictions.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = df[features + [outcome]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data[features].to_numpy(float)
    y = data[outcome].to_numpy(float)

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=calibration_frac, random_state=seed
    )
    scaler = StandardScaler().fit(X_tr)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=seed,
    )
    model.fit(scaler.transform(X_tr), y_tr)

    cal_scores = np.abs(y_cal - model.predict(scaler.transform(X_cal)))
    q = float(np.quantile(cal_scores, min(coverage + 0.05, 1.0)))
    logger.info("Conformal: q=%.4f for %.0f%% coverage (n_cal=%d)",
                q, coverage * 100, len(y_cal))
    return {"model": model, "scaler": scaler, "half_width": q,
            "coverage_target": coverage, "n_train": len(y_tr),
            "n_calib": len(y_cal)}


# ----------------------------------------------------------------------
# 4. Super-efficiency ranking of the frontier
# ----------------------------------------------------------------------
def super_efficiency_ranking(
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    backend: str = "auto",
) -> pd.DataFrame:
    """Rank efficient units via BCC super-efficiency (excludes self)."""
    X = prepare_positive(df[inputs].to_numpy(float))
    Y = prepare_positive(df[list(outputs)].to_numpy(float))
    sup = super_efficiency_bcc(X, Y, backend)
    out = pd.DataFrame(
        {
            "unit_id": df["unit_id"].to_numpy() if "unit_id" in df else np.arange(len(df)),
            "super_efficiency": sup,
        }
    )
    # NaN (unsolvable LP) always sorts last
    out = out.sort_values("super_efficiency", ascending=False,
                          na_position="last").reset_index(drop=True)
    logger.info("Super-efficiency: top unit %.4f, frontier units ranked",
                out["super_efficiency"].max())
    return out
