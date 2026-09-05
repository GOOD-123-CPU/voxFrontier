"""Quadratic (U-shape) tests, threshold regression, quantile bands, interactions.

All estimators operate on BCC efficiency as the dependent variable and
voice variables as the regressors of interest, with optional controls.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("voxfrontier.nonlinear")


def _sig(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def _clean(df: pd.DataFrame, cols: list[str], target: str):
    data = df[cols + [target]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


# ----------------------------------------------------------------------
# 1. Quadratic regression: U-shape / inverse-U detection
# ----------------------------------------------------------------------
def quadratic_tests(
    df: pd.DataFrame,
    voice_vars: list[str],
    target_col: str = "bcc_efficiency",
) -> pd.DataFrame:
    """Fit y = b0 + b1*x_s + b2*x_s^2 per voice variable.

    ``x_s`` is the standardized variable; the extremum is back-transformed
    to the raw scale. An F-test compares against the linear model and a
    t-test evaluates ``b2`` directly.
    """
    rows = []
    for col in voice_vars:
        data = _clean(df, [col], target_col)
        x = data[col].to_numpy(float)
        y = data[target_col].to_numpy(float)
        x_mean, x_std = x.mean(), x.std() or 1.0
        xs = (x - x_mean) / x_std

        lin = LinearRegression().fit(xs[:, None], y)
        ss_lin = float(np.sum((y - lin.predict(xs[:, None])) ** 2))

        Xq = np.column_stack([xs, xs ** 2])
        quad = LinearRegression().fit(Xq, y)
        ss_quad = float(np.sum((y - quad.predict(Xq)) ** 2))

        b1, b2 = float(quad.coef_[0]), float(quad.coef_[1])

        n = len(y)
        df1, df2 = 1, n - 3
        f_stat = ((ss_lin - ss_quad) / df1) / (ss_quad / df2)
        f_p = 1 - stats.f.cdf(f_stat, df1, df2)

        mse = ss_quad / df2
        Xc = np.column_stack([np.ones(n), xs, xs ** 2])
        var_b = mse * np.linalg.inv(Xc.T @ Xc)
        se_b2 = float(np.sqrt(var_b[2, 2]))
        t_b2 = b2 / se_b2
        p_b2 = 2 * (1 - stats.t.cdf(abs(t_b2), df2))

        extremum_raw = np.nan
        shape = "linear"
        if p_b2 < 0.1:
            s_star = -b1 / (2 * b2)
            extremum_raw = s_star * x_std + x_mean
            shape = "inverse_u" if b2 < 0 else "u_shape"

        rows.append(
            {
                "variable": col,
                "b1_linear": b1,
                "b2_quadratic": b2,
                "t_b2": t_b2,
                "p_b2": p_b2,
                "f_stat": f_stat,
                "f_p": f_p,
                "r2_linear": lin.score(xs[:, None], y),
                "r2_quadratic": quad.score(Xq, y),
                "shape": shape,
                "extremum_raw": extremum_raw,
                "significance": _sig(p_b2),
                "x_mean": x_mean,
                "x_std": x_std,
                "intercept": float(quad.intercept_),
            }
        )

    out = pd.DataFrame(rows)
    logger.info("Quadratic tests: %s", ", ".join(
        f"{r.variable}={r.shape}" for r in out.itertuples()
    ))
    return out


# ----------------------------------------------------------------------
# 2. Threshold (regime-switching) regression with grid search + LR test
# ----------------------------------------------------------------------
def threshold_regression(
    df: pd.DataFrame,
    voice_vars: list[str],
    controls: list[str],
    target_col: str = "bcc_efficiency",
    grid_lo: float = 0.10,
    grid_hi: float = 0.90,
    min_group: int = 15,
) -> pd.DataFrame:
    """Grid-search threshold ``gamma`` per voice variable.

    Model: y = a + b1*x*I(x<=g) + b2*x*I(x>g) + d*Z + e.
    A likelihood-ratio style statistic compares against the no-threshold
    linear model.
    """
    rows = []
    for col in voice_vars:
        data = _clean(df, [col] + controls, target_col)
        x = data[col].to_numpy(float)
        y = data[target_col].to_numpy(float)
        Z = data[controls].to_numpy(float)
        Zs = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
        n = len(y)
        if n < 2 * min_group:
            logger.warning("threshold: %s skipped (n=%d too small)", col, n)
            continue

        best = (np.inf, None)
        for gamma in np.percentile(x, np.arange(10, 91, 1.0)):
            d = (x <= gamma).astype(float)
            if d.sum() < min_group or (n - d.sum()) < min_group:
                continue
            Xt = np.column_stack([x * d, x * (1 - d), d, Zs])
            sse = float(np.sum((y - LinearRegression().fit(Xt, y).predict(Xt)) ** 2))
            if sse < best[0]:
                best = (sse, gamma)

        if best[1] is None:
            logger.warning("threshold: %s no valid gamma found", col)
            continue

        sse_best, gamma = best
        d = (x <= gamma).astype(float)
        Xb = np.column_stack([x * d, x * (1 - d), d, Zs])
        reg = LinearRegression().fit(Xb, y)
        b_below, b_above = float(reg.coef_[0]), float(reg.coef_[1])

        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_thresh = 1 - sse_best / ss_tot

        Xn = np.column_stack([x, Zs])
        reg_n = LinearRegression().fit(Xn, y)
        ss_no = float(np.sum((y - reg_n.predict(Xn)) ** 2))
        r2_no = 1 - ss_no / ss_tot

        k_thresh, k_no = Xb.shape[1] + 1, Xn.shape[1] + 1
        lr = n * np.log(ss_no / sse_best) if sse_best > 0 else 0.0
        lr_p = 1 - stats.chi2.cdf(lr, df=k_thresh - k_no)

        rows.append(
            {
                "variable": col,
                "threshold_gamma": gamma,
                "beta_below": b_below,
                "beta_above": b_above,
                "slope_diff": abs(b_below - b_above),
                "r2_threshold": r2_thresh,
                "r2_no_threshold": r2_no,
                "lr_stat": lr,
                "lr_p": lr_p,
                "n_below": int(d.sum()),
                "n_above": int(n - d.sum()),
                "significance": _sig(lr_p),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        logger.info(
            "Thresholds: %s",
            ", ".join(f"{r.variable}@{r.threshold_gamma:.2f}" for r in out.itertuples()),
        )
    return out


# ----------------------------------------------------------------------
# 3. Quantile-band effects (slope by efficiency band)
# ----------------------------------------------------------------------
def quantile_band_effects(
    df: pd.DataFrame,
    voice_vars: list[str],
    target_col: str = "bcc_efficiency",
    quantiles=(0.25, 0.50, 0.75, 0.90),
    band: float = 0.20,
    min_n: int = 20,
) -> pd.DataFrame:
    """Slope of each voice variable within bands around efficiency quantiles."""
    rows = []
    for col in voice_vars:
        data = _clean(df, [col], target_col)
        if len(data) < min_n:
            logger.warning("quantile bands: %s skipped (n=%d)", col, len(data))
            continue
        x = data[col].to_numpy(float)
        y = data[target_col].to_numpy(float)
        sd = x.std()
        xs = (x - x.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0

        row: dict = {"variable": col}
        for q in quantiles:
            y_lo, y_hi = np.quantile(y, max(q - band, 0)), np.quantile(
                y, min(q + band, 1)
            )
            mask = (y >= y_lo) & (y <= y_hi)
            label = f"q{int(q * 100)}"
            if mask.sum() < min_n:
                row[f"beta_{label}"] = np.nan
                row[f"p_{label}"] = np.nan
                continue
            xq, yq = xs[mask], y[mask]
            reg = LinearRegression().fit(xq[:, None], yq)
            beta = float(reg.coef_[0])
            resid = yq - reg.predict(xq[:, None])
            mse = float(np.sum(resid ** 2) / (len(yq) - 2))
            se = np.sqrt(mse / np.sum(xq ** 2)) if np.sum(xq ** 2) > 0 else 1e-6
            t = beta / se if se > 0 else 0.0
            p = 2 * (1 - stats.t.cdf(abs(t), len(yq) - 2))
            row[f"beta_{label}"] = beta
            row[f"p_{label}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. Subgroup interactions (product type / gender)
# ----------------------------------------------------------------------
def subgroup_effects(
    df: pd.DataFrame,
    voice_vars: list[str],
    group_col: str,
    target_col: str = "bcc_efficiency",
    min_n: int = 15,
) -> pd.DataFrame:
    """Within-group linear slope of each voice variable."""
    rows = []
    for col in voice_vars:
        # numeric columns only; the grouping column is kept as-is
        data = df[[col, group_col, target_col]].copy()
        for c in (col, target_col):
            data[c] = pd.to_numeric(data[c], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        x = data[col].to_numpy(float)
        y = data[target_col].to_numpy(float)
        g = data[group_col].to_numpy()
        sd = x.std()
        xs = (x - x.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0

        for level in pd.unique(g):
            mask = g == level
            if mask.sum() < min_n:
                continue
            reg = LinearRegression().fit(xs[mask][:, None], y[mask])
            beta = float(reg.coef_[0])
            resid = y[mask] - reg.predict(xs[mask][:, None])
            mse = float(np.sum(resid ** 2) / (mask.sum() - 2))
            se = (
                np.sqrt(mse / np.sum(xs[mask] ** 2))
                if np.sum(xs[mask] ** 2) > 0
                else 1e-6
            )
            t = beta / se if se > 0 else 0.0
            p = 2 * (1 - stats.t.cdf(abs(t), max(mask.sum() - 2, 1)))
            rows.append(
                {
                    "variable": col,
                    "group_col": group_col,
                    "group": str(level),
                    "n": int(mask.sum()),
                    "beta": beta,
                    "p_value": p,
                    "significance": _sig(p),
                }
            )
    return pd.DataFrame(rows)
