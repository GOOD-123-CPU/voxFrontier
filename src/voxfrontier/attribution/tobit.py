"""Tobit-style (censored) regression via OLS on standardized features.

The original study used an OLS-style Tobit on efficiency scores bounded in
[0, 1]. We implement the same two-step approach (standardized regressors +
classical t/p inference) and additionally provide a proper two-limit Tobit
MLE via ``scipy.optimize.minimize`` so the censored likelihood is available.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("voxfrontier.tobit")


def _sig(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def tobit_ols(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "bcc_efficiency",
) -> dict:
    """OLS on standardized features with classical inference.

    Suitable when the dependent variable is a proportion (efficiency) with
    light censoring; mirroring the original study's "Tobit-style" stage.
    """
    data = df[features + [target_col]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data[features].to_numpy(float)
    y = data[target_col].to_numpy(float)

    Xs = StandardScaler().fit_transform(X)
    ols = LinearRegression().fit(Xs, y)
    y_hat = ols.predict(Xs)

    n, k = Xs.shape
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mse = ss_res / (n - k - 1)

    Xc = np.column_stack([np.ones(n), Xs])
    var_beta = mse * np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(np.diag(var_beta)[1:])
    t_stats = ols.coef_ / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))

    table = pd.DataFrame(
        {
            "feature": features,
            "coef": ols.coef_,
            "std_err": se,
            "t_stat": t_stats,
            "p_value": p_vals,
            "significance": [_sig(p) for p in p_vals],
        }
    )
    logger.info("Tobit(OLS) R2=%.4f adjR2=%.4f n=%d", r2, adj_r2, n)
    return {"table": table, "r2": r2, "adj_r2": adj_r2, "n": n}


def tobit_mle(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "bcc_efficiency",
    lower: float = 0.0,
    upper: float = 1.0,
) -> dict:
    """Two-limit Tobit MLE (classic censored regression).

    Maximises the censored log-likelihood with ``scipy.optimize.minimize``.
    Returns coefficients on the *standardized* feature scale plus the
    implied scale ``sigma``.
    """
    data = df[features + [target_col]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data[features].to_numpy(float)
    y = data[target_col].to_numpy(float)
    Xs = StandardScaler().fit_transform(X)
    n, k = Xs.shape
    Xc = np.column_stack([np.ones(n), Xs])

    def neg_ll(params: np.ndarray) -> float:
        beta = params[: k + 1]
        log_sigma = params[k + 1]
        sigma = np.exp(log_sigma) + 1e-8
        xb = Xc @ beta
        unc = (y > lower) & (y < upper)
        ll = np.zeros(n)
        # uncensored: normal density
        z = (y[unc] - xb[unc]) / sigma
        ll[unc] = (
            -0.5 * z ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        ).sum()
        # left-censored: log Phi((lower - xb)/sigma)
        lo = y <= lower
        if lo.any():
            ll[lo] = stats.norm.logcdf((lower - xb[lo]) / sigma).sum()
        # right-censored
        hi = y >= upper
        if hi.any():
            ll[hi] = stats.norm.logcdf((xb[hi] - upper) / sigma).sum()
        return -ll.sum()

    x0 = np.zeros(k + 2)
    x0[-1] = np.log(max(np.std(y), 1e-3))
    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
    beta = res.x[: k + 1]
    sigma = float(np.exp(res.x[k + 1]))

    table = pd.DataFrame(
        {
            "feature": ["const"] + features,
            "coef_mle": beta,
        }
    )
    logger.info("Tobit MLE converged=%s sigma=%.4f", res.success, sigma)
    return {"table": table, "sigma": sigma, "converged": bool(res.success)}


def mediation_analysis(
    df: pd.DataFrame,
    x_col: str,
    mediator_col: str,
    y_col: str = "bcc_efficiency",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Baron-Kenny three-step mediation with Sobel test and bootstrap CI.

    Path: x -> mediator -> y. All variables standardized internally.
    """
    data = df[[x_col, mediator_col, y_col]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data[x_col].to_numpy(float)
    M = data[mediator_col].to_numpy(float)
    Y = data[y_col].to_numpy(float)
    Xs = (X - X.mean()) / (X.std() or 1.0)
    Ms = (M - M.mean()) / (M.std() or 1.0)

    # total effect c
    c = float(LinearRegression().fit(Xs[:, None], Y).coef_[0])
    # path a
    reg_a = LinearRegression().fit(Xs[:, None], Ms)
    a = float(reg_a.coef_[0])
    # paths b, c'
    XM = np.column_stack([Xs, Ms])
    reg_full = LinearRegression().fit(XM, Y)
    b = float(reg_full.coef_[1])
    c_prime = float(reg_full.coef_[0])

    indirect = a * b
    n = len(Y)

    # Sobel
    res_a = Ms - reg_a.predict(Xs[:, None])
    se_a = np.sqrt(np.sum(res_a ** 2) / (n - 2) / np.sum(Xs ** 2))
    res_full = Y - reg_full.predict(XM)
    XM_c = np.column_stack([np.ones(n), XM])
    var_b = np.sum(res_full ** 2) / (n - 3) * np.linalg.inv(XM_c.T @ XM_c)
    se_b = float(np.sqrt(var_b[2, 2]))
    sobel_se = float(np.sqrt(a ** 2 * se_b ** 2 + b ** 2 * se_a ** 2))
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0.0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    # bootstrap
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_i = float(LinearRegression().fit(Xs[idx, None], Ms[idx]).coef_[0])
        b_i = float(
            LinearRegression().fit(np.column_stack([Xs[idx], Ms[idx]]), Y[idx]).coef_[1]
        )
        boots[i] = a_i * b_i
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    out = {
        "c_total": c,
        "a_path": a,
        "b_path": b,
        "c_prime_direct": c_prime,
        "indirect_ab": indirect,
        "proportion_mediated_pct": abs(indirect / c) * 100 if abs(c) > 1e-6 else np.nan,
        "sobel_z": sobel_z,
        "sobel_p": sobel_p,
        "boot_ci_low": float(ci_lo),
        "boot_ci_high": float(ci_hi),
        "n": n,
    }
    logger.info(
        "Mediation: indirect=%.4f sobel_z=%.3f p=%.4f CI=[%.4f, %.4f]",
        indirect, sobel_z, sobel_p, ci_lo, ci_hi,
    )
    return out
