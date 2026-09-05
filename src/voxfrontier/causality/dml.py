"""Double Machine Learning (DML) for voice effects on efficiency.

Why DML here?
-------------
The stages upstream of this module (DEA -> Tobit/RF) are *predictive*:
they quantify association between voice features and efficiency, but
confounders (stream duration, price, traffic, category mix) may bias the
partial effect of any single voice variable. DML (Chernozhukov et al. 2018)
removes this bias by combining:

1. **Neyman orthogonality** — the target parameter is the coefficient of a
   Frisch–Waugh–Lovell (FWL) residual-on-residual regression, insensitive to
   small errors in the nuisance estimates;
2. **Cross-fitting** — nuisance functions are estimated out-of-fold, so the
   ML learners' own overfitting cannot leak into the effect estimate.

For a continuous treatment ``D`` (a voice variable) with controls ``X``:

    D = m(X) + v,      Y = theta * D + g(X) + eps
    theta_hat = sum(v * y_res) / sum(v * v)

with ``v = D - m_hat(X)`` and ``y_res = Y - g_hat(X)`` computed on held-out
folds. We report the point estimate, its std. error, 95% CI and p-value, plus
a heterogeneity split (effect by efficiency quantile band) since the U-shape
analysis implies effects are *not* constant across the efficiency
distribution.

Partially linear OLS ( FWL with linear nuisances ) is included as the
classical benchmark so users can see the ML-flexible correction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger("voxfrontier.dml")


@dataclass
class DMLEffect:
    """Container for one treatment variable's causal estimate."""

    treatment: str
    estimate: float
    std_error: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int
    n_folds: int
    learner: str = "rf"
    # effect by efficiency band (heterogeneity), filled optionally
    heterogeneity: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "treatment": self.treatment,
            "dml_estimate": self.estimate,
            "std_error": self.std_error,
            "z_stat": self.z_stat,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_folds": self.n_folds,
            "learner": self.learner,
        }
        for band, eff in self.heterogeneity.items():
            out[f"effect_{band}"] = eff
        return out


def _make_learner(name: str, seed: int):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            random_state=seed, n_jobs=-1,
        )
    if name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=seed,
        )
    if name == "ridge":
        return Ridge(alpha=1.0)
    return LinearRegression()


def dml_partially_linear(
    df: pd.DataFrame,
    treatments: list[str],
    controls: list[str],
    outcome: str = "bcc_efficiency",
    n_folds: int = 5,
    learner: str = "rf",
    seed: int = 42,
    heterogeneity_bands: tuple | None = (0.25, 0.75),
) -> pd.DataFrame:
    """Cross-fitted DML for the partially linear model.

    Parameters
    ----------
    df:
        Session-level table (already containing DEA efficiency).
    treatments:
        Voice variables whose partial effects are estimated one at a time.
    controls:
        Confounders to net out (basic operational + remaining voice vars).
    outcome:
        Dependent variable (default BCC efficiency).
    n_folds:
        Number of cross-fitting folds.
    learner:
        Nuisance learner: ``rf`` | ``gbm`` | ``ridge`` | ``ols``.
    seed:
        RNG seed for fold assignment and learners.
    heterogeneity_bands:
        Efficiency quantiles defining bands for subgroup effects
        (low: < q25, mid: between, high: > q75). ``None`` disables.

    Returns
    -------
    pd.DataFrame
        One row per treatment with estimate/SE/CI/p + per-band effects.
    """
    data = df[treatments + controls + [outcome]].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    Y = data[outcome].to_numpy(float)
    n = len(Y)
    if n < 50:
        raise ValueError(f"DML needs >=50 observations, got {n}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results: list[DMLEffect] = []

    for treat in treatments:
        D = data[treat].to_numpy(float)
        X = data[controls].to_numpy(float)
        # standardize controls once (learners are trees; OLS/ridge benefit)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        Xs = (X - mu) / sd

        v_res = np.zeros(n)
        y_res = np.zeros(n)
        for train_idx, test_idx in kf.split(Xs):
            m_model = _make_learner(learner, seed)
            g_model = _make_learner(learner, seed)
            m_model.fit(Xs[train_idx], D[train_idx])
            g_model.fit(Xs[train_idx], Y[train_idx])
            v_res[test_idx] = D[test_idx] - m_model.predict(Xs[test_idx])
            y_res[test_idx] = Y[test_idx] - g_model.predict(Xs[test_idx])

        denom = float(np.sum(v_res ** 2))
        if denom <= 0:
            logger.warning("DML: zero residual variance for %s; skipped", treat)
            continue
        theta = float(np.sum(v_res * y_res) / denom)

        # influence-function standard error
        psi = v_res * (y_res - theta * v_res)
        se = float(np.sqrt(np.mean(psi ** 2) / n))
        z = theta / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        ci = (theta - 1.959964 * se, theta + 1.959964 * se)

        eff = DMLEffect(
            treatment=treat, estimate=theta, std_error=se, z_stat=z,
            p_value=p, ci_low=ci[0], ci_high=ci[1], n=n,
            n_folds=n_folds, learner=learner,
        )

        # heterogeneity: split by outcome quantiles, refit FWL within band
        if heterogeneity_bands:
            q_lo, q_hi = heterogeneity_bands
            y_q25, y_q75 = np.quantile(Y, q_lo), np.quantile(Y, q_hi)
            for band_name, band_mask in (
                ("low", y_q25 >= Y),
                ("mid", (y_q25 < Y) & (y_q75 >= Y)),
                ("high", y_q75 < Y),
            ):
                if band_mask.sum() < 40:
                    eff.heterogeneity[band_name] = np.nan
                    continue
                Db, Xb, Yb = D[band_mask], Xs[band_mask], Y[band_mask]
                m_b = _make_learner(learner, seed).fit(Xb, Db)
                g_b = _make_learner(learner, seed).fit(Xb, Yb)
                v_b = Db - m_b.predict(Xb)
                yr_b = Yb - g_b.predict(Xb)
                den_b = float(np.sum(v_b ** 2))
                eff.heterogeneity[band_name] = (
                    float(np.sum(v_b * yr_b) / den_b) if den_b > 0 else np.nan
                )

        results.append(eff)
        logger.info(
            "DML %s: theta=%.4f se=%.4f p=%.4f CI=[%.4f, %.4f]",
            treat, theta, se, p, ci[0], ci[1],
        )

    return pd.DataFrame([r.to_dict() for r in results])


def fwl_ols_benchmark(
    df: pd.DataFrame,
    treatments: list[str],
    controls: list[str],
    outcome: str = "bcc_efficiency",
) -> pd.DataFrame:
    """Classical FWL/OLS partial-effect benchmark (linear nuisances)."""
    data = df[treatments + controls + [outcome]].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    Y = data[outcome].to_numpy(float)
    rows = []
    for treat in treatments:
        D = data[treat].to_numpy(float)
        Xc = data[controls].to_numpy(float)
        Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-12)
        m = LinearRegression().fit(Xc, D)
        g = LinearRegression().fit(Xc, Y)
        v = D - m.predict(Xc)
        yr = Y - g.predict(Xc)
        theta = float(np.sum(v * yr) / np.sum(v ** 2))
        resid = yr - theta * v
        n = len(Y)
        se = float(
            np.sqrt(np.sum(resid ** 2) / (n - len(controls) - 2)
                    / np.sum(v ** 2))
        )
        z = theta / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({"treatment": treat, "fwl_ols_estimate": theta,
                     "std_error": se, "z_stat": z, "p_value": p})
    return pd.DataFrame(rows)
