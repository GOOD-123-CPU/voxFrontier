"""Counterfactual scenario simulation and per-unit diagnostics.

Six scenarios (mirroring the corrected study design):

* ``baseline``      — keep everything as-is;
* ``voice_to_efficient`` — set the four voice variables of *every* session
  to the high-efficiency-group means;
* ``voice_to_valley``    — set them to the U-valley (worst case);
* ``voice_plus10``  — move each voice variable 10% further away from its
  valley (direction depends on which side of the valley it sits);
* ``full_optimal``  — per-product-type high-efficiency-group means
  (category-tailored);
* ``conservative``  — apply the efficient-group targets only to low-efficiency
  sessions (< 0.8).

Predictions come from a gradient-boosting regressor trained on the observed
data; the valley positions come from the quadratic tests (stage 3) rather
than hard-coded constants.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("voxfrontier.simulation")

SCENARIOS = (
    "baseline",
    "voice_to_efficient",
    "voice_to_valley",
    "voice_plus10",
    "full_optimal",
    "conservative",
)


def _clean(df: pd.DataFrame, features: list[str], target_col: str):
    data = df[features + [target_col, "product_type"]].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=features + [target_col]
    )
    return data


def fit_predictor(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "bcc_efficiency",
    seed: int = 42,
) -> dict:
    """Train the efficiency predictor used by all scenarios."""
    data = _clean(df, features, target_col)
    X = data[features].to_numpy(float)
    y = data[target_col].to_numpy(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=seed,
    )
    model.fit(Xs, y)
    cv = cross_val_score(model, Xs, y, cv=5, scoring="r2")
    logger.info("Predictor GBR CV R2=%.4f +/- %.4f", cv.mean(), cv.std())
    return {
        "model": model,
        "scaler": scaler,
        "data": data,
        "X": X,
        "y_pred": model.predict(Xs),
        "product_type": data["product_type"].to_numpy(),
        "cv_r2": float(cv.mean()),
    }


def _valleys_from_quadratic(quad_df: pd.DataFrame | None,
                            voice_vars: list[str],
                            fallback: dict[str, float]) -> dict[str, float]:
    """Take valley positions from stage-3 quadratic fits when available."""
    valleys = {}
    for col in voice_vars:
        v = fallback.get(col, np.nan)
        if quad_df is not None:
            hit = quad_df.loc[quad_df["variable"] == col]
            if len(hit) and hit.iloc[0]["shape"] == "u_shape":
                v = float(hit.iloc[0]["extremum_raw"])
        valleys[col] = v
    return valleys


def run_scenarios(
    df: pd.DataFrame,
    voice_vars: list[str],
    feature_cols: list[str],
    efficient_means: dict[str, float],
    valleys: dict[str, float],
    predictor: dict,
    target_col: str = "bcc_efficiency",
) -> pd.DataFrame:
    """Evaluate the six intervention scenarios with the fitted predictor."""
    X = predictor["X"].copy()
    model, scaler = predictor["model"], predictor["scaler"]
    y_base = predictor["y_pred"]
    ptype = predictor["product_type"]
    base_mean = float(y_base.mean())

    def predict(X_cf: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_cf))

    def summarise(name: str, y_cf: np.ndarray) -> dict:
        return {
            "scenario": name,
            "mean_efficiency": float(y_cf.mean()),
            "efficient_units": int((y_cf >= 0.999).sum()),
            "inefficient_units": int((y_cf < 0.8).sum()),
            "gain_vs_baseline": float(y_cf.mean() - base_mean),
        }

    results = [summarise("baseline", y_base)]

    # B: every unit to efficient-group means
    Xb = X.copy()
    for col in voice_vars:
        Xb[:, feature_cols.index(col)] = efficient_means[col]
    results.append(summarise("voice_to_efficient", predict(Xb)))

    # C: every unit to the valley (worst case)
    Xc = X.copy()
    for col in voice_vars:
        Xc[:, feature_cols.index(col)] = valleys[col]
    results.append(summarise("voice_to_valley", predict(Xc)))

    # D: push each unit 10% further from the valley
    Xd = X.copy()
    for col in voice_vars:
        j = feature_cols.index(col)
        valley = valleys[col]
        cur = Xd[:, j]
        Xd[:, j] = np.where(
            cur <= valley, valley * 0.80, valley * 1.20
        )
    results.append(summarise("voice_plus10", predict(Xd)))

    # E: per-product-type efficient-group means
    Xe = X.copy()
    for pt in pd.unique(ptype):
        mask = ptype == pt
        sub = df[(df["product_type"] == pt) & (df[target_col] >= 0.95)]
        if len(sub) < 3:
            sub = df[(df["product_type"] == pt) & (df[target_col] >= 0.9)]
        if len(sub) == 0:
            continue
        for col in voice_vars:
            Xe[mask, feature_cols.index(col)] = float(sub[col].mean())
    results.append(summarise("full_optimal", predict(Xe)))

    # F: only low-efficiency units get the efficient-group targets
    Xf = X.copy()
    low_mask = y_base < 0.8
    for col in voice_vars:
        Xf[low_mask, feature_cols.index(col)] = efficient_means[col]
    results.append(summarise("conservative", predict(Xf)))

    out = pd.DataFrame(results)
    best = out.loc[out["gain_vs_baseline"].idxmax(), "scenario"]
    logger.info("Best scenario: %s (+%.4f)",
                best, out["gain_vs_baseline"].max())
    return out


def diagnose_inefficient(
    df: pd.DataFrame,
    voice_vars: list[str],
    efficient_means: dict[str, float],
    valleys: dict[str, float],
    target_col: str = "bcc_efficiency",
    eff_threshold: float = 0.8,
) -> pd.DataFrame:
    """Per-unit advice for low-efficiency sessions.

    Suggestion logic mirrors the corrected study: (1) if a voice variable
    sits within +/-10% of the U-valley, move it toward the efficient-group
    mean (direction determined by where that mean is); (2) otherwise, if it
    deviates >20% from the efficient reference, suggest up/down.
    """
    rows = []
    low = df[df[target_col] < eff_threshold]
    for _, r in low.iterrows():
        unit = {
            "unit_id": r.get("unit_id", np.nan),
            "product_type": r["product_type"],
            "efficiency": round(float(r[target_col]), 4),
        }
        issues: list[str] = []
        for col in voice_vars:
            val, valley = float(r[col]), valleys[col]
            ref = efficient_means[col]
            advice = "ok"
            if np.isfinite(valley) and abs(val - valley) / max(valley, 1e-9) < 0.10:
                if ref < valley:
                    advice = "decrease (escape valley)"
                else:
                    advice = "increase (escape valley)"
                issues.append(f"{col}: {advice} ({val:.1f} -> {ref:.1f})")
            elif abs(val - ref) / max(abs(ref), 1e-9) > 0.20:
                advice = "increase" if val < ref else "decrease"
                issues.append(f"{col}: {advice} ({val:.1f} -> {ref:.1f})")
            unit[f"{col}_current"] = round(val, 2)
            unit[f"{col}_reference"] = round(ref, 2)
            unit[f"{col}_advice"] = advice
        unit["n_issues"] = len(issues)
        unit["recommendation"] = "; ".join(issues) if issues else "voice profile OK"
        rows.append(unit)
    out = pd.DataFrame(rows)
    logger.info("Diagnosed %d low-efficiency units", len(out))
    return out
