"""Random-forest importance and optional SHAP analysis of BCC efficiency."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("voxfrontier.forest")


def forest_importance(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "bcc_efficiency",
    n_trees: int = 300,
    seed: int = 42,
    basic_vars: tuple = (),
) -> dict:
    """Random-forest feature importance (plus 5-fold CV R^2)."""
    data = df[features + [target_col]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data[features].to_numpy(float)
    y = data[target_col].to_numpy(float)

    X_scaled = StandardScaler().fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=n_trees, max_depth=8, random_state=seed, n_jobs=-1
    )
    rf.fit(X_scaled, y)
    cv = cross_val_score(rf, X_scaled, y, cv=5, scoring="r2")

    imp = rf.feature_importances_
    out = pd.DataFrame(
        {
            "feature": features,
            "importance": imp,
            "importance_pct": imp / imp.sum() * 100,
        }
    )
    out["kind"] = np.where(
        out["feature"].isin(basic_vars), "basic", "voice"
    )
    out = out.sort_values("importance_pct", ascending=False).reset_index(drop=True)

    voice_total = float(out.loc[out["kind"] == "voice", "importance_pct"].sum())
    logger.info("RF importance: voice share=%.1f%%, CV R2=%.4f",
                voice_total, cv.mean())

    return {
        "table": out,
        "voice_share_pct": voice_total,
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "model": rf,
        "X_scaled": X_scaled,
        "y": y,
    }


def shap_analysis(forest: dict, seed: int = 42):
    """Optional SHAP values with a gradient-boosting surrogate.

    Returns ``None`` when the optional ``shap`` package is unavailable — the
    pipeline degrades gracefully instead of failing.
    """
    try:
        import shap
    except ImportError:
        logger.info("shap not installed; skipping SHAP analysis "
                    "(pip install shap to enable).")
        return None

    X_scaled = forest["X_scaled"]
    y = forest["y"]
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=seed
    )
    gb.fit(X_scaled, y)

    explainer = shap.TreeExplainer(gb)
    values = explainer.shap_values(X_scaled)
    mean_abs = np.abs(values).mean(axis=0)
    pct = mean_abs / mean_abs.sum() * 100

    table = pd.DataFrame(
        {
            "feature": forest["table"]["feature"],
            "mean_abs_shap": mean_abs,
            "contribution_pct": pct,
        }
    ).sort_values("contribution_pct", ascending=False).reset_index(drop=True)

    return {"table": table, "shap_values": values, "model": gb}
