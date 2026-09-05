"""End-to-end pipeline: synth -> DEA -> attribution -> nonlinear -> causality -> simulation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from voxfrontier import __version__
from voxfrontier.attribution.forest import forest_importance, shap_analysis
from voxfrontier.attribution.shapley import shapley_decomposition
from voxfrontier.attribution.tobit import mediation_analysis, tobit_mle, tobit_ols
from voxfrontier.config import Config, set_seed
from voxfrontier.data.schema import (
    BASIC_MODEL_INPUTS,
    BASIC_SHAPLEY_GROUPS,
    BCC_EFF,
    CCR_EFF,
    FULL_MODEL_INPUTS,
    HNR,
    JITTER,
    MEAN_F0,
    OUTPUTS,
    PRICE,
    SCALE_EFF,
    SHAPLEY_GROUPS,
    SPEECH_RATE,
    STREAM_HOURS,
    UNIT_ID,
    VIEWERS,
    VOICE_MODEL_INPUTS,
    VOICE_SHAPLEY_GROUPS,
)
from voxfrontier.dea.models import run_three_models
from voxfrontier.nonlinear.analysis import (
    quadratic_tests,
    quantile_band_effects,
    subgroup_effects,
    threshold_regression,
)
from voxfrontier.simulation.scenarios import (
    diagnose_inefficient,
    fit_predictor,
    run_scenarios,
)
from voxfrontier.utils.plotting import setup_style

logger = logging.getLogger("voxfrontier.pipeline")


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("saved %s", path)


# ---------------------------------------------------------------------- #
# stages
# ---------------------------------------------------------------------- #
def stage_synth(cfg: Config) -> pd.DataFrame:
    """Generate (or load cached) synthetic data."""
    from voxfrontier.data.synth import generate_synthetic

    csv_path = cfg.data_dir / "synthetic_sessions.csv"
    if csv_path.exists():
        logger.info("loading cached synthetic data: %s", csv_path)
        return pd.read_csv(csv_path)

    set_seed(cfg.seed)
    df = generate_synthetic(cfg.n_units, cfg.seed)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("generated synthetic data: %d sessions -> %s", len(df), csv_path)
    return df


def stage_dea(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    specs = {
        "A": BASIC_MODEL_INPUTS,
        "B": VOICE_MODEL_INPUTS,
        "C": FULL_MODEL_INPUTS,
    }
    results = run_three_models(df, specs, OUTPUTS, cfg.dea_backend)

    model_b = results["B"]
    df = df.copy()
    df[BCC_EFF] = model_b["bcc"]
    df[CCR_EFF] = model_b["ccr"]
    df[SCALE_EFF] = model_b["scale"]

    summary = pd.DataFrame(
        [
            {
                "model": name,
                "mean_bcc": float(np.nanmean(r["bcc"])),
                "mean_ccr": float(np.nanmean(r["ccr"])),
                "efficient_units_bcc": int((r["bcc"] >= 0.999).sum()),
            }
            for name, r in results.items()
        ]
    )
    _save_table(summary, cfg.output_dir / "dea_model_summary.csv")

    per_unit = df[[UNIT_ID, BCC_EFF, CCR_EFF, SCALE_EFF]].copy()
    per_unit["efficient_bcc"] = per_unit[BCC_EFF] >= 0.999
    _save_table(per_unit, cfg.output_dir / "dea_efficiency_by_unit.csv")
    return df


def stage_attribution(df: pd.DataFrame, cfg: Config):
    valid = df[df[BCC_EFF].notna()].copy()
    basic_vars = tuple(BASIC_MODEL_INPUTS)

    # 1) exact Shapley over input groups (computed from data, never cached)
    shapley = shapley_decomposition(
        valid,
        groups=SHAPLEY_GROUPS,
        outputs=OUTPUTS,
        backend=cfg.dea_backend,
        basic_groups=BASIC_SHAPLEY_GROUPS,
        voice_groups=VOICE_SHAPLEY_GROUPS,
    )
    voice_share = float(shapley.loc[shapley["kind"] == "voice",
                                    "contribution_pct"].sum())
    logger.info("Shapley voice share: %.1f%%", voice_share)
    _save_table(shapley, cfg.output_dir / "shapley_decomposition.csv")

    # 2) random-forest importance
    features = BASIC_MODEL_INPUTS + [MEAN_F0, HNR, JITTER, SPEECH_RATE]
    forest = forest_importance(
        valid, features, BCC_EFF, n_trees=cfg.rf_trees,
        seed=cfg.seed, basic_vars=basic_vars,
    )
    _save_table(forest["table"], cfg.output_dir / "forest_importance.csv")

    # 3) optional SHAP
    shap_out = None
    if cfg.use_shap:
        shap_out = shap_analysis(forest, seed=cfg.seed)
        if shap_out is not None:
            _save_table(shap_out["table"], cfg.output_dir / "shap_values.csv")

    # 4) Tobit-style + proper MLE
    tobit_feats = BASIC_MODEL_INPUTS + [MEAN_F0, HNR, JITTER, SPEECH_RATE]
    tob = tobit_ols(valid, tobit_feats, BCC_EFF)
    _save_table(tob["table"], cfg.output_dir / "tobit_ols.csv")
    tob_mle = tobit_mle(valid, tobit_feats, BCC_EFF)
    _save_table(tob_mle["table"], cfg.output_dir / "tobit_mle.csv")

    # 5) mediation: voice index -> viewers -> efficiency
    med = mediation_analysis(valid, "voice_index", VIEWERS, BCC_EFF, seed=cfg.seed)
    _save_table(
        pd.DataFrame([med]), cfg.output_dir / "mediation_analysis.csv"
    )

    return {
        "shapley": shapley,
        "voice_share_pct": voice_share,
        "forest": forest,
        "shap": shap_out,
        "tobit_ols": tob,
        "tobit_mle": tob_mle,
        "mediation": med,
    }


def stage_nonlinear(df: pd.DataFrame, cfg: Config):
    valid = df[df[BCC_EFF].notna()].copy()
    voice_vars = [MEAN_F0, HNR, JITTER, SPEECH_RATE]
    controls = [STREAM_HOURS, PRICE, VIEWERS]

    quad = quadratic_tests(valid, voice_vars, BCC_EFF)
    _save_table(quad, cfg.output_dir / "quadratic_tests.csv")

    thresh = threshold_regression(valid, voice_vars, controls, BCC_EFF)
    _save_table(thresh, cfg.output_dir / "threshold_regression.csv")

    qband = quantile_band_effects(valid, voice_vars, BCC_EFF)
    _save_table(qband, cfg.output_dir / "quantile_bands.csv")

    inter_pt = subgroup_effects(valid, voice_vars, "product_type", BCC_EFF)
    inter_g = subgroup_effects(valid, voice_vars, "gender", BCC_EFF)
    interactions = pd.concat([inter_pt, inter_g], ignore_index=True)
    _save_table(interactions, cfg.output_dir / "subgroup_effects.csv")

    return {"quadratic": quad, "threshold": thresh,
            "quantile_bands": qband, "interactions": interactions}


def stage_causality(df: pd.DataFrame, cfg: Config):
    """DML causal effects of voice variables + FWL/OLS benchmark.

    Controls are the basic operational inputs; each voice variable's partial
    effect is estimated with cross-fitted ML nuisances (Neyman-orthogonal).
    """
    from voxfrontier.causality.dml import dml_partially_linear, fwl_ols_benchmark

    valid = df[df[BCC_EFF].notna()].copy()
    voice_vars = [MEAN_F0, HNR, JITTER, SPEECH_RATE]
    controls = [STREAM_HOURS, PRICE, VIEWERS]

    dml = dml_partially_linear(
        valid, voice_vars, controls, BCC_EFF,
        n_folds=5, learner="rf", seed=cfg.seed,
    )
    _save_table(dml, cfg.output_dir / "dml_effects.csv")

    bench = fwl_ols_benchmark(valid, voice_vars, controls, BCC_EFF)
    _save_table(bench, cfg.output_dir / "fwl_ols_benchmark.csv")

    return {"dml": dml, "benchmark": bench}


def stage_simulation(df: pd.DataFrame, cfg: Config, nonlinear):
    valid = df[df[BCC_EFF].notna()].copy()
    voice_vars = [MEAN_F0, HNR, JITTER, SPEECH_RATE]
    feature_cols = list(BASIC_MODEL_INPUTS) + voice_vars

    high = valid[valid[BCC_EFF] >= 0.95]
    efficient_means = {
        col: float(high[col].mean()) if len(high) else float(valid[col].mean())
        for col in voice_vars
    }

    valleys = {}
    quad = nonlinear["quadratic"]
    for col in voice_vars:
        row = quad.loc[quad["variable"] == col]
        if len(row) and row.iloc[0]["shape"] == "u_shape":
            valleys[col] = float(row.iloc[0]["extremum_raw"])
        else:  # fall back to the empirical anti-mode (decile with min mean eff)
            bins = pd.qcut(valid[col], 10, duplicates="drop")
            means = valid.groupby(bins, observed=True)[BCC_EFF].mean()
            valleys[col] = float(means.idxmin().mid)

    predictor = fit_predictor(valid, feature_cols, BCC_EFF, cfg.seed)
    scenarios = run_scenarios(
        valid, voice_vars, feature_cols, efficient_means, valleys, predictor, BCC_EFF
    )
    _save_table(scenarios, cfg.output_dir / "scenario_simulation.csv")

    diag = diagnose_inefficient(
        valid, voice_vars, efficient_means, valleys, BCC_EFF
    )
    _save_table(diag, cfg.output_dir / "unit_diagnostics.csv")

    return {"scenarios": scenarios, "diagnostics": diag,
            "efficient_means": efficient_means, "valleys": valleys}


# ---------------------------------------------------------------------- #
# figures
# ---------------------------------------------------------------------- #
def stage_figures(df: pd.DataFrame, cfg: Config, results: dict) -> None:
    import matplotlib.pyplot as plt

    from voxfrontier.utils.plotting import PALETTE

    setup_style(cfg.dpi, cfg.plot_language)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    valid = df[df[BCC_EFF].notna()]

    # fig 1: efficiency distribution by DEA model spec (A/B/C)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    specs = {"A": BASIC_MODEL_INPUTS, "B": VOICE_MODEL_INPUTS,
             "C": FULL_MODEL_INPUTS}
    Y = valid[list(OUTPUTS)].to_numpy(float)
    for ax, (name, inputs) in zip(axes, specs.items()):
        from voxfrontier.dea.models import dea_efficiency, prepare_positive

        X = prepare_positive(valid[list(inputs)].to_numpy(float))
        Yp = prepare_positive(Y)
        bcc = dea_efficiency(X, Yp, "BCC", cfg.dea_backend)
        ax.hist(bcc, bins=20, color=PALETTE["primary"], alpha=0.85)
        mean_v = np.nanmean(bcc)
        if np.isfinite(mean_v):
            ax.axvline(mean_v, color=PALETTE["bad"], ls="--",
                       label=f"mean={mean_v:.3f}")
        ax.set_title(f"Model {name} (BCC)")
        ax.set_xlabel("efficiency")
        if np.isfinite(mean_v):
            ax.legend()
    axes[0].set_ylabel("sessions")
    fig.suptitle("DEA efficiency by model specification")
    fig.savefig(cfg.figure_dir / "dea_efficiency.png", bbox_inches="tight")
    plt.close(fig)

    # fig 2: Shapley contributions
    shapley = results["attribution"]["shapley"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE["voice"] if k == "voice" else PALETTE["basic"]
              for k in shapley["kind"]]
    ax.barh(shapley["group"], shapley["contribution_pct"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("contribution (%)")
    ax.set_title(
        f"Shapley decomposition of mean BCC efficiency "
        f"(voice share {results['attribution']['voice_share_pct']:.1f}%)"
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=PALETTE["voice"], label="voice"),
                 Patch(color=PALETTE["basic"], label="basic")]
    )
    fig.savefig(cfg.figure_dir / "shapley_decomposition.png", bbox_inches="tight")
    plt.close(fig)

    # fig 3: quadratic fits (U-shapes)
    quad = results["nonlinear"]["quadratic"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, [MEAN_F0, HNR, JITTER, SPEECH_RATE]):
        x = valid[col].to_numpy(float)
        y = valid[BCC_EFF].to_numpy(float)
        ax.scatter(x, y, s=14, alpha=0.45, color=PALETTE["neutral"])
        row = quad.loc[quad["variable"] == col].iloc[0]
        if np.isfinite(row["extremum_raw"]):
            ax.axvline(row["extremum_raw"], color=PALETTE["bad"], ls=":",
                       label=f"{row['shape']} @ {row['extremum_raw']:.1f}")
        ax.set_title(f"{col} (p={row['p_b2']:.3f})")
        ax.set_xlabel(col)
        ax.set_ylabel("BCC efficiency")
        if np.isfinite(row["extremum_raw"]):
            ax.legend()
    fig.suptitle("Voice variables vs efficiency: quadratic fits")
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "quadratic_shapes.png", bbox_inches="tight")
    plt.close(fig)

    # fig 4: scenario comparison
    scen = results["simulation"]["scenarios"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [PALETTE["good"] if g > 0 else PALETTE["bad"] if g < 0
              else PALETTE["neutral"] for g in scen["gain_vs_baseline"]]
    ax.bar(scen["scenario"], scen["mean_efficiency"], color=colors)
    ax.axhline(scen.loc[scen["scenario"] == "baseline",
                        "mean_efficiency"].iloc[0],
               color=PALETTE["neutral"], ls="--", label="baseline")
    ax.set_ylabel("mean predicted efficiency")
    ax.set_title("Counterfactual scenarios")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.savefig(cfg.figure_dir / "scenarios.png", bbox_inches="tight")
    plt.close(fig)

    # fig 5: executive dashboard (2x2): DML effects, Shapley, U-shape, scenarios
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) DML effects with 95% CI
    ax = axes[0, 0]
    dml = results["causality"]["dml"].sort_values("dml_estimate")
    y_pos = np.arange(len(dml))
    ax.errorbar(
        dml["dml_estimate"], y_pos,
        xerr=[dml["dml_estimate"] - dml["ci_low"],
              dml["ci_high"] - dml["dml_estimate"]],
        fmt="o", color=PALETTE["primary"], capsize=4,
    )
    ax.axvline(0, color=PALETTE["neutral"], ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dml["treatment"])
    ax.set_xlabel("DML partial effect on BCC efficiency (95% CI)")
    ax.set_title("(a) Causal effects (Double Machine Learning)")

    # (b) Shapley shares
    ax = axes[0, 1]
    shapley = results["attribution"]["shapley"]
    colors_b = [PALETTE["voice"] if k == "voice" else PALETTE["basic"]
                for k in shapley["kind"]]
    ax.barh(shapley["group"], shapley["contribution_pct"], color=colors_b)
    ax.invert_yaxis()
    ax.set_xlabel("contribution (%)")
    ax.set_title("(b) Shapley attribution")

    # (c) U-shape for mean_f0
    ax = axes[1, 0]
    x = valid[MEAN_F0].to_numpy(float)
    yv = valid[BCC_EFF].to_numpy(float)
    ax.scatter(x, yv, s=14, alpha=0.45, color=PALETTE["neutral"])
    row = results["nonlinear"]["quadratic"]
    row = row.loc[row["variable"] == MEAN_F0].iloc[0]
    if np.isfinite(row["extremum_raw"]):
        ax.axvline(row["extremum_raw"], color=PALETTE["bad"], ls=":",
                   label=f"valley @ {row['extremum_raw']:.0f} Hz")
        ax.legend()
    ax.set_xlabel(MEAN_F0)
    ax.set_ylabel("BCC efficiency")
    ax.set_title("(c) Non-linearity (mean_f0)")

    # (d) scenario gains
    ax = axes[1, 1]
    gains = results["simulation"]["scenarios"]
    colors_g = [PALETTE["good"] if g > 0 else PALETTE["bad"] if g < 0
                else PALETTE["neutral"] for g in gains["gain_vs_baseline"]]
    ax.barh(gains["scenario"], gains["gain_vs_baseline"], color=colors_g)
    ax.axvline(0, color=PALETTE["neutral"], lw=1)
    ax.set_xlabel("mean efficiency gain vs baseline")
    ax.set_title("(d) Counterfactual scenarios")

    fig.suptitle("VoxFrontier executive dashboard", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(cfg.figure_dir / "dashboard.png", bbox_inches="tight")
    plt.close(fig)

    logger.info("figures written to %s", cfg.figure_dir)


# ---------------------------------------------------------------------- #
# entry point
# ---------------------------------------------------------------------- #
def run_all(cfg: Config, skip_synth: bool = False) -> dict:
    """Execute the full pipeline and persist every result table."""

    started_at = datetime.now(timezone.utc)
    logger.info("voxfrontier v%s | seed=%d", __version__, cfg.seed)
    set_seed(cfg.seed)

    df = stage_dea(stage_synth(cfg), cfg) if not skip_synth else None
    if df is None:  # pragma: no cover
        raise RuntimeError("skip_synth requires a pre-existing dataset")

    attribution = stage_attribution(df, cfg)
    nonlinear = stage_nonlinear(df, cfg)
    causal = stage_causality(df, cfg)
    simulation = stage_simulation(df, cfg, nonlinear)

    results = {
        "attribution": attribution,
        "nonlinear": nonlinear,
        "causality": causal,
        "simulation": simulation,
    }
    stage_figures(df, cfg, results)

    # machine-readable summary
    summary = {
        "n_sessions": int(len(df)),
        "mean_bcc": float(df[BCC_EFF].mean()),
        "efficient_units": int((df[BCC_EFF] >= 0.999).sum()),
        "shapley_voice_share_pct": attribution["voice_share_pct"],
        "dml_effects": {
            row["treatment"]: row["dml_estimate"]
            for _, row in causal["dml"].iterrows()
        },
        "u_valleys": {
            k: (float(v) if np.isfinite(v) else None)
            for k, v in simulation["valleys"].items()
        },
        "best_scenario": simulation["scenarios"].loc[
            simulation["scenarios"]["gain_vs_baseline"].idxmax(), "scenario"
        ],
        "best_scenario_gain": float(
            simulation["scenarios"]["gain_vs_baseline"].max()
        ),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # reproducibility manifest (SHA-256 fingerprints)
    from voxfrontier.utils.manifest import write_manifest

    write_manifest(
        cfg.output_dir, cfg.data_dir / "synthetic_sessions.csv",
        cfg.seed, summary, started_at,
    )
    logger.info("pipeline complete: %s", json.dumps(summary))
    return results
