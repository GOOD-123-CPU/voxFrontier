"""Tests for the scenario simulator and unit diagnostics."""

from __future__ import annotations


def test_scenarios_table_structure(synth_with_eff):
    from voxfrontier.data.schema import BASIC_MODEL_INPUTS, MEAN_F0
    from voxfrontier.simulation.scenarios import (
        SCENARIOS,
        fit_predictor,
        run_scenarios,
    )

    df = synth_with_eff
    voice_vars = [MEAN_F0, "hnr", "jitter", "speech_rate"]
    feature_cols = list(BASIC_MODEL_INPUTS) + voice_vars
    high = df[df["bcc_efficiency"] >= 0.95]
    eff_means = {c: float(high[c].mean()) for c in voice_vars}
    valleys = {c: float(df[c].median()) for c in voice_vars}
    predictor = fit_predictor(df, feature_cols, "bcc_efficiency", seed=42)

    out = run_scenarios(
        df, voice_vars, feature_cols, eff_means, valleys, predictor,
        "bcc_efficiency",
    )
    assert list(out["scenario"]) == list(SCENARIOS)
    # baseline gain is exactly zero by definition
    assert out.loc[out["scenario"] == "baseline", "gain_vs_baseline"].abs().max() < 1e-12
    # all efficiencies stay within a sane band
    assert out["mean_efficiency"].between(0, 1.2).all()


def test_diagnostics_only_targets_low_efficiency(synth_with_eff):
    from voxfrontier.data.schema import MEAN_F0
    from voxfrontier.simulation.scenarios import diagnose_inefficient

    df = synth_with_eff
    voice_vars = [MEAN_F0, "hnr", "jitter", "speech_rate"]
    eff_means = {c: float(df[c].mean()) for c in voice_vars}
    valleys = {c: float(df[c].median()) for c in voice_vars}
    diag = diagnose_inefficient(df, voice_vars, eff_means, valleys)
    assert (diag["efficiency"] < 0.8).all()
    assert diag["n_issues"].min() >= 0
