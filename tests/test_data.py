"""Tests for the synthetic data generator and its privacy guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from voxfrontier.data.schema import (
    CONVERSION_PCT,
    HNR,
    JITTER,
    MEAN_F0,
    SALES_INDEX,
    SPEECH_RATE,
    VIEWERS,
    VOICE_VARS,
)
from voxfrontier.data.synth import generate_synthetic


def test_shape_and_columns(synth_df: pd.DataFrame):
    assert len(synth_df) == 207
    for col in (MEAN_F0, HNR, JITTER, SPEECH_RATE, VIEWERS, SALES_INDEX,
                CONVERSION_PCT):
        assert col in synth_df.columns, f"missing column {col}"


def test_no_real_identifiers(synth_df: pd.DataFrame):
    """Privacy: no name-like columns, unit ids are plain integers 1..n."""
    forbidden = ("达人", "name", "nickname", "room", "channel", "account")
    for col in synth_df.columns:
        low = col.lower()
        assert not any(tok in low for tok in forbidden), f"suspicious column {col}"
    ids = synth_df["unit_id"].to_numpy()
    assert ids.min() == 1 and ids.max() == len(synth_df)
    assert np.array_equal(ids, np.arange(1, len(synth_df) + 1))


def test_dea_inputs_strictly_positive(synth_df: pd.DataFrame):
    from voxfrontier.data.schema import BASIC_MODEL_INPUTS, FULL_MODEL_INPUTS, OUTPUTS

    for cols in (BASIC_MODEL_INPUTS, OUTPUTS, FULL_MODEL_INPUTS):
        values = synth_df[cols].to_numpy(float)
        assert np.all(values > 0), "DEA inputs/outputs must be positive"


def test_deterministic_given_seed():
    a = generate_synthetic(50, seed=7)
    b = generate_synthetic(50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_plausible_magnitudes(synth_df: pd.DataFrame):
    """Guard against drifting away from the calibrated distributions."""
    assert 200 < synth_df[MEAN_F0].mean() < 280          # Hz
    assert 6 < synth_df[HNR].mean() < 13                 # dB
    assert 1.5 < synth_df[JITTER].mean() < 3.2           # %
    assert 4 < synth_df[SPEECH_RATE].mean() < 6          # chars/s
    assert 0.03 < synth_df["conversion_rate"].mean() < 0.30


def test_latent_signal_exists(synth_with_eff: pd.DataFrame):
    """The generator must embed a recoverable voice->efficiency signal."""
    df = synth_with_eff
    assert df["bcc_efficiency"].between(0, 1.001).all()
    high = df[df["bcc_efficiency"] >= df["bcc_efficiency"].quantile(0.75)]
    low = df[df["bcc_efficiency"] <= df["bcc_efficiency"].quantile(0.25)]
    assert not np.isclose(
        high[VOICE_VARS].mean().sum(), low[VOICE_VARS].mean().sum()
    ), "voice variables should separate efficiency groups"
