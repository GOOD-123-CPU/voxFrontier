"""Synthetic data generator.

This module is the ONLY source of data shipped in the open-source repository.
It produces a session-level table whose *structure, magnitude, distribution
shape and latent relationships* mirror the original research dataset — while
containing no real streamer names, room identifiers, or any other
re-identifiable information.

Design (calibrated to the descriptive statistics of the original data):

* 207 sessions, 3 balanced product categories, 4 price tiers;
* acoustic variables sampled with realistic skew and cross-correlations
  (HNR negatively correlated with jitter/shimmer, F0 correlated with speech
  rate, etc.);
* a *latent* efficiency mechanism: log-viewers and acoustic features drive
  conversion and sales, and the voice effect on efficiency is deliberately
  **non-monotonic** (U-shaped in mean_f0 / speech_rate with a valley, and a
  threshold in HNR / jitter) so that every downstream analysis stage has a
  real signal to recover;
* DEA outputs (sales index, conversion %) are strictly positive.

The generator is deterministic given ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from voxfrontier.data.schema import (
    ACTIVE_FOLLOWERS,
    BCC_EFF,
    CONVERSION,
    CONVERSION_PCT,
    FAN_ACTIVITY,
    FOLLOWERS,
    GENDER,
    HNR,
    JITTER,
    MEAN_F0,
    PRICE,
    PRICE_TIER,
    PRODUCT_TYPE,
    RANGE_F0,
    SALES_INDEX,
    SALES_PER_VIEWER,
    SD_F0,
    SHIMMER,
    SPEECH_RATE,
    STREAM_HOURS,
    TIME_SLOT,
    UNIT_ID,
    VIEWERS,
    VOICE_INDEX,
)

PRODUCT_TYPES = ("apparel", "food", "daily_goods")
PRICE_TIERS = ("low", "mid", "mid_high", "high")
TIME_SLOTS = ("morning", "afternoon", "evening", "late_night")


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def _u_effect(x: np.ndarray, valley: float, depth: float) -> np.ndarray:
    """Normalised U-shaped (valley) effect of ``x`` around ``valley``."""
    rel = (x - valley) / max(valley, 1e-9)
    return depth * np.abs(rel)


def generate_synthetic(n_units: int = 207, seed: int = 42) -> pd.DataFrame:
    """Generate the synthetic session-level dataset.

    Parameters
    ----------
    n_units:
        Number of livestream sessions (decision-making units).
    seed:
        Random seed; the output is fully determined by it.

    Returns
    -------
    pd.DataFrame
        Table with the canonical schema (see :mod:`voxfrontier.data.schema`).
    """
    rng = np.random.default_rng(seed)
    n = n_units

    # ------------------------------------------------------------------
    # 1. Categorical attributes (balanced, like the original 69/69/69)
    # ------------------------------------------------------------------
    product_type = np.array(PRODUCT_TYPES)[rng.integers(0, 3, size=n)]
    gender = rng.integers(0, 2, size=n)

    # log-normal price; tiered
    price = _clip(rng.lognormal(mean=3.6, sigma=1.1, size=n), 2.0, 1000.0)
    price_tier = np.select(
        [price <= 30, price <= 80, price <= 150],
        ["low", "mid", "mid_high"],
        default="high",
    )

    hour = rng.choice([9, 14, 19, 23], size=n, p=[0.35, 0.15, 0.35, 0.15])
    time_slot = np.select(
        [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 17),
         (hour >= 17) & (hour < 22)],
        ["morning", "afternoon", "evening"],
        default="late_night",
    )

    # ------------------------------------------------------------------
    # 2. Operational variables
    # ------------------------------------------------------------------
    followers = _clip(rng.lognormal(3.9, 1.05, n), 0.5, 320.0)          # 10k
    stream_hours = _clip(rng.gamma(2.0, 3.9, n) + 0.4, 0.4, 24.0)
    viewers = np.round(_clip(rng.lognormal(6.9, 1.5, n), 10, 65000)).astype(int)

    # ------------------------------------------------------------------
    # 3. Acoustic variables with realistic covariance structure
    # ------------------------------------------------------------------
    mean_f0 = _clip(rng.normal(238, 52, n), 128, 425)
    # stronger HNR => less jitter & shimmer
    hnr = _clip(rng.normal(9.2, 2.6, n), 0.8, 16.5)
    jitter = _clip(3.55 - 0.14 * hnr + rng.normal(0, 0.35, n), 1.1, 4.0)
    shimmer = _clip(19.5 - 0.62 * hnr + rng.normal(0, 1.6, n), 8.0, 19.5)
    speech_rate = _clip(
        3.4 + 0.006 * mean_f0 + rng.normal(0, 0.72, n), 1.7, 6.9
    )
    range_f0 = _clip(rng.normal(470, 82, n), 60, 562)
    sd_f0 = _clip(0.16 * range_f0 + rng.normal(0, 9, n), 10, 152)

    # ------------------------------------------------------------------
    # 4. Latent voice effect on conversion (non-monotonic, threshold-like)
    #    Calibrated so the U-valleys / thresholds land inside the data:
    #      mean_f0 valley ~ 295 Hz ; speech_rate valley ~ 5.5 chars/s
    #      HNR threshold ~ 11 dB   ; jitter threshold ~ 2.45 %
    # ------------------------------------------------------------------
    voice_bonus = (
        +0.055 * _u_effect(mean_f0, 295.0, 1.0)          # U-shape (valley)
        + 0.030 * _u_effect(speech_rate, 5.5, 1.0)       # U-shape (valley)
        + 0.045 * np.maximum(hnr - 11.0, 0.0) / 5.0      # threshold (above)
        + 0.035 * np.maximum(2.45 - jitter, 0.0) / 1.4   # threshold (below)
    )

    # ------------------------------------------------------------------
    # 5. Outcomes driven by inputs + voice (all strictly positive)
    # ------------------------------------------------------------------
    base_conv = (
        0.10
        + 0.010 * np.log1p(viewers) / np.log(10)
        - 0.012 * np.log(price)
        + 0.015 * np.log1p(followers)
        + voice_bonus
        + rng.normal(0, 0.035, n)
    )
    conversion = _clip(base_conv, 0.03, 0.30)

    sales_index = np.round(
        _clip(
            viewers * conversion * rng.lognormal(0.4, 0.35, n),
            1300, 150000,
        )
    ).astype(int)

    active_followers = np.round(_clip(followers * rng.beta(1.6, 7.0, n), 0, 95), 1)
    fan_activity = np.where(
        followers > 0, active_followers / np.maximum(followers, 0.1), 0.0
    )
    sales_per_viewer = sales_index / np.maximum(viewers, 1)

    # composite voice quality index (HNR+, jitter-/shimmer-, rate mildly +)
    def _minmax(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / max(hi - lo, 1e-9)

    voice_index = (
        _minmax(hnr)
        - 0.5 * _minmax(jitter)
        - 0.5 * _minmax(shimmer)
        + 0.3 * _minmax(speech_rate)
    )

    df = pd.DataFrame(
        {
            UNIT_ID: np.arange(1, n + 1),
            PRODUCT_TYPE: product_type,
            PRICE: np.round(price, 1),
            GENDER: gender,
            TIME_SLOT: time_slot,
            PRICE_TIER: price_tier,
            FOLLOWERS: np.round(followers, 1),
            ACTIVE_FOLLOWERS: active_followers,
            STREAM_HOURS: np.round(stream_hours, 2),
            VIEWERS: viewers,
            SALES_INDEX: sales_index,
            CONVERSION: np.round(conversion, 4),
            CONVERSION_PCT: np.round(conversion * 100, 1),
            FAN_ACTIVITY: np.round(fan_activity, 4),
            SALES_PER_VIEWER: np.round(sales_per_viewer, 2),
            MEAN_F0: np.round(mean_f0, 2),
            RANGE_F0: np.round(range_f0, 2),
            SD_F0: np.round(sd_f0, 2),
            HNR: np.round(hnr, 2),
            JITTER: np.round(jitter, 4),
            SHIMMER: np.round(shimmer, 4),
            SPEECH_RATE: np.round(speech_rate, 3),
            VOICE_INDEX: np.round(voice_index, 4),
        }
    )
    return df


def ensure_efficiency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing efficiency columns with NaN placeholders (pre-DEA)."""
    for col in (BCC_EFF, "ccr_efficiency", "scale_efficiency"):
        if col not in df.columns:
            df[col] = np.nan
    return df
