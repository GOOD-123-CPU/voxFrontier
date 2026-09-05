"""Tests for DEA: known solutions, backend agreement, scale efficiency."""

from __future__ import annotations

import numpy as np
import pytest

from voxfrontier.dea.models import dea_efficiency, prepare_positive, run_three_models


def test_perfectly_proportional_units_are_efficient():
    """2 DMUs on the same ray: both must be CCR/BCC-efficient (=1)."""
    X = np.array([[1.0], [2.0]])
    Y = np.array([[1.0], [2.0]])
    for model in ("CCR", "BCC"):
        eff = dea_efficiency(X, Y, model, backend="scipy")
        assert np.allclose(eff, 1.0, atol=1e-6), (model, eff)


def test_dominated_unit_is_inefficient():
    """DMU2 uses twice the input for the same output => efficiency 0.5."""
    X = np.array([[1.0], [2.0]])
    Y = np.array([[1.0], [1.0]])
    eff = dea_efficiency(X, Y, "CCR", backend="scipy")
    assert np.isclose(eff[1], 0.5, atol=1e-6)
    assert np.isclose(eff[0], 1.0, atol=1e-6)


def test_bcc_at_least_ccr():
    """VRS frontier envelops the CRS frontier: BCC >= CCR for every DMU."""
    rng = np.random.default_rng(3)
    X = rng.uniform(1, 5, size=(40, 2))
    Y = rng.uniform(1, 5, size=(40, 2))
    ccr = dea_efficiency(X, Y, "CCR", backend="scipy")
    bcc = dea_efficiency(X, Y, "BCC", backend="scipy")
    assert np.all(bcc >= ccr - 1e-6)


def test_prepare_positive_sanitises():
    X = np.array([[1.0, np.nan], [0.0, 5.0], [-2.0, 7.0]])
    out = prepare_positive(X)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


def test_pulp_backend_matches_scipy(small_dea_data=None):
    """Optional: when pulp is installed both backends must agree."""
    pytest.importorskip("pulp")
    rng = np.random.default_rng(11)
    X = rng.uniform(1, 4, size=(25, 2))
    Y = rng.uniform(1, 4, size=(25, 2))
    eff_sp = dea_efficiency(X, Y, "BCC", backend="scipy")
    eff_pu = dea_efficiency(X, Y, "BCC", backend="pulp")
    assert np.allclose(eff_sp, eff_pu, atol=2e-3)


def test_run_three_models_shape(synth_df):
    from voxfrontier.data.schema import (
        BASIC_MODEL_INPUTS,
        FULL_MODEL_INPUTS,
        OUTPUTS,
    )

    res = run_three_models(
        synth_df.head(60),
        {"A": BASIC_MODEL_INPUTS, "C": FULL_MODEL_INPUTS},
        OUTPUTS,
        backend="scipy",
    )
    for name in ("A", "C"):
        assert len(res[name]["bcc"]) == 60
        assert np.nanmean(res[name]["bcc"]) > 0.5
        assert np.nanmean(res[name]["scale"]) <= 1.0 + 1e-6
