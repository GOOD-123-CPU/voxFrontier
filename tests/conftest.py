"""Test configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# make src/ importable without installation (source-checkout friendly)
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voxfrontier.data.synth import generate_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synth_df() -> pd.DataFrame:
    return generate_synthetic(n_units=207, seed=42)


@pytest.fixture(scope="session")
def synth_with_eff(synth_df) -> pd.DataFrame:
    from voxfrontier.data.schema import (
        OUTPUTS,
        VOICE_MODEL_INPUTS,
    )
    from voxfrontier.dea.models import run_three_models

    res = run_three_models(
        synth_df,
        {"B": VOICE_MODEL_INPUTS},
        OUTPUTS,
        backend="scipy",
    )
    df = synth_df.copy()
    df["bcc_efficiency"] = res["B"]["bcc"]
    return df
