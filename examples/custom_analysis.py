"""Custom analysis example: run VoxFrontier as a library on the synthetic
dataset and extend it with your own estimator.

Run from the repository root:

    python examples/custom_analysis.py

Everything below operates on synthetic data only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from voxfrontier.config import Config, set_seed
from voxfrontier.data.schema import BCC_EFF
from voxfrontier.data.synth import generate_synthetic
from voxfrontier.dea.models import run_three_models
from voxfrontier.utils.logging import get_logger
from voxfrontier.utils.plotting import setup_style

logger = get_logger(__name__)

VOICE_PREFIXES = ("mean_f0", "range_f0", "sd_f0", "hnr", "jitter",
                  "shimmer", "speech_rate", "voice_index")


def main() -> None:
    # 1. Build a config (defaults, or point to your own YAML) ------------
    cfg = Config.load()  # Config.load("my-config.yaml") for overrides
    set_seed(cfg.seed)
    setup_style(dpi=cfg.dpi, language=cfg.plot_language)

    # 2. Generate the synthetic dataset (deterministic under cfg.seed) ---
    df: pd.DataFrame = generate_synthetic(cfg.n_units, cfg.seed)
    logger.info("synthetic dataset: %d sessions", len(df))

    # 3. Compute DEA efficiencies for every model specification ----------
    #    Spec "B" is the voice-only input set; reuse it downstream.
    specs = {"A": None, "B": None, "C": None}  # replaced below from schema
    from voxfrontier.data.schema import (  # noqa: E402  (kept close to use)
        BASIC_MODEL_INPUTS, FULL_MODEL_INPUTS, OUTPUTS, VOICE_MODEL_INPUTS,
    )

    specs = {
        "A": BASIC_MODEL_INPUTS,
        "B": VOICE_MODEL_INPUTS,
        "C": FULL_MODEL_INPUTS,
    }
    results = run_three_models(df, specs, OUTPUTS, cfg.dea_backend)
    df[BCC_EFF] = results["B"]["bcc"]

    # 4. Your own analysis on top: correlation profile of voice vars -----
    voice_cols = [c for c in df.columns if c.startswith(VOICE_PREFIXES)]
    voice_corr = (
        df[voice_cols]
        .corrwith(df[BCC_EFF])
        .sort_values(key=abs, ascending=False)
    )
    print("Top |correlations| with BCC efficiency (synthetic data):")
    print(voice_corr.head(8).round(3).to_string())

    # 5. Persist wherever you like ---------------------------------------
    out = Path(cfg.output_dir) / "custom_example.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    voice_corr.rename("corr_with_bcc").to_csv(out, header=True)
    logger.info("saved %s", out)

    # 6. Reminder: reuse the full pipeline when possible ------------------
    # The CLI (`vxf run-all`) already chains DEA -> attribution ->
    # nonlinearity -> DML -> simulation with a run manifest; prefer it for
    # production analyses and use the library API to add estimators.


if __name__ == "__main__":
    main()
