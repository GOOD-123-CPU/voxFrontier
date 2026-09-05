"""Configuration loading and global random-seed management."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_RELPATH = Path("config") / "default.yaml"


def repo_root() -> Path:
    """Repository root (the folder that contains ``pyproject.toml``).

    Works both from an installed package and from a source checkout. When the
    package is installed site-packages-only, fall back to the current working
    directory so that relative ``data/`` / ``output/`` paths still resovoxfrontier.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass
class Config:
    """Typed view over ``config/default.yaml``."""

    seed: int = 42
    raw: dict[str, Any] = field(default_factory=dict)

    # resolved paths
    data_dir: Path = Path("data/synthetic")
    output_dir: Path = Path("output")
    figure_dir: Path = Path("figures")

    # data generator
    n_units: int = 207
    product_types: tuple = ("apparel", "food", "daily_goods")

    # dea
    dea_backend: str = "auto"

    # attribution
    shapley_max_vars: int = 10
    rf_trees: int = 300
    use_shap: bool = True

    # plots
    dpi: int = 150
    plot_language: str = "en"

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Config:
        """Load config from YAML (default: bundled ``config/default.yaml``)."""
        cfg_path = Path(path) if path else repo_root() / DEFAULT_CONFIG_RELPATH
        raw: dict[str, Any] = {}
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

        root = repo_root()
        paths = raw.get("paths", {}) or {}

        cfg = cls(
            seed=int(raw.get("seed", 42)),
            raw=raw,
            data_dir=root / paths.get("data_dir", "data/synthetic"),
            output_dir=root / paths.get("output_dir", "output"),
            figure_dir=root / paths.get("figure_dir", "figures"),
            n_units=int((raw.get("data") or {}).get("n_units", 207)),
            dea_backend=str((raw.get("dea") or {}).get("backend", "auto")),
            shapley_max_vars=int(
                (raw.get("attribution") or {}).get("shapley_max_vars", 10)
            ),
            rf_trees=int((raw.get("attribution") or {}).get("rf_trees", 300)),
            use_shap=bool((raw.get("attribution") or {}).get("use_shap", True)),
            dpi=int((raw.get("plot") or {}).get("dpi", 150)),
            plot_language=str((raw.get("plot") or {}).get("language", "en")),
        )
        return cfg


def set_seed(seed: int) -> None:
    """Seed python's :mod:`random` and :mod:`numpy` for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # sklearn uses its own global state via numpy, but be explicit
        from sklearn.utils import check_random_state  # noqa: F401

        _ = check_random_state(seed)
    except Exception:  # pragma: no cover - sklearn always present in practice
        pass
