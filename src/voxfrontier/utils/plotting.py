"""Cross-platform matplotlib styling.

Figures must render identically on Windows / macOS / Linux CI without relying
on Windows-only fonts (SimHei etc.). Strategy:

* default to English labels (fully portable);
* if CJK characters are requested, try a list of commonly available CJK
  fonts across the three major OSes and fall back gracefully.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe (CI, containers)
import matplotlib.pyplot as plt  # noqa: E402

CJK_FONT_CANDIDATES: Sequence[str] = (
    # Windows
    "Microsoft YaHei",
    "SimHei",
    # macOS
    "PingFang SC",
    "Hiragino Sans GB",
    # Linux (common packages: fonts-noto-cjk / fonts-wqy-*)
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
)


def setup_style(dpi: int = 150, language: str = "en") -> None:
    """Apply a consistent, portable plot style."""
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.autolayout": False,
        }
    )
    if language.lower() in {"zh", "cn", "chinese"}:
        available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        chosen = [f for f in CJK_FONT_CANDIDATES if f in available]
        if chosen:
            plt.rcParams["font.sans-serif"] = chosen + plt.rcParams["font.sans-serif"]
        # DejaVu Sans still covers latin glyphs; minus sign must be ASCII-safe
        plt.rcParams["axes.unicode_minus"] = False


PALETTE = {
    "primary": "#2f6fde",
    "voice": "#e05555",
    "basic": "#3f83d6",
    "good": "#2e9e5b",
    "bad": "#d05050",
    "neutral": "#8a8f98",
    "accent": "#f2a33c",
}
