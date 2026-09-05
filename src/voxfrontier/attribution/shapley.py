"""Exact Shapley decomposition of average DEA efficiency over input groups.

For each candidate subset ``S`` of the ``k`` input groups, compute the mean
BCC efficiency of a DEA restricted to the inputs in ``S``. The Shapley value
of group ``i`` is its expectation-weighted marginal contribution across all
subsets:

    phi_i = sum_{S subset of N\\{i}} w(|S|) * [v(S u {i}) - v(S)]

with ``w(|S|) = |S|! (k-|S|-1)! / k!``.

Unlike a hard-coded lookup, the values here are **always computed from the
data** — the number of DEA solves is ``2^k - 1`` (k=7 -> 127 LPs per subset is
cheap with the HiGHS backend; for larger k use permutation sampling).

The resulting contributions sum (in absolute value) to the efficiency spread
v(N) - v(empty set), which the unit tests verify.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from voxfrontier.dea.models import dea_efficiency, prepare_positive

logger = logging.getLogger("voxfrontier.shapley")

BASIC_GROUPS_DEFAULT = ("stream_hours", "price", "traffic")
VOICE_GROUPS_DEFAULT = ("voice_f0", "voice_quality", "voice_stability", "speech_rate")


def shapley_decomposition(
    df: pd.DataFrame,
    groups: dict[str, Sequence[str]],
    outputs: Sequence[str],
    backend: str = "auto",
    basic_groups: Sequence[str] = BASIC_GROUPS_DEFAULT,
    voice_groups: Sequence[str] = VOICE_GROUPS_DEFAULT,
) -> pd.DataFrame:
    """Exact Shapley decomposition over input groups.

    Parameters
    ----------
    df:
        Session-level DataFrame.
    groups:
        Mapping group name -> input columns belonging to the group.
    outputs:
        Output column names for the DEA.
    backend:
        DEA solver backend.
    basic_groups / voice_groups:
        Group names tagged as "basic" vs "voice" for aggregated reporting.

    Returns
    -------
    pd.DataFrame
        Columns: ``group, shapley_value, contribution_pct, kind`` sorted by
        contribution (descending).
    """
    names: list[str] = list(groups.keys())
    k = len(names)
    if k > 20:
        raise ValueError(
            f"{k} groups is too many for exact Shapley (2^{k}-1 DEA solves). "
            "Reduce the number of groups or implement permutation sampling."
        )

    Y = prepare_positive(df[list(outputs)].to_numpy(float))
    col_of = {g: list(cols) for g, cols in groups.items()}

    def value_of(subset: tuple) -> float:
        if not subset:
            return 0.0
        cols: list[str] = []
        for g in subset:
            cols.extend(col_of[g])
        X = prepare_positive(df[cols].to_numpy(float))
        eff = dea_efficiency(X, Y, "BCC", backend)
        return float(np.nanmean(eff))

    total_subsets = 2 ** k - 1
    logger.info("Shapley: computing %d subset DEA evaluations ...", total_subsets)

    v = {}
    for size in range(1, k + 1):
        for combo in combinations(range(k), size):
            subset = tuple(sorted(names[j] for j in combo))
            v[subset] = value_of(subset)

    shapley: dict[str, float] = {}
    for i, name in enumerate(names):
        others = [j for j in range(k) if j != i]
        sv = 0.0
        for size in range(0, k):
            for combo in combinations(others, size):
                s_without = tuple(sorted(names[j] for j in combo))
                s_with = tuple(sorted(s_without + (name,)))
                marginal = v.get(s_with, 0.0) - v.get(s_without, 0.0)
                weight = (
                    math.factorial(size) * math.factorial(k - size - 1)
                ) / math.factorial(k)
                sv += weight * marginal
        shapley[name] = sv

    total_abs = sum(abs(x) for x in shapley.values()) or 1.0
    out = pd.DataFrame(
        {
            "group": names,
            "shapley_value": [shapley[g] for g in names],
            "contribution_pct": [abs(shapley[g]) / total_abs * 100 for g in names],
        }
    )
    kind = {g: ("voice" if g in set(voice_groups) else "basic") for g in names}
    out["kind"] = out["group"].map(kind)
    return out.sort_values("contribution_pct", ascending=False).reset_index(drop=True)
