"""Input-oriented DEA (CCR / BCC) via linear programming.

Two interchangeable backends:

* ``scipy`` (default): one LP per DMU solved with ``scipy.optimize.linprog``
  (HiGHS). No extra dependency, fast, CI-friendly.
* ``pulp``: exact formulation with CBC (install with ``pip install pulp``).
  Useful for reproducibility against textbook MILP solvers.

Formulation (input-oriented, DMU ``o``):

    min theta
    s.t.  sum_j lambda_j * x_j  <= theta * x_o        (m input constraints)
          sum_j lambda_j * y_j  >= y_o                (s output constraints)
          sum_j lambda_j = 1                          (BCC / VRS only)
          lambda_j >= 0, theta free

Super-efficiency (BCC, excluding the unit itself) is provided for ranking
efficient units.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger("voxfrontier.dea")

_MODEL_LABELS = {"CCR": "CRS", "BCC": "VRS"}


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def prepare_positive(data: np.ndarray) -> np.ndarray:
    """DEA requires strictly positive inputs/outputs.

    NaN/inf and non-positive values are replaced by half of the smallest
    positive value of the respective column (or 1e-3 if the column has none).
    """
    out = np.asarray(data, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    for j in range(out.shape[1]):
        col = out[:, j]
        pos = col[np.isfinite(col) & (col > 0)]
        repl = pos.min() / 2 if pos.size else 1e-3
        col[~np.isfinite(col) | (col <= 0)] = repl
        out[:, j] = col
    return out


def _solve_lp(c, A_ub, b_ub, A_eq, b_eq):
    """Solve one LP; return theta or NaN on failure."""
    from scipy.optimize import linprog

    res = linprog(
        c,
        A_ub=A_ub if A_ub is not None and len(A_ub) else None,
        b_ub=b_ub if b_ub is not None and len(b_ub) else None,
        A_eq=A_eq if A_eq is not None and len(A_eq) else None,
        b_eq=b_eq if b_eq is not None and len(b_eq) else None,
        bounds=[(0, None)] * len(c),
        method="highs",
    )
    return float(res.x[0]) if res.success else np.nan


def _build_constraints(X, Y, i, vrs: bool):
    n, m = X.shape
    s = Y.shape[1]
    # variables: [theta, lambda_1..lambda_n]
    c = np.zeros(n + 1)
    c[0] = 1.0

    A_ub, b_ub = [], []
    # inputs:  sum lam_j x_jk - theta * x_ik <= 0
    for k in range(m):
        row = np.zeros(n + 1)
        row[1:] = X[:, k]
        row[0] = -X[i, k]
        A_ub.append(row)
        b_ub.append(0.0)
    # outputs: -sum lam_j y_jr <= -y_ir   (i.e. sum >= y_i)
    for r in range(s):
        row = np.zeros(n + 1)
        row[1:] = -Y[:, r]
        A_ub.append(row)
        b_ub.append(-Y[i, r])

    A_eq, b_eq = None, None
    if vrs:
        A_eq = np.zeros((1, n + 1))
        A_eq[0, 1:] = 1.0
        b_eq = np.array([1.0])
    return c, np.array(A_ub), np.array(b_ub), A_eq, b_eq


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def dea_efficiency(
    X: np.ndarray,
    Y: np.ndarray,
    model_type: str = "BCC",
    backend: str = "auto",
) -> np.ndarray:
    """Input-oriented DEA efficiency for every DMU.

    Parameters
    ----------
    X, Y:
        Input matrix (n x m) and output matrix (n x s). Non-finite and
        non-positive values are sanitised automatically.
    model_type:
        ``"CCR"`` (constant returns to scale) or ``"BCC"`` (variable).
    backend:
        ``"scipy"`` | ``"pulp"`` | ``"auto"`` (scipy first, pulp fallback).
    """
    if model_type not in _MODEL_LABELS:
        raise ValueError(f"model_type must be one of {list(_MODEL_LABELS)}")

    X = prepare_positive(X)
    Y = prepare_positive(Y)
    n = X.shape[0]
    vrs = model_type == "BCC"

    chosen = "scipy" if backend in ("auto", "scipy") else backend
    eff = np.full(n, np.nan)

    if chosen in ("scipy", "auto"):
        try:
            for i in range(n):
                c, A_ub, b_ub, A_eq, b_eq = _build_constraints(X, Y, i, vrs)
                eff[i] = _solve_lp(c, A_ub, b_ub, A_eq, b_eq)
            if not np.isnan(eff).any():
                return np.round(eff, 6)
            logger.warning("scipy backend produced NaN efficiencies; "
                           "falling back to pulp if available.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("scipy DEA backend failed (%s); trying pulp.", exc)

    if backend in ("pulp", "auto"):
        try:
            return _dea_pulp(X, Y, vrs)
        except ImportError as exc:
            raise RuntimeError(
                "DEA could not be solved: install PuLP via `pip install pulp` "
                "or check your scipy version (>=1.9 required for HiGHS)."
            ) from exc
    return np.round(eff, 6)


def _cbc_solver(msg: int = 0):
    """Return a working CBC solver handle across PuLP 2.7-3.x APIs.

    PuLP>=3.2 deprecates ``PULP_CBC_CMD`` (bundled CBC) in favour of
    ``COIN_CMD`` + ``pulp[cbc]``. Older versions only ship the former.
    Strategy: prefer the new API *if its binary actually exists*; then fall
    back to the bundled-CBC API; otherwise raise a clear install hint.
    """
    import pulp  # optional dependency

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if hasattr(pulp, "COIN_CMD"):
            try:
                solver = pulp.COIN_CMD(msg=msg)
                if solver.executable(solver.path):
                    return solver
            except Exception:  # pragma: no cover - probing failure
                pass
        bundled = pulp.PULP_CBC_CMD(msg=msg)
        if bundled.executable(bundled.path):
            return bundled
    raise RuntimeError(
        "No CBC solver available. Install with `pip install pulp[cbc]` "
        "(PuLP>=3.2) or use the default scipy DEA backend."
    )


def _dea_pulp(X: np.ndarray, Y: np.ndarray, vrs: bool) -> np.ndarray:
    import pulp  # optional dependency

    n, m = X.shape
    s = Y.shape[1]
    eff = np.zeros(n)
    for i in range(n):
        prob = pulp.LpProblem(f"dea_{i}", pulp.LpMinimize)
        # NOTE: LpVariable construction is deprecation-warned in PuLP>=3.x
        # migration shims but remains the only API supported across PuLP
        # 2.7-3.x; filter the shim warning instead of breaking old versions.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            theta = pulp.LpVariable("theta", lowBound=0)
            lam = [pulp.LpVariable(f"lam_{j}", lowBound=0) for j in range(n)]
        prob += theta
        for k in range(m):
            prob += pulp.lpSum(lam[j] * X[j, k] for j in range(n)) <= theta * X[i, k]
        for r in range(s):
            prob += pulp.lpSum(lam[j] * Y[j, r] for j in range(n)) >= Y[i, r]
        if vrs:
            prob += pulp.lpSum(lam) == 1
        prob.solve(_cbc_solver())
        eff[i] = pulp.value(theta) if prob.status == 1 else np.nan
    return np.round(eff, 6)


def super_efficiency_bcc(X: np.ndarray, Y: np.ndarray, backend: str = "auto") -> np.ndarray:
    """BCC super-efficiency: re-solve each DMU excluding itself from peers.

    For unit ``i`` the evaluated point (``theta * x_i``, ``y_i``) stays in the
    constraints while the peer reference set contains every *other* unit.
    """
    X = prepare_positive(X)
    Y = prepare_positive(Y)
    n = X.shape[0]
    sup = np.full(n, np.nan)
    if backend in ("auto", "scipy"):
        try:
            for i in range(n):
                keep = np.ones(n, dtype=bool)
                keep[i] = False
                X_peers = X[keep]
                Y_peers = Y[keep]
                np_peers = X_peers.shape[0]

                # variables: [theta, lambda_1..lambda_{n-1}]
                c = np.zeros(np_peers + 1)
                c[0] = 1.0
                A_ub, b_ub = [], []
                for k in range(X.shape[1]):
                    row = np.zeros(np_peers + 1)
                    row[1:] = X_peers[:, k]
                    row[0] = -X[i, k]
                    A_ub.append(row)
                    b_ub.append(0.0)
                for r in range(Y.shape[1]):
                    row = np.zeros(np_peers + 1)
                    row[1:] = -Y_peers[:, r]
                    A_ub.append(row)
                    b_ub.append(-Y[i, r])
                A_eq = np.zeros((1, np_peers + 1))
                A_eq[0, 1:] = 1.0
                b_eq = np.array([1.0])
                sup[i] = _solve_lp(
                    c, np.array(A_ub), np.array(b_ub), A_eq, b_eq
                )
            return np.round(sup, 6)
        except Exception as exc:  # pragma: no cover
            logger.warning("scipy super-efficiency failed (%s); trying pulp.", exc)
    return _super_pulp(X, Y)


def _super_pulp(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    import pulp  # optional

    n, m = X.shape
    s = Y.shape[1]
    sup = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        prob = pulp.LpProblem(f"super_{i}", pulp.LpMinimize)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            theta = pulp.LpVariable("theta", lowBound=0)
            lam = [pulp.LpVariable(f"lam_{t}", lowBound=0) for t in others]
        prob += theta
        for k in range(m):
            prob += (
                pulp.lpSum(lam[t] * X[others[t], k] for t in range(len(others)))
                <= theta * X[i, k]
            )
        for r in range(s):
            prob += (
                pulp.lpSum(lam[t] * Y[others[t], r] for t in range(len(others)))
                >= Y[i, r]
            )
        prob += pulp.lpSum(lam) == 1
        prob.solve(_cbc_solver())
        sup[i] = pulp.value(theta) if prob.status == 1 else np.nan
    return np.round(sup, 6)


def run_three_models(
    df, specs: dict, outputs: tuple[str, ...], backend: str = "auto"
):
    """Run DEA for several input specifications.

    Parameters
    ----------
    df:
        Session-level DataFrame.
    specs:
        Mapping model name -> list of input column names
        (e.g. ``{"A": [...], "B": [...], "C": [...]}``).
    outputs:
        Output column names.
    backend:
        Solver backend, see :func:`dea_efficiency`.

    Returns
    -------
    dict
        ``{model: {"bcc": array, "ccr": array, "scale": array}}``
    """
    Y = prepare_positive(df[list(outputs)].to_numpy(float))
    results = {}
    for name, inputs in specs.items():
        X = prepare_positive(df[list(inputs)].to_numpy(float))
        ccr = dea_efficiency(X, Y, "CCR", backend)
        bcc = dea_efficiency(X, Y, "BCC", backend)
        scale = np.where(bcc > 1e-6, ccr / bcc, 0.0)
        results[name] = {"bcc": bcc, "ccr": ccr, "scale": scale}
        logger.info(
            "DEA model %s: mean BCC=%.4f, efficient DMUs=%d",
            name, np.nanmean(bcc), int((bcc >= 0.999).sum()),
        )
    return results
