"""Thermodynamic equilibrium of CO2 methanation (Sabatier reaction).

    CO2 + 4 H2  <=>  CH4 + 2 H2O(g)      dH0(298 K) = -165.0 kJ/mol

The reaction is strongly exothermic with a mole-number decrease (dn = -2), so
equilibrium conversion falls with temperature and rises with pressure; at
1 bar it degrades noticeably above ~600-650 K, which produces the
characteristic high-temperature decline of CH4 in light-off traces (see
LITERATURE.md).  Above ~700-800 K the reverse water-gas shift additionally
diverts CO2 to CO.

Implementation: Gibbs energy from dH0/dS0 at 298.15 K with a constant-dCp
correction (dCp = -45 J mol-1 K-1, an effective average over 300-900 K fitted
to standard thermochemical tables).  This reproduces literature Kp values for
CO2 methanation to within a few tens of percent over 300-900 K — in
particular Kp ~ 1.5e3 bar^-2 at 673 K and the Kp = 1 crossover near 870 K —
which is far more accurate than needed for the equilibrium *cap* used in
kinetic fitting.

The equilibrium conversion solver considers methanation only (no simultaneous
RWGS equilibrium); this is a documented approximation adequate below ~800 K
at methanation feed ratios.  RWGS is treated kinetically in
:mod:`zeoal.kinetics` via a parallel CO channel.
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.314462618  # J mol-1 K-1
T_STD = 298.15  # K

# CO2 + 4 H2 <=> CH4 + 2 H2O(g); standard formation data (NIST-JANAF)
DH298_METHANATION = -165.0e3  # J mol-1
DS298_METHANATION = -172.6  # J mol-1 K-1
DCP_METHANATION = -45.0  # J mol-1 K-1, effective constant over 300-900 K

# CO2 + H2 <=> CO + H2O(g)  (reverse water-gas shift); dCp ~ 0
DH298_RWGS = 41.2e3  # J mol-1
DS298_RWGS = 42.0  # J mol-1 K-1

__all__ = [
    "R_GAS",
    "delta_g_methanation",
    "keq_methanation",
    "keq_rwgs",
    "equilibrium_conversion",
]


def _delta_g(T_K: np.ndarray, dh298: float, ds298: float, dcp: float) -> np.ndarray:
    T = np.asarray(T_K, dtype=float)
    dh = dh298 + dcp * (T - T_STD)
    ds = ds298 + dcp * np.log(T / T_STD)
    return dh - T * ds


def delta_g_methanation(T_K: np.ndarray) -> np.ndarray:
    """Standard Gibbs energy change [J/mol] of CO2 methanation at ``T_K``."""
    return _delta_g(T_K, DH298_METHANATION, DS298_METHANATION, DCP_METHANATION)


def keq_methanation(T_K: np.ndarray) -> np.ndarray:
    """Equilibrium constant of CO2 methanation (1 bar standard state).

    Dimensionless in activities; equals Kp in bar^-2 when partial pressures
    are expressed in bar (mole-number change dn = -2).
    """
    T = np.asarray(T_K, dtype=float)
    return np.exp(-delta_g_methanation(T) / (R_GAS * T))


def keq_rwgs(T_K: np.ndarray) -> np.ndarray:
    """Equilibrium constant of the reverse water-gas shift (dn = 0)."""
    T = np.asarray(T_K, dtype=float)
    return np.exp(-_delta_g(T, DH298_RWGS, DS298_RWGS, 0.0) / (R_GAS * T))


def _log_q(x: float, y_co2: float, y_h2: float, y_inert: float, p_bar: float) -> float:
    """ln of the reaction quotient at CO2 conversion ``x`` (methanation only).

    Basis 1 mol feed; extent per mole feed xi = x * y_co2; total moles 1 - 2 xi.
    """
    xi = x * y_co2
    n_tot = 1.0 - 2.0 * xi
    n_co2 = y_co2 - xi
    n_h2 = y_h2 - 4.0 * xi
    n_ch4 = xi
    n_h2o = 2.0 * xi
    tiny = 1e-300
    return (
        np.log(max(n_ch4, tiny))
        + 2.0 * np.log(max(n_h2o, tiny))
        - np.log(max(n_co2, tiny))
        - 4.0 * np.log(max(n_h2, tiny))
        + 2.0 * np.log(max(n_tot, tiny))  # from (n_i / n_tot)^nu_i with sum(nu) = -2
        - 2.0 * np.log(max(p_bar, tiny))
    )


def equilibrium_conversion(
    T_K: np.ndarray,
    y_co2: float = 0.2,
    y_h2: float = 0.8,
    y_inert: float = 0.0,
    pressure_bar: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """Equilibrium CO2 conversion for a CO2/H2/inert feed, vs temperature.

    Solves ``Q(X) = Keq(T)`` by bisection in ``X`` on ``(0, X_max)`` where
    ``X_max = min(1, y_h2 / (4 y_co2))`` (H2-limited ceiling).  ``ln Q`` is
    strictly increasing in ``X`` so the root is unique.

    Parameters
    ----------
    T_K:
        Scalar or array of temperatures [K].
    y_co2, y_h2, y_inert:
        Inlet mole fractions (normalized internally; must be positive for
        CO2 and H2).
    pressure_bar:
        Total pressure [bar].

    Returns
    -------
    Array (or scalar) of equilibrium conversion in [0, 1].
    """
    total = y_co2 + y_h2 + y_inert
    if y_co2 <= 0 or y_h2 <= 0 or total <= 0:
        raise ValueError("feed must contain CO2 and H2 with positive fractions")
    y_co2, y_h2, y_inert = y_co2 / total, y_h2 / total, y_inert / total

    T = np.atleast_1d(np.asarray(T_K, dtype=float))
    ln_keq = np.log(keq_methanation(T))
    x_max = min(1.0, y_h2 / (4.0 * y_co2))
    out = np.empty_like(T)
    for i, lk in enumerate(ln_keq):
        lo, hi = 0.0, x_max
        # ln Q -> -inf as X -> 0+ and +inf as X -> x_max-; bisection is safe.
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if _log_q(mid, y_co2, y_h2, y_inert, pressure_bar) < lk:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        out[i] = 0.5 * (lo + hi)
    return out[0] if np.isscalar(T_K) or np.ndim(T_K) == 0 else out
