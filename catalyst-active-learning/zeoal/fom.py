"""Figures of merit (FOMs) for CO2-methanation light-off traces.

The baseline FOM requested for this campaign is the **total CH4-temperature
integral** ``∫ c_ch4 dT`` — the trapezoidal integral of the outlet CH4
concentration over the (optionally windowed) temperature ramp, in units of
concentration × kelvin (DESIGN.md, section ``zeoal/fom.py``).  A menu of
alternatives rewards low-temperature activity (``weighted_ch4_integral``),
selectivity against CO (``selectivity_weighted_integral``,
``co_penalized_integral``) or classic light-off descriptors (``t50`` and
friends; see LITERATURE.md on light-off/ignition temperatures).

Registry contract
-----------------
Every entry of :data:`FOM_REGISTRY` is a callable

    ``f(T_K, c_ch4, c_co2, c_co, **kw) -> float``

with temperatures in kelvin and the three concentration channels in the
trace's (arbitrary but shared) concentration unit.  Inputs are sorted by
temperature internally if needed.  :data:`FOM_INFO` carries per-FOM metadata:
at least ``maximize`` (bool), ``description`` (str) and ``units`` (str), plus
``analytic_sd`` (bool — whether an analytic error propagation is available).

Uncertainty model
-----------------
Trace noise is assumed **iid Gaussian per temperature point** with 1-sigma
``noise_sd`` (scalar or per-point), applying independently to the CH4 and CO
channels (DESIGN.md error-propagation summary).  For the four integral FOMs
the trapezoid rule makes the FOM (to first order) linear in the measured
concentrations, so the sd follows analytically from the effective quadrature
weights:

    ``sd^2 = sum_i (a_i * sd_i)^2 + sum_i (b_i * sd_i)^2``

where ``a_i = dFOM/dc_ch4_i`` and ``b_i = dFOM/dc_co_i`` include the trapezoid
weights, any temperature weight function, the CO penalty and — for the
windowed case — the linear-interpolation stencil at the window edges.  For
``selectivity_weighted_integral`` (quadratic in the concentrations through the
selectivity factor) the coefficients are a first-order delta-method
linearization at the measured trace.  Nonlinear FOMs (``t50``/``t10``/``t90``,
``max_ch4``, ``peak_temperature``, ``ch4_at_T``) fall back to Monte Carlo
perturbation of the trace, deterministic given ``rng``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .data import TemperatureTrace, as_generator

__all__ = [
    "FOMResult",
    "FOM_REGISTRY",
    "FOM_INFO",
    "fom_from_trace",
    "fom_from_fit",
    "ch4_temperature_integral",
    "weighted_ch4_integral",
    "selectivity_weighted_integral",
    "co_penalized_integral",
    "t10",
    "t50",
    "t90",
    "max_ch4",
    "peak_temperature",
    "ch4_at_T",
]


@dataclass
class FOMResult:
    """A figure-of-merit value with (optional) 1-sigma uncertainty.

    Attributes
    ----------
    value:
        Point estimate of the FOM.
    sd:
        1-sigma standard deviation, or ``None`` when no noise information was
        supplied.
    name:
        Registry key of the FOM (e.g. ``"ch4_temperature_integral"``).
    details:
        Diagnostics: FOM keyword arguments, sd method (``"analytic"``, ``"mc"``
        or ``"mc_kinetic_params"``), MC sample statistics, notes (e.g. a
        threshold FOM that never crossed), units, ...
    """

    value: float
    sd: float | None
    name: str
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# numerical helpers
# --------------------------------------------------------------------------

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral, ``np.trapezoid`` with ``np.trapz`` compat."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # numpy < 2
        integrate = np.trapz
    return float(integrate(y, x))


def _trapezoid_weights(T: np.ndarray) -> np.ndarray:
    """Quadrature weights ``w`` with ``sum(w * y) == trapezoid(y, T)`` exactly.

    ``w_0 = dT_0/2``, ``w_i = (dT_{i-1} + dT_i)/2``, ``w_{n-1} = dT_{n-2}/2``.
    """
    T = np.asarray(T, dtype=float)
    if T.size < 2:
        raise ValueError("trapezoid integration needs at least 2 points")
    w = np.zeros_like(T)
    dT = np.diff(T)
    w[:-1] += 0.5 * dT
    w[1:] += 0.5 * dT
    return w


def _prep(T_K: np.ndarray, *channels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Validate and sort FOM inputs by temperature.

    Returns ``(T, channel_1, ...)`` as float arrays sorted ascending in ``T``.
    """
    T = np.asarray(T_K, dtype=float).ravel()
    if T.size < 2:
        raise ValueError("FOM evaluation needs at least 2 temperature points")
    if np.any(~np.isfinite(T)):
        raise ValueError("non-finite temperatures passed to FOM")
    outs = []
    for k, c in enumerate(channels):
        arr = np.asarray(c, dtype=float).ravel()
        if arr.size != T.size:
            raise ValueError(
                f"concentration channel {k} has {arr.size} points, "
                f"temperature grid has {T.size}"
            )
        outs.append(arr)
    if np.any(np.diff(T) < 0):
        order = np.argsort(T, kind="stable")
        T = T[order]
        outs = [c[order] for c in outs]
    return (T, *outs)


def _apply_window(
    T: np.ndarray, T_min: float | None, T_max: float | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Restrict a sorted grid ``T`` to the window ``[T_min, T_max]``.

    The window is applied by masking the interior points and adding
    linearly-interpolated points at the window edges when those fall between
    grid points.  Returns ``(T_w, M)`` where ``M`` is the ``(n_w, n)``
    stencil matrix such that any channel restricted to the window is
    ``c_w = M @ c`` (``M is None`` means identity, i.e. no window).

    The window is clipped to the data range (documented clipping); a window
    that does not overlap the data raises ``ValueError``.
    """
    if T_min is None and T_max is None:
        return T, None
    n = T.size
    lo = float(T[0]) if T_min is None else max(float(T_min), float(T[0]))
    hi = float(T[-1]) if T_max is None else min(float(T_max), float(T[-1]))
    if not hi > lo:
        raise ValueError(
            f"integration window [T_min={T_min}, T_max={T_max}] K does not "
            f"overlap the trace range [{T[0]:.6g}, {T[-1]:.6g}] K"
        )

    def interp_row(t: float) -> np.ndarray:
        row = np.zeros(n)
        j = int(np.searchsorted(T, t))  # T[j-1] < t <= T[j]; here t < T[j]
        left, right = j - 1, j
        dt = T[right] - T[left]
        if dt <= 0:  # duplicate grid temperatures
            row[left] = 1.0
        else:
            frac = (t - T[left]) / dt
            row[left] = 1.0 - frac
            row[right] = frac
        return row

    idx = np.where((T >= lo) & (T <= hi))[0]
    T_list: list[float] = []
    rows: list[np.ndarray] = []
    if idx.size == 0 or T[idx[0]] > lo:
        T_list.append(lo)
        rows.append(interp_row(lo))
    for i in idx:
        r = np.zeros(n)
        r[i] = 1.0
        T_list.append(float(T[i]))
        rows.append(r)
    if idx.size == 0 or T[idx[-1]] < hi:
        T_list.append(hi)
        rows.append(interp_row(hi))
    return np.asarray(T_list), np.vstack(rows)


def _weight_values(
    weight: str, T: np.ndarray, T0: float | None, tau_w: float
) -> np.ndarray:
    """Temperature weight function for :func:`weighted_ch4_integral`."""
    if weight == "one_over_T":
        return 1.0 / T
    if weight == "low_T_boltzmann":
        if tau_w <= 0:
            raise ValueError(f"tau_w must be positive, got {tau_w}")
        T0v = float(T.min()) if T0 is None else float(T0)
        return np.exp(-(T - T0v) / float(tau_w))
    raise ValueError(
        f"unknown weight {weight!r}; use 'one_over_T' or 'low_T_boltzmann'"
    )


def _threshold_temperature(
    T: np.ndarray, c: np.ndarray, fraction: float
) -> tuple[float, bool]:
    """First temperature where ``c`` crosses ``fraction * max(c)``.

    Linear interpolation between the bracketing grid points.  Returns
    ``(temperature, crossed)``; when the threshold is never reached (all
    concentrations <= 0, or ``fraction > 1``) it returns ``(T.max(), False)``.
    """
    cmax = float(np.max(c))
    if cmax <= 0.0:
        return float(T[-1]), False
    thr = fraction * cmax
    above = c >= thr
    idx = int(np.argmax(above))
    if not above[idx]:  # never reached (only possible for fraction > 1)
        return float(T[-1]), False
    if idx == 0:
        return float(T[0]), True
    c0, c1 = float(c[idx - 1]), float(c[idx])
    # c0 < thr <= c1 by construction, so c1 - c0 > 0
    t = T[idx - 1] + (thr - c0) / (c1 - c0) * (T[idx] - T[idx - 1])
    return float(t), True


def _integral_analysis(
    name: str,
    T_K: np.ndarray,
    c_ch4: np.ndarray,
    c_co: np.ndarray,
    T_min: float | None = None,
    T_max: float | None = None,
    weight: str = "one_over_T",
    T0: float | None = None,
    tau_w: float = 100.0,
    lambda_co: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Value and first-order sensitivity of an integral FOM.

    Returns ``(value, a, b)`` where ``a_i = dFOM/dc_ch4_i`` and
    ``b_i = dFOM/dc_co_i`` on the *original* grid (window/interpolation
    stencil folded in), so that under iid per-point Gaussian noise
    ``sd^2 = sum((a*sd_i)**2) + sum((b*sd_i)**2)`` (DESIGN.md, fom section).
    For ``selectivity_weighted_integral`` the coefficients are the
    delta-method linearization at the measured trace.
    """
    T, ch4, co = _prep(T_K, c_ch4, c_co)
    T_w, M = _apply_window(T, T_min, T_max)
    ch4_w = ch4 if M is None else M @ ch4
    co_w = co if M is None else M @ co
    w = _trapezoid_weights(T_w)

    if name == "ch4_temperature_integral":
        integrand = ch4_w
        g_ch4, g_co = w.copy(), np.zeros_like(w)
    elif name == "weighted_ch4_integral":
        wt = _weight_values(weight, T_w, T0, tau_w)
        integrand = wt * ch4_w
        g_ch4, g_co = w * wt, np.zeros_like(w)
    elif name == "selectivity_weighted_integral":
        den = ch4_w + co_w
        safe = den > 0
        den_safe = np.where(safe, den, 1.0)
        S = np.where(safe, ch4_w / den_safe, 0.0)  # 0/0 -> 0 by convention
        integrand = ch4_w * S
        # d(c^2/(c+o))/dc = c(c+2o)/(c+o)^2 ; d(c^2/(c+o))/do = -c^2/(c+o)^2
        g_ch4 = w * np.where(safe, ch4_w * (ch4_w + 2.0 * co_w) / den_safe**2, 0.0)
        g_co = w * np.where(safe, -(ch4_w**2) / den_safe**2, 0.0)
    elif name == "co_penalized_integral":
        lam = float(lambda_co)
        integrand = ch4_w - lam * co_w
        g_ch4, g_co = w.copy(), -lam * w
    else:  # pragma: no cover - internal misuse
        raise ValueError(f"{name!r} is not an integral FOM")

    value = _trapezoid(integrand, T_w)
    a = g_ch4 if M is None else M.T @ g_ch4
    b = g_co if M is None else M.T @ g_co
    return value, a, b


# --------------------------------------------------------------------------
# registry FOM functions — f(T_K, c_ch4, c_co2, c_co, **kw) -> float
# --------------------------------------------------------------------------

def ch4_temperature_integral(
    T_K: np.ndarray,
    c_ch4: np.ndarray,
    c_co2: np.ndarray,
    c_co: np.ndarray,
    T_min: float | None = None,
    T_max: float | None = None,
) -> float:
    """Baseline FOM: total CH4-temperature integral ``∫ c_ch4 dT``.

    Trapezoid rule over the optional ``[T_min, T_max]`` window (DESIGN.md,
    fom section); windowing masks interior points and adds
    linearly-interpolated points at the window edges.  Units: conc * K.
    """
    value, _, _ = _integral_analysis(
        "ch4_temperature_integral", T_K, c_ch4, c_co, T_min=T_min, T_max=T_max
    )
    return value


def weighted_ch4_integral(
    T_K: np.ndarray,
    c_ch4: np.ndarray,
    c_co2: np.ndarray,
    c_co: np.ndarray,
    weight: str = "one_over_T",
    T0: float | None = None,
    tau_w: float = 100.0,
    T_min: float | None = None,
    T_max: float | None = None,
) -> float:
    """Temperature-weighted CH4 integral ``∫ c_ch4 * w(T) dT``.

    Weights (DESIGN.md, fom section) reward low-temperature activity:

    * ``"one_over_T"``: ``w = 1/T``;
    * ``"low_T_boltzmann"``: ``w = exp(-(T - T0)/tau_w)`` with ``T0``
      defaulting to the minimum of the (windowed) grid and ``tau_w = 100 K``.
    """
    value, _, _ = _integral_analysis(
        "weighted_ch4_integral",
        T_K,
        c_ch4,
        c_co,
        T_min=T_min,
        T_max=T_max,
        weight=weight,
        T0=T0,
        tau_w=tau_w,
    )
    return value


def selectivity_weighted_integral(
    T_K: np.ndarray,
    c_ch4: np.ndarray,
    c_co2: np.ndarray,
    c_co: np.ndarray,
    T_min: float | None = None,
    T_max: float | None = None,
) -> float:
    """Selectivity-weighted integral ``∫ c_ch4 * S(T) dT``.

    ``S = c_ch4 / (c_ch4 + c_co)`` is the CH4 selectivity among carbon
    products; ``0/0`` (and non-positive denominators) map to ``S = 0``
    (DESIGN.md, fom section).
    """
    value, _, _ = _integral_analysis(
        "selectivity_weighted_integral", T_K, c_ch4, c_co, T_min=T_min, T_max=T_max
    )
    return value


def co_penalized_integral(
    T_K: np.ndarray,
    c_ch4: np.ndarray,
    c_co2: np.ndarray,
    c_co: np.ndarray,
    lambda_co: float = 1.0,
    T_min: float | None = None,
    T_max: float | None = None,
) -> float:
    """CO-penalized integral ``∫ (c_ch4 - lambda_co * c_co) dT``.

    ``lambda_co`` (default 1.0) sets the linear penalty on co-produced CO
    (DESIGN.md, fom section).
    """
    value, _, _ = _integral_analysis(
        "co_penalized_integral",
        T_K,
        c_ch4,
        c_co,
        T_min=T_min,
        T_max=T_max,
        lambda_co=lambda_co,
    )
    return value


def t50(T_K: np.ndarray, c_ch4: np.ndarray, c_co2: np.ndarray, c_co: np.ndarray) -> float:
    """Light-off temperature T50: first T where CH4 reaches 50% of its max.

    Linearly interpolated between grid points; lower is better
    (``maximize=False`` in :data:`FOM_INFO`).  When CH4 never rises above
    zero the trace maximum temperature is returned (see
    :func:`fom_from_trace` details for the non-crossing note).
    """
    T, ch4 = _prep(T_K, c_ch4)
    return _threshold_temperature(T, ch4, 0.50)[0]


def t10(T_K: np.ndarray, c_ch4: np.ndarray, c_co2: np.ndarray, c_co: np.ndarray) -> float:
    """Ignition temperature T10: first T where CH4 reaches 10% of its max."""
    T, ch4 = _prep(T_K, c_ch4)
    return _threshold_temperature(T, ch4, 0.10)[0]


def t90(T_K: np.ndarray, c_ch4: np.ndarray, c_co2: np.ndarray, c_co: np.ndarray) -> float:
    """T90: first T where CH4 reaches 90% of its max."""
    T, ch4 = _prep(T_K, c_ch4)
    return _threshold_temperature(T, ch4, 0.90)[0]


def max_ch4(T_K: np.ndarray, c_ch4: np.ndarray, c_co2: np.ndarray, c_co: np.ndarray) -> float:
    """Maximum outlet CH4 concentration over the ramp."""
    _, ch4 = _prep(T_K, c_ch4)
    return float(np.max(ch4))


def peak_temperature(
    T_K: np.ndarray, c_ch4: np.ndarray, c_co2: np.ndarray, c_co: np.ndarray
) -> float:
    """Temperature of the CH4 concentration maximum (first grid point at max)."""
    T, ch4 = _prep(T_K, c_ch4)
    return float(T[int(np.argmax(ch4))])


def ch4_at_T(
    T_K: np.ndarray,
    c_ch4: np.ndarray,
    c_co2: np.ndarray,
    c_co: np.ndarray,
    T_eval: float | None = None,
) -> float:
    """CH4 concentration at ``T_eval`` [K], linearly interpolated.

    ``T_eval`` outside the trace range clamps to the nearest endpoint value
    (``np.interp`` behavior, documented — no extrapolation).
    """
    if T_eval is None:
        raise ValueError("ch4_at_T requires the keyword argument T_eval (kelvin)")
    T, ch4 = _prep(T_K, c_ch4)
    return float(np.interp(float(T_eval), T, ch4))


FOM_REGISTRY: dict[str, Callable[..., float]] = {
    "ch4_temperature_integral": ch4_temperature_integral,
    "weighted_ch4_integral": weighted_ch4_integral,
    "selectivity_weighted_integral": selectivity_weighted_integral,
    "co_penalized_integral": co_penalized_integral,
    "t50": t50,
    "t10": t10,
    "t90": t90,
    "max_ch4": max_ch4,
    "peak_temperature": peak_temperature,
    "ch4_at_T": ch4_at_T,
}

#: Names of integral FOMs with analytic (first-order) error propagation.
_ANALYTIC_SD_FOMS = frozenset(
    {
        "ch4_temperature_integral",
        "weighted_ch4_integral",
        "selectivity_weighted_integral",
        "co_penalized_integral",
    }
)

_THRESHOLD_FRACTIONS = {"t50": 0.50, "t10": 0.10, "t90": 0.90}

FOM_INFO: dict[str, dict[str, Any]] = {
    "ch4_temperature_integral": {
        "maximize": True,
        "description": (
            "Baseline FOM: trapezoidal integral of outlet CH4 concentration "
            "over temperature, optional [T_min, T_max] window."
        ),
        "units": "conc*K",
        "analytic_sd": True,
    },
    "weighted_ch4_integral": {
        "maximize": True,
        "description": (
            "CH4 integral weighted by 1/T ('one_over_T') or a low-temperature "
            "Boltzmann factor exp(-(T-T0)/tau_w) ('low_T_boltzmann'); rewards "
            "low-temperature activity."
        ),
        "units": "conc*K (weight-dependent; conc for 'one_over_T')",
        "analytic_sd": True,
    },
    "selectivity_weighted_integral": {
        "maximize": True,
        "description": (
            "Integral of c_ch4 * S dT with CH4 selectivity "
            "S = c_ch4/(c_ch4 + c_co) (0/0 -> 0); penalizes CO co-production."
        ),
        "units": "conc*K",
        "analytic_sd": True,
    },
    "co_penalized_integral": {
        "maximize": True,
        "description": "Integral of (c_ch4 - lambda_co * c_co) dT, default lambda_co=1.",
        "units": "conc*K",
        "analytic_sd": True,
    },
    "t50": {
        "maximize": False,
        "description": (
            "Light-off temperature: first T where CH4 reaches 50% of its "
            "maximum (interpolated); lower is better."
        ),
        "units": "K",
        "analytic_sd": False,
    },
    "t10": {
        "maximize": False,
        "description": "First T where CH4 reaches 10% of its maximum (ignition onset).",
        "units": "K",
        "analytic_sd": False,
    },
    "t90": {
        "maximize": False,
        "description": "First T where CH4 reaches 90% of its maximum.",
        "units": "K",
        "analytic_sd": False,
    },
    "max_ch4": {
        "maximize": True,
        "description": "Maximum outlet CH4 concentration over the ramp.",
        "units": "conc",
        "analytic_sd": False,
    },
    "peak_temperature": {
        "maximize": False,
        "description": (
            "Temperature of the CH4 maximum; lower peaks indicate "
            "low-temperature activity."
        ),
        "units": "K",
        "analytic_sd": False,
    },
    "ch4_at_T": {
        "maximize": True,
        "description": "CH4 concentration interpolated at T_eval [K] (required kw).",
        "units": "conc",
        "analytic_sd": False,
    },
}


def _lookup(name: str) -> Callable[..., float]:
    try:
        return FOM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown FOM {name!r}; available: {sorted(FOM_REGISTRY)}"
        ) from None


# --------------------------------------------------------------------------
# high-level entry points
# --------------------------------------------------------------------------

def fom_from_trace(
    trace: TemperatureTrace,
    name: str = "ch4_temperature_integral",
    noise_sd: float | np.ndarray | None = None,
    rng: np.random.Generator | int | None = None,
    n_samples: int = 500,
    sd_method: str = "auto",
    **kw: Any,
) -> FOMResult:
    """Evaluate a figure of merit on a measured temperature trace.

    Parameters
    ----------
    trace:
        :class:`zeoal.data.TemperatureTrace` (or any object with sorted
        ``temperature_K``, ``c_co2``, ``c_ch4``, ``c_co`` arrays).
    name:
        Key of :data:`FOM_REGISTRY` (default: the baseline
        ``"ch4_temperature_integral"``).
    noise_sd:
        Optional 1-sigma measurement noise per concentration point (scalar or
        per-point array aligned with the trace).  Assumed iid Gaussian per
        point, independently on the CH4 and CO channels.  ``None`` -> no
        uncertainty (``sd=None``).
    rng:
        Random source for the Monte Carlo sd path (Generator, int seed or
        None; normalized via :func:`zeoal.data.as_generator`).  The result is
        deterministic given ``rng``.
    n_samples:
        Number of Monte Carlo perturbations for nonlinear FOMs (>= 2).
    sd_method:
        ``"auto"`` (default): analytic error propagation for the four integral
        FOMs, Monte Carlo otherwise; ``"analytic"`` or ``"mc"`` to force one
        (``"analytic"`` raises for nonlinear FOMs).
    **kw:
        Forwarded to the FOM callable (e.g. ``T_min``/``T_max`` window,
        ``lambda_co``, ``weight``, ``T_eval``).

    Returns
    -------
    FOMResult
        ``value`` from the unperturbed trace; ``sd`` per the uncertainty model
        in the module docstring (``sd^2 = sum_i (w_i * sd_i)^2`` with
        effective quadrature weights for linear integrals, MC otherwise);
        ``details`` records the sd method, kwargs and any notes.
    """
    f = _lookup(name)
    T = np.asarray(trace.temperature_K, dtype=float).ravel()
    ch4 = np.asarray(trace.c_ch4, dtype=float).ravel()
    co2 = np.asarray(trace.c_co2, dtype=float).ravel()
    co = np.asarray(trace.c_co, dtype=float).ravel()
    sigma = None
    if noise_sd is not None:
        sigma = np.broadcast_to(np.asarray(noise_sd, dtype=float), T.shape).astype(float)
        if np.any(sigma < 0) or np.any(~np.isfinite(sigma)):
            raise ValueError("noise_sd must be finite and non-negative")
    if np.any(np.diff(T) < 0):  # duck-typed traces may be unsorted
        order = np.argsort(T, kind="stable")
        T, ch4, co2, co = T[order], ch4[order], co2[order], co[order]
        if sigma is not None:
            sigma = sigma[order]

    value = float(f(T, ch4, co2, co, **kw))
    details: dict[str, Any] = {
        "fom_kwargs": dict(kw),
        "n_points": int(T.size),
        "units": FOM_INFO[name]["units"],
        "concentration_unit": getattr(trace, "concentration_unit", None),
    }
    if "T_min" in kw or "T_max" in kw:
        details["window"] = (kw.get("T_min"), kw.get("T_max"))
    if name in _THRESHOLD_FRACTIONS:
        Ts, cs = _prep(T, ch4)
        _, crossed = _threshold_temperature(Ts, cs, _THRESHOLD_FRACTIONS[name])
        details["crossed"] = bool(crossed)
        if not crossed:
            details["note"] = (
                f"CH4 never crosses {_THRESHOLD_FRACTIONS[name]:.0%} of its "
                f"maximum (max(c_ch4) <= 0); returning T.max() = {value:g} K"
            )

    if sigma is None:
        return FOMResult(value=value, sd=None, name=name, details=details)

    if sd_method == "auto":
        sd_method = "analytic" if name in _ANALYTIC_SD_FOMS else "mc"
    if sd_method == "analytic":
        if name not in _ANALYTIC_SD_FOMS:
            raise ValueError(
                f"analytic sd is only available for {sorted(_ANALYTIC_SD_FOMS)}; "
                f"use sd_method='mc' for {name!r}"
            )
        _, a, b = _integral_analysis(name, T, ch4, co, **kw)
        sd = float(np.sqrt(np.sum((a * sigma) ** 2) + np.sum((b * sigma) ** 2)))
        details["sd_method"] = "analytic"
    elif sd_method == "mc":
        n_samples = int(n_samples)
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 for Monte Carlo sd")
        gen = as_generator(rng)
        vals = np.empty(n_samples)
        for i in range(n_samples):
            ch4_p = ch4 + gen.standard_normal(T.size) * sigma
            co_p = co + gen.standard_normal(T.size) * sigma
            vals[i] = f(T, ch4_p, co2, co_p, **kw)
        sd = float(np.std(vals, ddof=1))
        details.update(
            sd_method="mc",
            n_samples=n_samples,
            samples_mean=float(np.mean(vals)),
            samples_sd=sd,
        )
    else:
        raise ValueError(
            f"sd_method must be 'auto', 'analytic' or 'mc', got {sd_method!r}"
        )
    return FOMResult(value=value, sd=sd, name=name, details=details)


def fom_from_fit(
    fit_result: Any,
    model_name: str,
    T_grid: np.ndarray,
    feed: Any,
    name: str = "ch4_temperature_integral",
    n_samples: int = 500,
    rng: np.random.Generator | int | None = None,
    **kw: Any,
) -> FOMResult:
    """Figure of merit (with uncertainty) from a kinetic fit, by Monte Carlo.

    Samples kinetic parameters from ``fit_result.sample_params`` (multivariate
    normal from the fit covariance, see ``zeoal.kinetics``), predicts the
    concentration trace on ``T_grid`` for each draw via
    ``zeoal.kinetics.predict_trace(model_name, T_grid, params, feed)`` and
    evaluates the FOM on each predicted trace; ``value``/``sd`` are the mean
    and 1-sigma sd over the MC distribution (DESIGN.md, fom section and
    error-propagation summary).

    Parameters
    ----------
    fit_result:
        A ``zeoal.kinetics.KineticFitResult`` (anything exposing
        ``sample_params(n, rng) -> (n, p)`` and optionally ``params``).
    model_name:
        Kinetic model registry key (e.g. ``"arrhenius"``).
    T_grid:
        Temperature grid [K] on which the trace is predicted and integrated.
    feed:
        ``zeoal.kinetics.FeedConditions`` passed through to the predictor.
    name, n_samples, rng, **kw:
        As in :func:`fom_from_trace`; MC draws whose FOM is non-finite (or
        whose prediction raises) are dropped and counted in
        ``details['n_failed']``.

    Notes
    -----
    ``zeoal.kinetics`` is imported lazily inside this function so that the
    fom module stays importable independently of the kinetics stack.
    """
    f = _lookup(name)
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for Monte Carlo FOM propagation")
    kinetics = importlib.import_module("zeoal.kinetics")  # lazy per DESIGN.md
    gen = as_generator(rng)
    T_grid = np.asarray(T_grid, dtype=float).ravel()

    param_samples = np.atleast_2d(
        np.asarray(fit_result.sample_params(n_samples, gen), dtype=float)
    )
    vals = np.full(param_samples.shape[0], np.nan)
    first_error: str | None = None
    for i, p in enumerate(param_samples):
        try:
            tr = kinetics.predict_trace(model_name, T_grid, p, feed)
            vals[i] = f(
                np.asarray(tr.temperature_K, dtype=float),
                np.asarray(tr.c_ch4, dtype=float),
                np.asarray(tr.c_co2, dtype=float),
                np.asarray(tr.c_co, dtype=float),
                **kw,
            )
        except Exception as exc:  # pathological parameter draws are dropped
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
    finite = np.isfinite(vals)
    n_used = int(finite.sum())
    if n_used < 2:
        raise ValueError(
            f"only {n_used} of {param_samples.shape[0]} Monte Carlo draws produced a "
            f"finite {name!r}"
            + (f" (first error: {first_error})" if first_error else "")
            + "; check the kinetic fit, feed and T_grid"
        )
    good = vals[finite]
    value = float(np.mean(good))
    sd = float(np.std(good, ddof=1))

    best_fit_value: float | None = None
    params_best = getattr(fit_result, "params", None)
    if params_best is not None:
        try:
            tr = kinetics.predict_trace(
                model_name, T_grid, np.asarray(params_best, dtype=float), feed
            )
            best_fit_value = float(
                f(
                    np.asarray(tr.temperature_K, dtype=float),
                    np.asarray(tr.c_ch4, dtype=float),
                    np.asarray(tr.c_co2, dtype=float),
                    np.asarray(tr.c_co, dtype=float),
                    **kw,
                )
            )
        except Exception:  # keep MC result even if the point prediction fails
            best_fit_value = None

    details: dict[str, Any] = {
        "model_name": str(model_name),
        "fom_kwargs": dict(kw),
        "sd_method": "mc_kinetic_params",
        "n_samples": n_samples,
        "n_used": n_used,
        "n_failed": int(param_samples.shape[0] - n_used),
        "samples_mean": value,
        "samples_sd": sd,
        "best_fit_value": best_fit_value,
        "units": FOM_INFO[name]["units"],
    }
    if first_error is not None:
        details["first_error"] = first_error
    return FOMResult(value=value, sd=sd, name=name, details=details)
