"""Kinetic light-off models and bounded least-squares fitting.

Fits :class:`zeoal.data.TemperatureTrace` objects (outlet CO2/CH4/CO
concentrations versus temperature) to mechanistic or empirical models and
returns best-fit parameters *with covariance* (:class:`KineticFitResult`) so
parameter uncertainty can be propagated downstream (DESIGN.md
"Error-propagation summary").

Models (registry :data:`KINETIC_MODELS`)
-----------------------------------------
``"arrhenius"`` : :class:`ArrheniusFirstOrder`
    Two parallel pseudo-first-order channels from CO2 — methanation and
    reverse water-gas shift (RWGS) — in an integral PFR at constant space
    time (absorbed into the rate prefactors), capped by the methanation
    equilibrium conversion from :mod:`zeoal.equilibrium`.
``"eyring"`` : :class:`EyringFirstOrder`
    Same structure with transition-state-theory temperature dependence
    ("Eyring 1935" tag, LITERATURE.md).
``"lhhw"`` : :class:`LHHWKinetics`
    "Koschany 2016"-inspired Langmuir-Hinshelwood-Hougen-Watson rate law,
    integrated as an isothermal ideal PFR per temperature point.
``"sigmoid"`` : :class:`SigmoidLightOff`
    Empirical sigmoidal light-off; robust fallback that yields ``T50``
    directly.

Conventions
-----------
* Temperatures in kelvin, activation energies/enthalpies in J/mol
  (internally; bounds 1e4-2.5e5 J/mol, i.e. 10-250 kJ/mol).
* Rate-constant parameterization (DESIGN.md "Rate-constant
  parameterization"): ``k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))`` with
  ``T_ref`` = mid-range of the fitted data; ``log k_ref`` is the fitted
  (internal, scaled) parameter, which decorrelates prefactor and activation
  energy.  :func:`physical_params` maps internal to physical values with
  delta-method uncertainties.
* All stochastic routines take ``rng`` normalized via
  :func:`zeoal.data.as_generator`; fitting is deterministic given ``rng``.
* Uncertainties are 1-sigma.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expit

from .data import TemperatureTrace, as_generator
from .equilibrium import R_GAS, equilibrium_conversion, keq_methanation

__all__ = [
    "FeedConditions",
    "KineticFitResult",
    "KineticModel",
    "ArrheniusFirstOrder",
    "EyringFirstOrder",
    "LHHWKinetics",
    "SigmoidLightOff",
    "KINETIC_MODELS",
    "fit_trace",
    "fit_all_models",
    "select_best_model",
    "predict_trace",
    "trace_band",
    "physical_params",
]

#: Boltzmann constant over Planck constant, kB/h [K^-1 s^-1] (Eyring prefactor).
KB_OVER_H = 2.083661912e10

_CHANNELS = ("co2", "ch4", "co")


# ---------------------------------------------------------------------------
# Feed conditions
# ---------------------------------------------------------------------------
@dataclass
class FeedConditions:
    """Inlet feed description used to map conversion to concentrations.

    Parameters
    ----------
    y_co2, y_h2, y_inert:
        Inlet mole fractions of CO2, H2 and inert (normalized internally
        where ratios matter; CO2 and H2 must be positive).  Default is the
        stoichiometric methanation feed CO2:H2 = 1:4.
    pressure_bar:
        Total pressure [bar].
    total_concentration:
        Inlet CO2 concentration ``c0`` in the trace's concentration unit,
        used to scale predicted yields to concentrations.  ``None`` (default)
        means "resolve at fit time from the trace" — :meth:`KineticModel.fit`
        substitutes ``max(trace.c_co2)``; standalone :meth:`predict` calls
        with ``None`` fall back to ``c0 = 1`` (normalized yields, documented).
    """

    y_co2: float = 0.2
    y_h2: float = 0.8
    y_inert: float = 0.0
    pressure_bar: float = 1.0
    total_concentration: float | None = None

    def __post_init__(self) -> None:
        if self.y_co2 <= 0 or self.y_h2 <= 0:
            raise ValueError("feed must contain CO2 and H2 with positive mole fractions")
        if self.y_inert < 0:
            raise ValueError("y_inert must be >= 0")
        if self.pressure_bar <= 0:
            raise ValueError("pressure_bar must be > 0")
        if self.total_concentration is not None and self.total_concentration <= 0:
            raise ValueError("total_concentration must be > 0 (or None to infer from the trace)")

    def normalized_fractions(self) -> tuple[float, float, float]:
        """Inlet mole fractions normalized to sum to 1: ``(y_co2, y_h2, y_inert)``."""
        tot = self.y_co2 + self.y_h2 + self.y_inert
        return self.y_co2 / tot, self.y_h2 / tot, self.y_inert / tot


def _c0(feed: FeedConditions) -> float:
    """Inlet CO2 concentration scale; 1.0 when the feed does not specify it."""
    return 1.0 if feed.total_concentration is None else float(feed.total_concentration)


# ---------------------------------------------------------------------------
# Fit result
# ---------------------------------------------------------------------------
@dataclass
class KineticFitResult:
    """Best-fit kinetic parameters with covariance and fit statistics.

    Parameters are in the model's *internal* (scaled) parameterization —
    rate prefactors as ``log k_ref``, energies in J/mol; see
    :func:`physical_params` for physical values.  ``cov`` is the
    Gauss-Newton covariance ``sigma2 * pinv(J^T J)`` of those internal
    parameters (1-sigma ``param_sd`` on its diagonal).  ``lower``/``upper``
    are the box bounds used during fitting, kept so that
    :meth:`sample_params` can truncate posterior draws.
    """

    model_name: str
    param_names: list[str]
    params: np.ndarray
    cov: np.ndarray
    param_sd: np.ndarray
    T_ref: float
    rss: float
    dof: int
    sigma2: float
    aic: float
    bic: float
    r2: float
    success: bool
    message: str
    n_multistart: int
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.params = np.asarray(self.params, dtype=float).ravel()
        p = self.params.size
        self.cov = np.asarray(self.cov, dtype=float).reshape(p, p)
        self.param_sd = np.asarray(self.param_sd, dtype=float).ravel()
        self.lower = np.asarray(self.lower, dtype=float).ravel()
        self.upper = np.asarray(self.upper, dtype=float).ravel()

    @property
    def n_params(self) -> int:
        return self.params.size

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower, self.upper

    def sample_params(
        self, n: int, rng: np.random.Generator | int | None = None
    ) -> np.ndarray:
        """Draw ``n`` parameter vectors from ``N(params, cov)`` truncated to bounds.

        Draws from the multivariate normal (eigen-decomposition of the
        symmetrized covariance with negative eigenvalues clipped to zero, so
        a numerically indefinite ``cov`` never crashes), resamples rows that
        violate the fit bounds (up to 20 rounds), then clips any remaining
        stragglers into the box (documented clipping).

        Returns
        -------
        ndarray, shape ``(n, n_params)``.
        """
        gen = as_generator(rng)
        n = int(n)
        p = self.params.size
        cov = 0.5 * (self.cov + self.cov.T)
        evals, evecs = np.linalg.eigh(cov)
        root = evecs * np.sqrt(np.clip(evals, 0.0, None))  # cov = root @ root.T
        draws = self.params + gen.standard_normal((n, p)) @ root.T
        for _ in range(20):
            bad = np.any((draws < self.lower) | (draws > self.upper), axis=1)
            if not bad.any():
                break
            m = int(bad.sum())
            draws[bad] = self.params + gen.standard_normal((m, p)) @ root.T
        return np.clip(draws, self.lower, self.upper)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (round-trips via :meth:`from_dict`)."""
        return {
            "model_name": self.model_name,
            "param_names": list(self.param_names),
            "params": self.params.tolist(),
            "cov": self.cov.tolist(),
            "param_sd": self.param_sd.tolist(),
            "T_ref": float(self.T_ref),
            "rss": float(self.rss),
            "dof": int(self.dof),
            "sigma2": float(self.sigma2),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "r2": float(self.r2),
            "success": bool(self.success),
            "message": str(self.message),
            "n_multistart": int(self.n_multistart),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "KineticFitResult":
        return cls(
            model_name=str(d["model_name"]),
            param_names=list(d["param_names"]),
            params=np.asarray(d["params"], dtype=float),
            cov=np.asarray(d["cov"], dtype=float),
            param_sd=np.asarray(d["param_sd"], dtype=float),
            T_ref=float(d["T_ref"]),
            rss=float(d["rss"]),
            dof=int(d["dof"]),
            sigma2=float(d["sigma2"]),
            aic=float(d["aic"]),
            bic=float(d["bic"]),
            r2=float(d["r2"]),
            success=bool(d["success"]),
            message=str(d["message"]),
            n_multistart=int(d["n_multistart"]),
            lower=np.asarray(d["lower"], dtype=float),
            upper=np.asarray(d["upper"], dtype=float),
        )


def physical_params(result: KineticFitResult) -> dict[str, tuple[float, float]]:
    """Map internal (scaled/log) fit parameters to physical values with 1-sigma sd.

    ``log_*`` parameters are exponentiated; their uncertainty follows the
    first-order delta method ``sd(exp(x)) = exp(x) * sd(x)``.  All other
    parameters (activation energies, enthalpies, sigmoid parameters) pass
    through unchanged.

    Returns
    -------
    dict mapping physical parameter name (``log_`` prefix stripped) to
    ``(value, sd)``.
    """
    out: dict[str, tuple[float, float]] = {}
    for name, val, sd in zip(result.param_names, result.params, result.param_sd):
        if name.startswith("log_"):
            v = float(np.exp(val))
            out[name[4:]] = (v, float(v * sd))
        else:
            out[name] = (float(val), float(sd))
    return out


# ---------------------------------------------------------------------------
# Shared fitting machinery
# ---------------------------------------------------------------------------
def _resolve_weights(
    weights: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    """Per-channel residual weights in order (co2, ch4, co); default all 1."""
    if weights is None:
        return np.ones(3)
    if isinstance(weights, Mapping):
        unknown = set(weights) - set(_CHANNELS)
        if unknown:
            raise ValueError(f"unknown weight channel(s) {sorted(unknown)}; use {_CHANNELS}")
        w = np.array([float(weights.get(ch, 1.0)) for ch in _CHANNELS])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != 3:
            raise ValueError("weights must be a mapping over channels or a length-3 sequence")
    if np.any(w < 0) or np.any(~np.isfinite(w)):
        raise ValueError("channel weights must be finite and non-negative")
    return w


def _half_rise_temperature(T: np.ndarray, c: np.ndarray) -> float:
    """First temperature where ``c`` reaches half of its maximum (heuristic)."""
    cmax = float(np.max(c))
    if cmax <= 0:
        return float(0.5 * (T[0] + T[-1]))
    return float(T[int(np.argmax(c >= 0.5 * cmax))])


def _covariance_from_jacobian(J: np.ndarray, sigma2: float, rcond: float = 1e-12) -> np.ndarray:
    """Gauss-Newton covariance ``sigma2 * pinv(J^T J)`` with rank handling.

    ``J^T J`` is eigen-decomposed; eigenvalues below ``rcond * max_eig`` are
    treated as *unidentifiable* directions and their inverse is set to zero
    (Moore-Penrose pseudo-inverse).  Documented consequence: rank-deficient
    parameter directions get zero variance rather than infinite — a
    conservative choice; sampling then never explores those directions.
    """
    jtj = J.T @ J
    jtj = 0.5 * (jtj + jtj.T)
    evals, evecs = np.linalg.eigh(jtj)
    emax = float(np.max(evals)) if evals.size else 0.0
    cutoff = max(emax, 0.0) * rcond
    inv = np.where(evals > cutoff, 1.0 / np.where(evals > cutoff, evals, 1.0), 0.0)
    cov = (evecs * inv) @ evecs.T * sigma2
    return 0.5 * (cov + cov.T)


class KineticModel(ABC):
    """Abstract kinetic model with a shared bounded least-squares :meth:`fit`.

    Subclasses provide ``name``, ``param_names``, :meth:`default_bounds`,
    :meth:`initial_guesses` and the channel evaluation :meth:`_evaluate`.

    Parameters
    ----------
    T_ref:
        Reference temperature for the rate-constant reparameterization
        ``k = k_ref * exp(-Ea/R * (1/T - 1/T_ref))``.  ``None`` (default)
        resolves to the mid-range of the temperatures being fitted or
        predicted.
    """

    name: str = "abstract"
    param_names: list[str] = []
    #: default number of multistarts used when ``fit(n_multistart=None)``.
    default_multistart: int = 8
    #: extra keyword arguments forwarded to ``scipy.optimize.least_squares``.
    _ls_kwargs: dict[str, Any] = {}

    def __init__(self, T_ref: float | None = None):
        self.T_ref = None if T_ref is None else float(T_ref)

    # -- interface ----------------------------------------------------------
    @abstractmethod
    def default_bounds(self, trace: TemperatureTrace) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds ``(lo, hi)`` on the internal parameters."""

    @abstractmethod
    def initial_guesses(
        self,
        trace: TemperatureTrace,
        n: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """``(n, p)`` heuristic multistart initial guesses (first row unjittered)."""

    @abstractmethod
    def _evaluate(
        self,
        T: np.ndarray,
        params: np.ndarray,
        feed: FeedConditions,
        cache: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Predicted concentrations ``{"co2","ch4","co"}`` on grid ``T`` using ``cache``."""

    # -- shared pieces -------------------------------------------------------
    def _resolve_t_ref(self, T: np.ndarray) -> float:
        return self.T_ref if self.T_ref is not None else float(0.5 * (np.min(T) + np.max(T)))

    def _x_scale(self) -> np.ndarray | str:
        """Characteristic parameter scales for the TRF solver ('jac' = adaptive)."""
        return "jac"

    def _make_cache(self, T: np.ndarray, feed: FeedConditions, t_ref: float) -> dict[str, Any]:
        """Precompute per-grid quantities (built **once per fit** for speed).

        The equilibrium cap ``X_eq(T)`` calls
        :func:`zeoal.equilibrium.equilibrium_conversion` — a per-temperature
        bisection loop — so it must not be re-evaluated inside the residual
        function.
        """
        x_eq = np.atleast_1d(
            equilibrium_conversion(T, feed.y_co2, feed.y_h2, feed.y_inert, feed.pressure_bar)
        )
        return {"T_ref": float(t_ref), "X_eq": x_eq}

    def predict(
        self,
        T_K: np.ndarray,
        params: np.ndarray,
        feed: FeedConditions | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict outlet concentrations on grid ``T_K``.

        Parameters
        ----------
        T_K:
            Temperatures [K], scalar or array.
        params:
            Internal parameter vector (see ``param_names``).
        feed:
            Feed conditions; ``None`` uses the stoichiometric default.  When
            ``feed.total_concentration`` is ``None`` the outputs are
            normalized yields (``c0 = 1``, documented fallback).

        Returns
        -------
        dict with keys ``"co2"``, ``"ch4"``, ``"co"`` (same concentration
        unit as ``c0``).
        """
        feed = feed if feed is not None else FeedConditions()
        T = np.atleast_1d(np.asarray(T_K, dtype=float))
        params = np.asarray(params, dtype=float).ravel()
        if params.size != len(self.param_names):
            raise ValueError(
                f"model {self.name!r} expects {len(self.param_names)} parameters "
                f"{self.param_names}, got {params.size}"
            )
        cache = self._make_cache(T, feed, self._resolve_t_ref(T))
        return self._evaluate(T, params, feed, cache)

    def fit(
        self,
        trace: TemperatureTrace,
        feed: FeedConditions | None = None,
        n_multistart: int | None = None,
        weights: Mapping[str, float] | Sequence[float] | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> KineticFitResult:
        """Fit the model to a trace by bounded multistart least squares.

        Shared implementation (DESIGN.md `zeoal/kinetics.py`):

        * residuals stacked over the co2/ch4/co channels with optional
          per-channel ``weights`` (mapping or length-3 sequence);
        * ``scipy.optimize.least_squares`` (``method='trf'``, box bounds)
          from ``n_multistart`` jittered :meth:`initial_guesses`
          (``None`` = model default, 8; ``lhhw`` uses 4), keeping the best
          RSS; deterministic given ``rng``;
        * covariance ``sigma2 * pinv(J^T J)`` with ``sigma2 = RSS/dof``,
          ``dof = max(n_residuals - n_params, 1)``; rank-deficient
          directions receive zero variance (see
          :func:`_covariance_from_jacobian`);
        * ``feed.total_concentration is None`` resolves to
          ``max(trace.c_co2)`` (documented fallback);
        * AIC/BIC use the Gaussian profile ``n*ln(RSS/n) + penalty`` with
          ``k = n_params + 1`` (noise variance counted); R^2 on the stacked
          (weighted) channels.

        Returns
        -------
        :class:`KineticFitResult`
        """
        gen = as_generator(rng)
        feed = feed if feed is not None else FeedConditions()
        if feed.total_concentration is None:
            c0 = float(np.max(trace.c_co2))
            if c0 <= 0:
                raise ValueError(
                    "cannot infer total_concentration: trace c_co2 is non-positive; "
                    "set FeedConditions.total_concentration explicitly"
                )
            feed = replace(feed, total_concentration=c0)
        n_ms = int(self.default_multistart if n_multistart is None else n_multistart)
        if n_ms < 1:
            raise ValueError("n_multistart must be >= 1")

        T = trace.temperature_K
        t_ref = self._resolve_t_ref(T)
        cache = self._make_cache(T, feed, t_ref)
        w3 = _resolve_weights(weights)
        w_full = np.repeat(w3, T.size)
        obs = np.concatenate([trace.c_co2, trace.c_ch4, trace.c_co])

        def residuals(p: np.ndarray) -> np.ndarray:
            pred = self._evaluate(T, p, feed, cache)
            r = w_full * (np.concatenate([pred["co2"], pred["ch4"], pred["co"]]) - obs)
            # sanitize so a pathological parameter vector never crashes TRF
            return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

        lo, hi = (np.asarray(b, dtype=float).ravel() for b in self.default_bounds(trace))
        guesses = np.atleast_2d(np.asarray(self.initial_guesses(trace, n_ms, gen), dtype=float))
        inset = 1e-9 * (hi - lo)
        guesses = np.clip(guesses, lo + inset, hi - inset)

        best = None
        best_rss = np.inf
        n_failed = 0
        for x0 in guesses:
            try:
                res = least_squares(
                    residuals,
                    x0,
                    bounds=(lo, hi),
                    method="trf",
                    x_scale=self._x_scale(),
                    **self._ls_kwargs,
                )
            except Exception:
                n_failed += 1
                continue
            rss = float(np.sum(res.fun**2))
            if np.isfinite(rss) and rss < best_rss and np.all(np.isfinite(res.x)):
                best, best_rss = res, rss
        if best is None:
            raise ValueError(
                f"kinetic fit {self.name!r} failed for all {n_ms} multistarts; "
                "check the trace/feed or supply better initial guesses"
            )

        f = best.fun
        n_res = f.size
        p = best.x.size
        dof = max(n_res - p, 1)
        rss = float(np.sum(f**2))
        sigma2 = rss / dof
        cov = _covariance_from_jacobian(best.jac, sigma2)
        param_sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        yw = w_full * obs
        tss = float(np.sum((yw - yw.mean()) ** 2))
        r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
        k_eff = p + 1  # + fitted noise variance
        aic = n_res * np.log(max(rss, 1e-300) / n_res) + 2.0 * k_eff
        bic = n_res * np.log(max(rss, 1e-300) / n_res) + k_eff * np.log(n_res)
        message = str(best.message)
        if n_failed:
            message += f" ({n_failed}/{n_ms} multistarts raised and were skipped)"

        return KineticFitResult(
            model_name=self.name,
            param_names=list(self.param_names),
            params=best.x.copy(),
            cov=cov,
            param_sd=param_sd,
            T_ref=t_ref,
            rss=rss,
            dof=int(dof),
            sigma2=float(sigma2),
            aic=float(aic),
            bic=float(bic),
            r2=r2,
            success=bool(best.success),
            message=message,
            n_multistart=n_ms,
            lower=lo.copy(),
            upper=hi.copy(),
        )


# ---------------------------------------------------------------------------
# Model 1: Arrhenius parallel first-order channels
# ---------------------------------------------------------------------------
class ArrheniusFirstOrder(KineticModel):
    """Parallel pseudo-first-order methanation + RWGS channels ("arrhenius").

    Integral PFR at constant space time (absorbed into the prefactors):

    .. math::
        X_{tot}(T) = X_{eq}(T)\\,(1 - e^{-(k_m + k_r)}),\\qquad
        Y_{CH4} = X_{tot}\\frac{k_m}{k_m + k_r},\\quad
        Y_{CO} = X_{tot}\\frac{k_r}{k_m + k_r},

    with the equilibrium cap ``X_eq`` from
    :func:`zeoal.equilibrium.equilibrium_conversion` and
    ``k = k_ref * exp(-Ea/R * (1/T - 1/T_ref))`` (fit ``log k_ref``,
    ``Ea`` in J/mol, bounds 10-250 kJ/mol).  Concentrations:
    ``c_co2 = c0 (1 - X_tot)``, ``c_ch4 = c0 Y_ch4``, ``c_co = c0 Y_co``.
    Outlet mole-fraction renormalization from the mole-number change
    (dn = -2 per CO2) is a second-order effect and is *not* applied — we fit
    concentrations proportional to yields (documented approximation,
    DESIGN.md model 1).
    """

    name = "arrhenius"
    param_names = ["log_km_ref", "Ea_m", "log_kr_ref", "Ea_r"]

    def default_bounds(self, trace: TemperatureTrace) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([np.log(1e-6), 1.0e4, np.log(1e-6), 1.0e4])
        hi = np.array([np.log(1e4), 2.5e5, np.log(1e4), 2.5e5])
        return lo, hi

    def _x_scale(self) -> np.ndarray:
        return np.array([1.0, 2.0e4, 1.0, 2.0e4])

    def _rate_constants(
        self, T: np.ndarray, params: np.ndarray, t_ref: float
    ) -> tuple[np.ndarray, np.ndarray]:
        log_km, ea_m, log_kr, ea_r = params
        inv = 1.0 / T - 1.0 / t_ref
        k_m = np.exp(np.clip(log_km - ea_m / R_GAS * inv, -60.0, 60.0))
        k_r = np.exp(np.clip(log_kr - ea_r / R_GAS * inv, -60.0, 60.0))
        return k_m, k_r

    def initial_guesses(
        self,
        trace: TemperatureTrace,
        n: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Heuristic: place ``k_m(T50) = ln 2`` at the CH4 half-rise, jitter the rest."""
        gen = as_generator(rng)
        T = trace.temperature_K
        t_ref = self._resolve_t_ref(T)
        t50 = _half_rise_temperature(T, trace.c_ch4)
        ea0 = 8.0e4
        log_km0 = float(np.log(np.log(2.0)) + ea0 / R_GAS * (1.0 / t50 - 1.0 / t_ref))
        base = np.array([log_km0, ea0, log_km0 - 3.0, 1.1e5])
        scale = np.array([1.5, 2.5e4, 1.5, 3.0e4])
        guesses = base + gen.normal(size=(int(n), base.size)) * scale
        guesses[0] = base
        lo, hi = self.default_bounds(trace)
        inset = 1e-6 * (hi - lo)
        return np.clip(guesses, lo + inset, hi - inset)

    def _evaluate(
        self,
        T: np.ndarray,
        params: np.ndarray,
        feed: FeedConditions,
        cache: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        k_m, k_r = self._rate_constants(T, params, cache["T_ref"])
        k_tot = k_m + k_r
        x_tot = cache["X_eq"] * (-np.expm1(-k_tot))
        safe = np.where(k_tot > 0.0, k_tot, 1.0)  # guard division by 0
        frac_m = np.where(k_tot > 0.0, k_m / safe, 0.5)
        c0 = _c0(feed)
        return {
            "co2": c0 * (1.0 - x_tot),
            "ch4": c0 * x_tot * frac_m,
            "co": c0 * x_tot * (1.0 - frac_m),
        }


# ---------------------------------------------------------------------------
# Model 2: Eyring (transition-state theory) first-order channels
# ---------------------------------------------------------------------------
class EyringFirstOrder(ArrheniusFirstOrder):
    """Eyring/TST variant of the parallel first-order model ("eyring").

    Same channel structure as :class:`ArrheniusFirstOrder`, but

    .. math::
        k(T) = \\frac{k_B T}{h} e^{\\Delta S^\\ddagger/R}
               e^{-\\Delta H^\\ddagger/(R T)}
             = k_{ref}\\,\\frac{T}{T_{ref}}\\,
               e^{-\\Delta H^\\ddagger/R\\,(1/T - 1/T_{ref})}.

    Internal parameters: ``log_km_ref, dH_m, log_kr_ref, dH_r`` (enthalpies
    in J/mol).  The activation entropy is recovered from the fitted
    prefactor via ``dS = R * (log_k_ref - ln(kB*T_ref/h)) + dH/T_ref``
    (:data:`KB_OVER_H`; the absorbed space time shifts ``dS`` by an unknown
    additive constant — documented).
    """

    name = "eyring"
    param_names = ["log_km_ref", "dH_m", "log_kr_ref", "dH_r"]

    def _rate_constants(
        self, T: np.ndarray, params: np.ndarray, t_ref: float
    ) -> tuple[np.ndarray, np.ndarray]:
        log_km, dh_m, log_kr, dh_r = params
        inv = 1.0 / T - 1.0 / t_ref
        ratio = T / t_ref
        k_m = ratio * np.exp(np.clip(log_km - dh_m / R_GAS * inv, -60.0, 60.0))
        k_r = ratio * np.exp(np.clip(log_kr - dh_r / R_GAS * inv, -60.0, 60.0))
        return k_m, k_r


# ---------------------------------------------------------------------------
# Model 3: LHHW (Koschany 2016-inspired)
# ---------------------------------------------------------------------------
class LHHWKinetics(KineticModel):
    """Langmuir-Hinshelwood-Hougen-Watson methanation kinetics ("lhhw").

    "Koschany 2016"-inspired rate law (Appl. Catal. B 181, 504; see
    LITERATURE.md):

    .. math::
        r = k(T)\\,\\sqrt{p_{H2}\\,p_{CO2}}\\,(1 - Q/K_{eq})/\\mathrm{DEN}^2,

    with ``DEN = 1 + K_OH p_H2O / sqrt(p_H2) + K_H2 sqrt(p_H2)
    + K_mix sqrt(p_CO2)``, approach to equilibrium
    ``Q = p_CH4 p_H2O^2 / (p_CO2 p_H2^4)`` and ``Keq`` from
    :func:`zeoal.equilibrium.keq_methanation` (both in bar^-2).  An
    isothermal ideal PFR ``dX/dz = k * r_norm`` is integrated over the
    normalized space coordinate ``z in [0, 1]`` per temperature with
    ``scipy.integrate.solve_ivp`` (LSODA, Radau fallback); the space time /
    Damkoehler number is absorbed into ``k_ref``.  ``X`` is clipped to
    ``[0, X_eq(T)]``.  Partial pressures follow the methanation
    stoichiometry as functions of ``X`` (basis 1 mol feed, mole change
    dn = -2 per CO2).

    Robustness (DESIGN.md): a solver failure at one temperature reuses the
    previous temperature's conversion (or 0) and records a warning — the fit
    never crashes.

    Identifiability: ``K_H2`` and ``K_mix`` are weakly identified from a
    single light-off trace and are **fixed** at Koschany-like values
    (``K_H2 = 0.44``, ``K_mix = 0.88`` bar^-0.5 at the 555 K reference)
    unless ``release_adsorption=True`` adds ``log_K_H2, log_K_mix``
    (T-independent) to the fit.  ``K_OH`` is fitted with van 't Hoff
    T-dependence ``K_OH = K_OH_ref * exp(dH_OH/R * (1/T_ref - 1/T))``
    (Koschany: ``dH_OH ~ +22 kJ/mol`` around ``K_OH_ref ~ 0.5 bar^-0.5``).
    The CO channel is the same first-order RWGS add-on as ``"arrhenius"``,
    applied to the CO2 left after methanation (sequential variant,
    documented): ``c_co = c0 (1 - X)(1 - e^{-k_r})``,
    ``c_co2 = c0 (1 - X) e^{-k_r}``, ``c_ch4 = c0 X`` (channels sum to c0).
    """

    name = "lhhw"
    param_names = ["log_k_ref", "Ea", "log_K_OH_ref", "dH_OH", "log_kr_ref", "Ea_r"]
    default_multistart = 4  # hardest model; keep multistart small for speed
    _ls_kwargs = {"ftol": 1e-8, "xtol": 1e-8, "max_nfev": 300}

    def __init__(
        self,
        T_ref: float | None = None,
        release_adsorption: bool = False,
        K_H2: float = 0.44,
        K_mix: float = 0.88,
    ):
        super().__init__(T_ref=T_ref)
        self.release_adsorption = bool(release_adsorption)
        self.K_H2 = float(K_H2)
        self.K_mix = float(K_mix)
        if self.release_adsorption:
            self.param_names = [*LHHWKinetics.param_names, "log_K_H2", "log_K_mix"]

    def default_bounds(self, trace: TemperatureTrace) -> tuple[np.ndarray, np.ndarray]:
        lo = [np.log(1e-5), 1.0e4, np.log(1e-3), -5.0e4, np.log(1e-6), 1.0e4]
        hi = [np.log(1e5), 2.5e5, np.log(1e2), 1.2e5, np.log(1e4), 2.5e5]
        if self.release_adsorption:
            lo += [np.log(1e-3), np.log(1e-3)]
            hi += [np.log(1e2), np.log(1e2)]
        return np.asarray(lo), np.asarray(hi)

    def _x_scale(self) -> np.ndarray:
        scale = [1.0, 2.0e4, 1.0, 2.0e4, 1.0, 2.0e4]
        if self.release_adsorption:
            scale += [1.0, 1.0]
        return np.asarray(scale)

    def initial_guesses(
        self,
        trace: TemperatureTrace,
        n: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Heuristic around Koschany 2016 values, light-off placed at the CH4 half-rise."""
        gen = as_generator(rng)
        T = trace.temperature_K
        t_ref = self._resolve_t_ref(T)
        t50 = _half_rise_temperature(T, trace.c_ch4)
        feed = FeedConditions()  # heuristic scale only
        y_co2, y_h2, _ = feed.normalized_fractions()
        den0 = 1.0 + self.K_H2 * np.sqrt(y_h2) + self.K_mix * np.sqrt(y_co2)
        g0 = np.sqrt(y_h2 * y_co2) / den0**2
        ea0 = 8.0e4
        log_k0 = float(np.log(np.log(2.0) / g0) + ea0 / R_GAS * (1.0 / t50 - 1.0 / t_ref))
        base = [log_k0, ea0, np.log(0.5), 2.24e4, log_k0 - 4.0, 1.1e5]
        scale = [1.5, 2.0e4, 0.8, 2.0e4, 1.5, 3.0e4]
        if self.release_adsorption:
            base += [np.log(self.K_H2), np.log(self.K_mix)]
            scale += [0.5, 0.5]
        base = np.asarray(base)
        guesses = base + gen.normal(size=(int(n), base.size)) * np.asarray(scale)
        guesses[0] = base
        lo, hi = self.default_bounds(trace)
        inset = 1e-6 * (hi - lo)
        return np.clip(guesses, lo + inset, hi - inset)

    def _make_cache(self, T: np.ndarray, feed: FeedConditions, t_ref: float) -> dict[str, Any]:
        cache = super()._make_cache(T, feed, t_ref)
        y_co2, y_h2, y_inert = feed.normalized_fractions()
        cache["keq"] = np.atleast_1d(keq_methanation(T))
        cache["y_co2"], cache["y_h2"], cache["y_inert"] = y_co2, y_h2, y_inert
        cache["x_max"] = min(1.0, y_h2 / (4.0 * y_co2)) - 1e-9
        return cache

    def _evaluate(
        self,
        T: np.ndarray,
        params: np.ndarray,
        feed: FeedConditions,
        cache: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        if self.release_adsorption:
            log_k, ea, log_koh, dh_oh, log_kr, ea_r, log_kh2, log_kmix = params
            k_h2 = float(np.exp(log_kh2))
            k_mix = float(np.exp(log_kmix))
        else:
            log_k, ea, log_koh, dh_oh, log_kr, ea_r = params
            k_h2, k_mix = self.K_H2, self.K_mix
        t_ref = cache["T_ref"]
        inv = 1.0 / T - 1.0 / t_ref
        k = np.exp(np.clip(log_k - ea / R_GAS * inv, -60.0, 60.0))
        k_oh = np.exp(np.clip(log_koh - dh_oh / R_GAS * inv, -60.0, 60.0))  # van 't Hoff
        k_r = np.exp(np.clip(log_kr - ea_r / R_GAS * inv, -60.0, 60.0))

        keq = cache["keq"]
        x_eq = cache["X_eq"]
        y_co2, y_h2 = cache["y_co2"], cache["y_h2"]
        x_hard = cache["x_max"]
        p_tot = float(feed.pressure_bar)
        # inlet rate factor (no H2O at inlet -> no K_OH term); used to skip
        # temperatures where the rate is negligible: X(1) <= k * g0.
        den0 = 1.0 + k_h2 * np.sqrt(p_tot * y_h2) + k_mix * np.sqrt(p_tot * y_co2)
        g0 = np.sqrt(p_tot * y_h2 * p_tot * y_co2) / den0**2

        X = np.zeros(T.size)
        prev = 0.0
        n_fail = 0
        for i in range(T.size):
            cap = min(float(x_eq[i]), x_hard)
            if cap <= 0.0 or k[i] * g0 < 1e-10:
                X[i] = 0.0
                prev = 0.0
                continue
            ki, keqi, kohi = float(k[i]), float(keq[i]), float(k_oh[i])

            def rhs(z: float, state: np.ndarray) -> tuple[float]:
                x = state[0]
                x = 0.0 if x < 0.0 else (cap if x > cap else x)
                xi = x * y_co2
                ntot = 1.0 - 2.0 * xi
                p_co2 = max(p_tot * (y_co2 - xi) / ntot, 1e-12)
                p_h2 = max(p_tot * (y_h2 - 4.0 * xi) / ntot, 1e-12)
                p_ch4 = p_tot * xi / ntot
                p_h2o = 2.0 * p_tot * xi / ntot
                s_h2 = np.sqrt(p_h2)
                den = 1.0 + kohi * p_h2o / s_h2 + k_h2 * s_h2 + k_mix * np.sqrt(p_co2)
                q = p_ch4 * p_h2o * p_h2o / (p_co2 * p_h2**4)
                rate = ki * s_h2 * np.sqrt(p_co2) * (1.0 - q / keqi) / (den * den)
                if state[0] >= cap and rate > 0.0:
                    rate = 0.0
                return (rate,)

            x_end = None
            for method in ("LSODA", "Radau"):
                try:
                    sol = solve_ivp(rhs, (0.0, 1.0), (0.0,), method=method, rtol=1e-5, atol=1e-8)
                except Exception:
                    continue
                if sol.success:
                    x_end = float(sol.y[0, -1])
                    break
            if x_end is None:
                n_fail += 1
                x_end = prev
            X[i] = min(max(x_end, 0.0), float(x_eq[i]))  # documented clip to [0, X_eq]
            prev = X[i]
        if n_fail:
            warnings.warn(
                "lhhw: PFR integration failed at one or more temperatures; "
                "the previous temperature's conversion was carried over",
                RuntimeWarning,
                stacklevel=2,
            )
        c0 = _c0(feed)
        surv = np.exp(-k_r)  # first-order RWGS survival of leftover CO2
        return {
            "co2": c0 * (1.0 - X) * surv,
            "ch4": c0 * X,
            "co": c0 * (1.0 - X) * (1.0 - surv),
        }


# ---------------------------------------------------------------------------
# Model 4: empirical sigmoid light-off
# ---------------------------------------------------------------------------
class SigmoidLightOff(KineticModel):
    """Empirical sigmoidal light-off model ("sigmoid"), robust fallback.

    .. math::
        Y_{CH4}(T) = A\\,\\mathrm{expit}((T - T_{50})/w)\\,f_{eq}(T),\\qquad
        Y_{CO}(T) = A_{co}\\,\\mathrm{expit}((T - T_{50,co})/w_{co}),

    where ``f_eq(T) = X_eq(T) / max(X_eq)`` is the normalized equilibrium
    decline evaluated on the current grid (the fit grid during fitting).
    Concentrations: ``c_ch4 = c0 Y_ch4``, ``c_co = c0 Y_co`` and the carbon
    balance ``c_co2 = c0 - c_ch4 - c_co`` **clipped at >= 0** (documented
    clipping — the empirical channels are not constrained to conserve
    carbon).  Parameters ``A, T50, w, A_co, T50_co, w_co`` are already
    physical (no log-scaling); ``T50`` is directly interpretable.
    """

    name = "sigmoid"
    param_names = ["A", "T50", "w", "A_co", "T50_co", "w_co"]

    def default_bounds(self, trace: TemperatureTrace) -> tuple[np.ndarray, np.ndarray]:
        T = trace.temperature_K
        lo = np.array([0.0, T[0] - 200.0, 1.0, 0.0, T[0] - 100.0, 1.0])
        hi = np.array([2.0, T[-1] + 200.0, 300.0, 1.0, T[-1] + 400.0, 400.0])
        return lo, hi

    def _x_scale(self) -> np.ndarray:
        return np.array([0.2, 30.0, 15.0, 0.2, 30.0, 15.0])

    def initial_guesses(
        self,
        trace: TemperatureTrace,
        n: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        gen = as_generator(rng)
        T = trace.temperature_K
        c0_est = float(np.max(trace.c_co2))
        if c0_est <= 0:
            c0_est = max(float(np.max(trace.c_ch4)), 1e-12)
        a0 = float(np.clip(np.max(trace.c_ch4) / c0_est, 0.05, 1.5))
        a_co0 = float(np.clip(np.max(trace.c_co) / c0_est, 0.005, 0.9))
        base = np.array(
            [a0, _half_rise_temperature(T, trace.c_ch4), 25.0, a_co0, float(T[-1]), 60.0]
        )
        scale = np.array([0.15, 40.0, 12.0, 0.05, 60.0, 30.0])
        guesses = base + gen.normal(size=(int(n), base.size)) * scale
        guesses[0] = base
        lo, hi = self.default_bounds(trace)
        inset = 1e-6 * (hi - lo)
        return np.clip(guesses, lo + inset, hi - inset)

    def _make_cache(self, T: np.ndarray, feed: FeedConditions, t_ref: float) -> dict[str, Any]:
        cache = super()._make_cache(T, feed, t_ref)
        x_eq = cache["X_eq"]
        cache["f_eq"] = x_eq / max(float(np.max(x_eq)), 1e-12)
        return cache

    def _evaluate(
        self,
        T: np.ndarray,
        params: np.ndarray,
        feed: FeedConditions,
        cache: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        a, t50, w, a_co, t50_co, w_co = params
        y_ch4 = a * expit((T - t50) / max(w, 1e-6)) * cache["f_eq"]
        y_co = a_co * expit((T - t50_co) / max(w_co, 1e-6))
        c0 = _c0(feed)
        c_ch4 = c0 * y_ch4
        c_co = c0 * y_co
        c_co2 = np.clip(c0 - c_ch4 - c_co, 0.0, None)  # documented clip at >= 0
        return {"co2": c_co2, "ch4": c_ch4, "co": c_co}


# ---------------------------------------------------------------------------
# Registry + helpers
# ---------------------------------------------------------------------------
KINETIC_MODELS: dict[str, type[KineticModel]] = {
    ArrheniusFirstOrder.name: ArrheniusFirstOrder,
    EyringFirstOrder.name: EyringFirstOrder,
    LHHWKinetics.name: LHHWKinetics,
    SigmoidLightOff.name: SigmoidLightOff,
}


def _get_model(
    model: str | KineticModel, T_ref: float | None = None, **model_kwargs: Any
) -> KineticModel:
    """Instantiate a registered model by name (or pass an instance through)."""
    if isinstance(model, KineticModel):
        return model
    try:
        cls = KINETIC_MODELS[model]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown kinetic model {model!r}; available: {sorted(KINETIC_MODELS)}"
        ) from None
    return cls(T_ref=T_ref, **model_kwargs)


def fit_trace(
    trace: TemperatureTrace,
    model: str | KineticModel = "arrhenius",
    feed: FeedConditions | None = None,
    n_multistart: int | None = None,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    rng: np.random.Generator | int | None = None,
    T_ref: float | None = None,
    **model_kwargs: Any,
) -> KineticFitResult:
    """Fit one trace with one kinetic model (see :meth:`KineticModel.fit`).

    Parameters
    ----------
    trace:
        Measured light-off trace.
    model:
        Registry name (``"arrhenius"``, ``"eyring"``, ``"lhhw"``,
        ``"sigmoid"``) or a :class:`KineticModel` instance.
    feed:
        Feed conditions; ``None`` uses the stoichiometric default and
        ``total_concentration`` resolves to ``max(trace.c_co2)``.
    n_multistart:
        ``None`` = model default (8; ``lhhw`` uses 4).
    weights:
        Optional per-channel residual weights.
    rng:
        Random source for the jittered multistarts (deterministic given it).
    T_ref:
        Override the reference temperature (default: trace mid-range).
    **model_kwargs:
        Extra constructor arguments (e.g. ``release_adsorption=True`` for
        ``lhhw``); ignored when ``model`` is already an instance.
    """
    m = _get_model(model, T_ref=T_ref, **model_kwargs)
    return m.fit(trace, feed=feed, n_multistart=n_multistart, weights=weights, rng=rng)


def fit_all_models(
    trace: TemperatureTrace,
    models: Sequence[str] | None = None,
    feed: FeedConditions | None = None,
    n_multistart: int | None = None,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    rng: np.random.Generator | int | None = None,
) -> dict[str, KineticFitResult]:
    """Fit several kinetic models to one trace.

    ``models=None`` fits every registered model (note ``"lhhw"`` is the
    slowest).  A model whose fit raises is skipped with a warning, so one
    pathological model never loses the others' results.  Deterministic given
    ``rng`` (the models draw from one shared generator in list order).
    """
    gen = as_generator(rng)
    names = list(KINETIC_MODELS) if models is None else list(models)
    out: dict[str, KineticFitResult] = {}
    for name in names:
        try:
            out[name] = fit_trace(
                trace, model=name, feed=feed, n_multistart=n_multistart, weights=weights, rng=gen
            )
        except Exception as exc:  # robustness: keep the other models
            warnings.warn(f"kinetic model {name!r} failed to fit: {exc}", RuntimeWarning)
    return out


def select_best_model(
    results: Mapping[str, KineticFitResult], criterion: str = "aic"
) -> str:
    """Name of the best fit by ``criterion`` (``"aic"``/``"bic"``/``"rss"`` min, ``"r2"`` max).

    Successful fits (``success=True``) are preferred; unsuccessful ones are
    only considered when no fit converged.  Non-finite criteria rank last.
    """
    if not results:
        raise ValueError("no fit results to select from")
    crit = str(criterion).lower()
    if crit not in ("aic", "bic", "rss", "r2"):
        raise ValueError(f"criterion must be one of 'aic', 'bic', 'rss', 'r2'; got {criterion!r}")
    sign = -1.0 if crit == "r2" else 1.0

    def score(res: KineticFitResult) -> float:
        v = float(getattr(res, crit))
        return sign * v if np.isfinite(v) else np.inf

    pool = {k: r for k, r in results.items() if r.success}
    if not pool:
        pool = dict(results)
    return min(pool, key=lambda k: score(pool[k]))


def predict_trace(
    model_name: str | KineticModel,
    T_K: np.ndarray,
    params: np.ndarray,
    feed: FeedConditions | None = None,
    T_ref: float | None = None,
    concentration_unit: str = "mol_frac",
) -> TemperatureTrace:
    """Forward-predict a :class:`~zeoal.data.TemperatureTrace` from parameters.

    ``T_ref`` should be the fit's reference temperature
    (:attr:`KineticFitResult.T_ref`) when ``params`` come from a fit on a
    different grid; ``None`` uses the mid-range of ``T_K``.  ``T_K`` needs at
    least 3 points (:class:`~zeoal.data.TemperatureTrace` requirement).
    """
    model = _get_model(model_name, T_ref=T_ref)
    T = np.atleast_1d(np.asarray(T_K, dtype=float))
    pred = model.predict(T, params, feed)
    return TemperatureTrace(T, pred["co2"], pred["ch4"], pred["co"], concentration_unit)


def trace_band(
    model_name: str | KineticModel,
    T_K: np.ndarray,
    fit_result: KineticFitResult,
    feed: FeedConditions | None = None,
    n_samples: int = 200,
    rng: np.random.Generator | int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Monte-Carlo uncertainty band of the predicted trace.

    Draws ``n_samples`` parameter vectors from
    :meth:`KineticFitResult.sample_params` (truncated MVN), pushes each
    through the model on grid ``T_K`` (using the fit's ``T_ref``; the
    per-grid cache incl. the equilibrium cap is built once) and summarizes
    per channel.

    Returns
    -------
    dict mapping channel (``"co2"``, ``"ch4"``, ``"co"``) to
    ``(mean, lo, hi)`` arrays over ``T_K``, where ``lo``/``hi`` are the
    16th/84th percentiles (~ +/- 1 sigma).
    """
    gen = as_generator(rng)
    model = _get_model(model_name, T_ref=fit_result.T_ref)
    feed = feed if feed is not None else FeedConditions()
    T = np.atleast_1d(np.asarray(T_K, dtype=float))
    cache = model._make_cache(T, feed, float(fit_result.T_ref))
    draws = fit_result.sample_params(int(n_samples), gen)
    sims = {ch: np.empty((draws.shape[0], T.size)) for ch in _CHANNELS}
    for i, p in enumerate(draws):
        pred = model._evaluate(T, p, feed, cache)
        for ch in _CHANNELS:
            sims[ch][i] = pred[ch]
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ch in _CHANNELS:
        arr = sims[ch]
        lo, hi = np.percentile(arr, [16.0, 84.0], axis=0)
        out[ch] = (arr.mean(axis=0), lo, hi)
    return out
