"""Tests for zeoal.fom (figures of merit)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from scipy.special import expit, logit

from zeoal.data import TemperatureTrace, as_generator
from zeoal.fom import (
    FOM_INFO,
    FOM_REGISTRY,
    FOMResult,
    ch4_temperature_integral,
    fom_from_fit,
    fom_from_trace,
)

_trap = getattr(np, "trapezoid", None) or np.trapz


def make_trace(T, ch4, co2=None, co=None):
    T = np.asarray(T, dtype=float)
    ch4 = np.asarray(ch4, dtype=float)
    co2 = np.full_like(T, 0.2) if co2 is None else np.asarray(co2, dtype=float)
    co = np.zeros_like(T) if co is None else np.asarray(co, dtype=float)
    return TemperatureTrace(T, co2, ch4, co)


def triangle_trace(n=101, T0=400.0, T1=500.0, peak=0.08):
    """Piecewise-linear triangle peaking mid-range; trapezoid is exact on it."""
    T = np.linspace(T0, T1, n)
    mid = 0.5 * (T0 + T1)
    ch4 = peak * np.clip(1.0 - np.abs(T - mid) / (mid - T0), 0.0, None)
    return make_trace(T, ch4)


def trapezoid_weights(T):
    w = np.zeros_like(T)
    dT = np.diff(T)
    w[:-1] += 0.5 * dT
    w[1:] += 0.5 * dT
    return w


# --------------------------------------------------------------------------
# registry / metadata
# --------------------------------------------------------------------------

def test_fom_info_covers_registry():
    assert set(FOM_INFO) == set(FOM_REGISTRY)
    for name, info in FOM_INFO.items():
        assert isinstance(info["maximize"], bool), name
        assert isinstance(info["description"], str) and info["description"], name
        assert isinstance(info["units"], str) and info["units"], name
    # temperatures are minimized, integrals/concentrations maximized
    for name in ("t10", "t50", "t90", "peak_temperature"):
        assert FOM_INFO[name]["maximize"] is False
    assert FOM_INFO["ch4_temperature_integral"]["maximize"] is True


def test_registry_callable_direct_and_unsorted_input():
    # registry contract: f(T_K, c_ch4, c_co2, c_co, **kw) -> float; inputs
    # are sorted by temperature internally
    T = np.array([500.0, 400.0, 450.0])
    ch4 = np.array([0.0, 0.0, 0.08])  # triangle when sorted
    zeros = np.zeros_like(T)
    val = ch4_temperature_integral(T, ch4, zeros, zeros)
    assert val == pytest.approx(0.5 * 100.0 * 0.08, rel=1e-12)
    assert val == FOM_REGISTRY["ch4_temperature_integral"](T, ch4, zeros, zeros)


def test_unknown_fom_raises():
    tr = triangle_trace(n=11)
    with pytest.raises(ValueError, match="unknown FOM"):
        fom_from_trace(tr, name="not_a_fom")


def test_fom_result_fields():
    res = FOMResult(value=1.5, sd=None, name="x", details={"k": 1})
    assert res.value == 1.5 and res.sd is None and res.name == "x"
    assert res.details == {"k": 1}


# --------------------------------------------------------------------------
# baseline integral
# --------------------------------------------------------------------------

def test_triangle_integral_matches_closed_form():
    peak, T0, T1 = 0.08, 400.0, 500.0
    tr = triangle_trace(n=101, T0=T0, T1=T1, peak=peak)
    res = fom_from_trace(tr)  # default name = baseline
    expected = 0.5 * (T1 - T0) * peak  # triangle area
    assert res.name == "ch4_temperature_integral"
    assert res.value == pytest.approx(expected, rel=1e-12)
    assert res.sd is None  # no noise information given


def test_integral_equals_numpy_trapezoid():
    rng = np.random.default_rng(0)
    T = np.linspace(350.0, 650.0, 31)
    ch4 = rng.uniform(0.0, 0.1, size=T.size)
    tr = make_trace(T, ch4)
    res = fom_from_trace(tr)
    assert res.value == pytest.approx(float(_trap(ch4, T)), rel=1e-12)


def test_window_clipping_constant_trace():
    T = np.linspace(300.0, 500.0, 26)  # step 8 K; 350/450 are off-grid
    c0 = 0.05
    tr = make_trace(T, np.full_like(T, c0))
    res = fom_from_trace(tr, T_min=350.0, T_max=450.0)
    assert res.value == pytest.approx(c0 * 100.0, rel=1e-12)
    assert res.details["window"] == (350.0, 450.0)
    # window wider than data clamps to the data range (documented clipping)
    res_wide = fom_from_trace(tr, T_min=100.0, T_max=900.0)
    assert res_wide.value == pytest.approx(c0 * 200.0, rel=1e-12)
    # one-sided window
    res_lo = fom_from_trace(tr, T_min=421.0)
    assert res_lo.value == pytest.approx(c0 * (500.0 - 421.0), rel=1e-12)
    # non-overlapping window is an error
    with pytest.raises(ValueError, match="window"):
        fom_from_trace(tr, T_min=600.0)


def test_window_edge_interpolation_linear_ramp():
    # exact for piecewise-linear data with off-grid window edges
    T = np.linspace(300.0, 500.0, 26)
    ch4 = 1e-4 * (T - 300.0)
    tr = make_trace(T, ch4)
    res = fom_from_trace(tr, T_min=350.0, T_max=450.0)
    # ∫_350^450 1e-4 (T-300) dT = 1e-4 * (150^2 - 50^2)/2 = 1.0
    assert res.value == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------
# weighted / selectivity / penalized integrals
# --------------------------------------------------------------------------

def test_weighted_integral_one_over_T():
    T = np.linspace(400.0, 600.0, 21)
    ch4 = 0.02 + 0.03 * np.sin((T - 400.0) / 60.0)
    tr = make_trace(T, ch4)
    res = fom_from_trace(tr, name="weighted_ch4_integral", weight="one_over_T")
    assert res.value == pytest.approx(float(_trap(ch4 / T, T)), rel=1e-12)


def test_weighted_integral_low_T_boltzmann_defaults():
    T = np.linspace(400.0, 600.0, 21)
    ch4 = np.full_like(T, 0.05)
    tr = make_trace(T, ch4)
    res = fom_from_trace(tr, name="weighted_ch4_integral", weight="low_T_boltzmann")
    w = np.exp(-(T - T.min()) / 100.0)  # defaults: T0 = T.min(), tau_w = 100
    assert res.value == pytest.approx(float(_trap(ch4 * w, T)), rel=1e-12)
    # explicit T0 / tau_w
    res2 = fom_from_trace(
        tr, name="weighted_ch4_integral", weight="low_T_boltzmann", T0=450.0, tau_w=50.0
    )
    w2 = np.exp(-(T - 450.0) / 50.0)
    assert res2.value == pytest.approx(float(_trap(ch4 * w2, T)), rel=1e-12)


def test_weighted_integral_unknown_weight_raises():
    tr = triangle_trace(n=11)
    with pytest.raises(ValueError, match="weight"):
        fom_from_trace(tr, name="weighted_ch4_integral", weight="bogus")


def test_selectivity_integral_zero_denominator():
    T = np.linspace(400.0, 500.0, 11)
    ch4 = np.where(T >= 450.0, 0.04, 0.0)  # zeros at low T
    co = np.zeros_like(T)  # 0/0 at low T, S=1 where ch4>0
    tr = make_trace(T, ch4, co=co)
    res = fom_from_trace(tr, name="selectivity_weighted_integral")
    plain = fom_from_trace(tr, name="ch4_temperature_integral")
    assert np.isfinite(res.value)
    assert res.value == pytest.approx(plain.value, rel=1e-12)
    # all-zero trace -> integral 0, still finite with analytic sd
    tr0 = make_trace(T, np.zeros_like(T), co=np.zeros_like(T))
    res0 = fom_from_trace(tr0, name="selectivity_weighted_integral", noise_sd=0.01)
    assert res0.value == 0.0
    assert np.isfinite(res0.sd)


def test_selectivity_integral_partial_selectivity():
    T = np.linspace(400.0, 500.0, 11)
    ch4 = np.full_like(T, 0.03)
    co = np.full_like(T, 0.01)  # S = 0.75 everywhere
    tr = make_trace(T, ch4, co=co)
    res = fom_from_trace(tr, name="selectivity_weighted_integral")
    assert res.value == pytest.approx(0.03 * 0.75 * 100.0, rel=1e-12)


def test_co_penalized_integral():
    T = np.linspace(400.0, 500.0, 11)
    tr = make_trace(T, np.full_like(T, 0.05), co=np.full_like(T, 0.01))
    res = fom_from_trace(tr, name="co_penalized_integral")  # lambda_co = 1
    assert res.value == pytest.approx((0.05 - 0.01) * 100.0, rel=1e-12)
    res2 = fom_from_trace(tr, name="co_penalized_integral", lambda_co=2.0)
    assert res2.value == pytest.approx((0.05 - 0.02) * 100.0, rel=1e-12)


# --------------------------------------------------------------------------
# threshold / pointwise FOMs
# --------------------------------------------------------------------------

def test_t50_t10_t90_on_clean_sigmoid():
    T = np.arange(300.0, 601.0, 1.0)
    ch4 = expit((T - 450.0) / 20.0)
    tr = make_trace(T, ch4)
    cmax = expit((T.max() - 450.0) / 20.0)
    for name, frac in (("t50", 0.5), ("t10", 0.1), ("t90", 0.9)):
        res = fom_from_trace(tr, name=name)
        expected = 450.0 + 20.0 * logit(frac * cmax)  # exact crossing of frac*max
        assert res.value == pytest.approx(expected, abs=0.5), name  # grid tol
        assert res.details["crossed"] is True
    # sanity: midpoint of the sigmoid
    assert fom_from_trace(tr, name="t50").value == pytest.approx(450.0, abs=1.0)


def test_threshold_non_crossing_returns_tmax():
    T = np.linspace(400.0, 500.0, 11)
    tr = make_trace(T, np.zeros_like(T))
    res = fom_from_trace(tr, name="t50")
    assert res.value == T.max()
    assert res.details["crossed"] is False
    assert "note" in res.details


def test_pointwise_foms():
    T = np.linspace(400.0, 500.0, 11)
    ch4 = np.exp(-((T - 460.0) / 25.0) ** 2) * 0.07
    tr = make_trace(T, ch4)
    assert fom_from_trace(tr, name="max_ch4").value == pytest.approx(ch4.max())
    assert fom_from_trace(tr, name="peak_temperature").value == T[np.argmax(ch4)]
    res = fom_from_trace(tr, name="ch4_at_T", T_eval=455.5)
    assert res.value == pytest.approx(float(np.interp(455.5, T, ch4)), rel=1e-12)
    with pytest.raises(ValueError, match="T_eval"):
        fom_from_trace(tr, name="ch4_at_T")


# --------------------------------------------------------------------------
# uncertainty: analytic vs Monte Carlo
# --------------------------------------------------------------------------

def test_analytic_sd_matches_trapezoid_weight_formula():
    tr = triangle_trace(n=15)
    T = tr.temperature_K
    sigma = 0.004
    res = fom_from_trace(tr, noise_sd=sigma)
    w = trapezoid_weights(T)
    expected = sigma * np.sqrt(np.sum(w**2))
    assert res.details["sd_method"] == "analytic"
    assert res.sd == pytest.approx(expected, rel=1e-12)
    # per-point noise array
    sig_arr = np.linspace(0.001, 0.01, T.size)
    res2 = fom_from_trace(tr, noise_sd=sig_arr)
    assert res2.sd == pytest.approx(np.sqrt(np.sum((w * sig_arr) ** 2)), rel=1e-12)


def test_analytic_sd_matches_mc_on_linear_fom():
    tr = triangle_trace(n=21)
    sigma = 0.005
    res_a = fom_from_trace(tr, noise_sd=sigma)
    res_mc = fom_from_trace(tr, noise_sd=sigma, sd_method="mc", n_samples=3000, rng=1)
    assert res_a.details["sd_method"] == "analytic"
    assert res_mc.details["sd_method"] == "mc"
    assert res_mc.sd == pytest.approx(res_a.sd, rel=0.15)


def test_analytic_sd_matches_mc_with_window():
    tr = triangle_trace(n=21)  # grid step 5 K
    sigma = 0.005
    kw = dict(T_min=422.0, T_max=478.0)  # off-grid edges exercise the stencil
    res_a = fom_from_trace(tr, noise_sd=sigma, **kw)
    res_mc = fom_from_trace(
        tr, noise_sd=sigma, sd_method="mc", n_samples=3000, rng=2, **kw
    )
    assert res_mc.sd == pytest.approx(res_a.sd, rel=0.15)


def test_co_penalized_analytic_sd_includes_co_noise():
    T = np.linspace(400.0, 500.0, 15)
    tr = make_trace(T, np.full_like(T, 0.05), co=np.full_like(T, 0.01))
    sigma = 0.003
    res = fom_from_trace(tr, name="co_penalized_integral", noise_sd=sigma)
    w = trapezoid_weights(T)
    # independent noise on c_ch4 and c_co doubles the variance at lambda=1
    expected = sigma * np.sqrt(2.0 * np.sum(w**2))
    assert res.sd == pytest.approx(expected, rel=1e-12)


def test_mc_sd_for_t50_deterministic_given_rng():
    T = np.arange(350.0, 551.0, 5.0)
    tr = make_trace(T, expit((T - 450.0) / 20.0) * 0.08)
    r1 = fom_from_trace(tr, name="t50", noise_sd=0.004, rng=7, n_samples=200)
    r2 = fom_from_trace(tr, name="t50", noise_sd=0.004, rng=7, n_samples=200)
    r3 = fom_from_trace(tr, name="t50", noise_sd=0.004, rng=8, n_samples=200)
    assert r1.details["sd_method"] == "mc"
    assert r1.sd == r2.sd and r1.details["samples_mean"] == r2.details["samples_mean"]
    assert r1.sd != r3.sd
    assert r1.sd > 0.0
    # forcing analytic on a nonlinear FOM is an error
    with pytest.raises(ValueError, match="analytic"):
        fom_from_trace(tr, name="t50", noise_sd=0.004, sd_method="analytic")


def test_noise_sd_validation():
    tr = triangle_trace(n=11)
    with pytest.raises(ValueError, match="noise_sd"):
        fom_from_trace(tr, noise_sd=-0.1)
    with pytest.raises(ValueError, match="n_samples"):
        fom_from_trace(tr, name="t50", noise_sd=0.01, n_samples=1)


# --------------------------------------------------------------------------
# fom_from_fit with a stubbed zeoal.kinetics
# --------------------------------------------------------------------------

class FakeFitResult:
    """Stub of kinetics.KineticFitResult: one amplitude parameter ~ N(mu, sd)."""

    def __init__(self, mu=1.0, sd=0.1):
        self.params = np.array([mu, 0.0])
        self._mu = mu
        self._sd = sd

    def sample_params(self, n, rng):
        gen = as_generator(rng)
        amp = self._mu + self._sd * gen.standard_normal(n)
        return np.column_stack([amp, np.zeros(n)])


def install_stub_kinetics(monkeypatch):
    """Install a fake zeoal.kinetics whose predict_trace scales a fixed bump."""
    mod = types.ModuleType("zeoal.kinetics")

    def predict_trace(model_name, T_K, params, feed):
        T = np.asarray(T_K, dtype=float)
        ch4 = float(params[0]) * np.exp(-(((T - 450.0) / 30.0) ** 2))
        return TemperatureTrace(T, 0.2 - 0.1 * ch4, ch4, np.zeros_like(T))

    mod.predict_trace = predict_trace
    monkeypatch.setitem(sys.modules, "zeoal.kinetics", mod)
    return mod


def test_fom_from_fit_with_stub_kinetics(monkeypatch):
    install_stub_kinetics(monkeypatch)
    T_grid = np.linspace(400.0, 500.0, 21)
    base = float(_trap(np.exp(-(((T_grid - 450.0) / 30.0) ** 2)), T_grid))
    fit = FakeFitResult(mu=1.0, sd=0.1)
    res = fom_from_fit(fit, "stub_model", T_grid, feed=None, n_samples=400, rng=3)
    # FOM is linear in the sampled amplitude: mean ~ base, sd ~ 0.1 * base
    assert res.value == pytest.approx(base, rel=0.03)
    assert res.sd == pytest.approx(0.1 * base, rel=0.25)
    assert res.details["samples_sd"] == res.sd
    assert res.details["samples_mean"] == res.value
    assert res.details["model_name"] == "stub_model"
    assert res.details["n_used"] == 400 and res.details["n_failed"] == 0
    # point prediction at the best-fit params (amplitude exactly 1)
    assert res.details["best_fit_value"] == pytest.approx(base, rel=1e-12)


def test_fom_from_fit_deterministic_and_kw_forwarding(monkeypatch):
    install_stub_kinetics(monkeypatch)
    T_grid = np.linspace(400.0, 500.0, 21)
    fit = FakeFitResult()
    r1 = fom_from_fit(fit, "m", T_grid, feed=None, n_samples=64, rng=11)
    r2 = fom_from_fit(fit, "m", T_grid, feed=None, n_samples=64, rng=11)
    assert r1.value == r2.value and r1.sd == r2.sd
    # FOM kwargs are forwarded (windowed integral < full integral)
    rw = fom_from_fit(
        fit, "m", T_grid, feed=None, n_samples=64, rng=11, T_min=440.0, T_max=460.0
    )
    assert rw.value < r1.value
    # nonlinear FOM through the same path
    rt = fom_from_fit(fit, "m", T_grid, feed=None, name="t50", n_samples=64, rng=11)
    assert 400.0 < rt.value < 450.0  # rising edge of the bump
    assert rt.sd >= 0.0


def test_fom_from_fit_validation(monkeypatch):
    install_stub_kinetics(monkeypatch)
    fit = FakeFitResult()
    T_grid = np.linspace(400.0, 500.0, 11)
    with pytest.raises(ValueError, match="n_samples"):
        fom_from_fit(fit, "m", T_grid, feed=None, n_samples=1)
    with pytest.raises(ValueError, match="unknown FOM"):
        fom_from_fit(fit, "m", T_grid, feed=None, name="nope")
