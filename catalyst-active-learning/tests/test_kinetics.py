"""Tests for zeoal.kinetics (fast: small grids, few multistarts, fixed seeds)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from zeoal.data import TemperatureTrace
from zeoal.kinetics import (
    KINETIC_MODELS,
    ArrheniusFirstOrder,
    EyringFirstOrder,
    FeedConditions,
    KineticFitResult,
    LHHWKinetics,
    SigmoidLightOff,
    fit_all_models,
    fit_trace,
    physical_params,
    predict_trace,
    select_best_model,
    trace_band,
)

C0 = 0.2
FEED = FeedConditions(y_co2=0.2, y_h2=0.8, y_inert=0.0, pressure_bar=1.0, total_concentration=C0)
NOISE = 0.002  # 1% of c0

# true internal parameters of the arrhenius ground truth
TRUE_ARR = np.array([np.log(2.0), 8.0e4, np.log(0.05), 1.2e5])
# true parameters of the sigmoid ground truth
TRUE_SIG = np.array([0.85, 540.0, 22.0, 0.10, 730.0, 45.0])


def _noisy_trace(model, true_params, n_T=28, noise=NOISE, seed=3):
    T = np.linspace(423.0, 773.0, n_T)
    pred = model.predict(T, true_params, FEED)
    gen = np.random.default_rng(seed)
    chans = {k: np.clip(v + gen.normal(0.0, noise, size=T.size), 0.0, None) for k, v in pred.items()}
    return TemperatureTrace(T, chans["co2"], chans["ch4"], chans["co"])


@pytest.fixture(scope="module")
def arr_trace():
    return _noisy_trace(ArrheniusFirstOrder(), TRUE_ARR)


@pytest.fixture(scope="module")
def arr_fit(arr_trace):
    return fit_trace(arr_trace, model="arrhenius", feed=FEED, n_multistart=3, rng=0)


@pytest.fixture(scope="module")
def sig_trace():
    return _noisy_trace(SigmoidLightOff(), TRUE_SIG, seed=11)


@pytest.fixture(scope="module")
def sig_fit(sig_trace):
    return fit_trace(sig_trace, model="sigmoid", feed=FEED, n_multistart=3, rng=1)


# ---------------------------------------------------------------------------
# registry / basic API
# ---------------------------------------------------------------------------
def test_registry_names():
    assert set(KINETIC_MODELS) == {"arrhenius", "eyring", "lhhw", "sigmoid"}
    for name, cls in KINETIC_MODELS.items():
        assert cls.name == name
        assert len(cls().param_names) >= 4


def test_unknown_model_raises(arr_trace):
    with pytest.raises(ValueError, match="unknown kinetic model"):
        fit_trace(arr_trace, model="nope", feed=FEED)


def test_feed_validation():
    with pytest.raises(ValueError):
        FeedConditions(y_co2=-0.1)
    with pytest.raises(ValueError):
        FeedConditions(pressure_bar=0.0)
    with pytest.raises(ValueError):
        FeedConditions(total_concentration=-1.0)
    y = FeedConditions(y_co2=1.0, y_h2=4.0).normalized_fractions()
    assert np.isclose(sum(y), 1.0) and np.isclose(y[0], 0.2)


def test_predict_shapes_and_keys(arr_trace):
    T = arr_trace.temperature_K
    for name, cls in KINETIC_MODELS.items():
        model = cls()
        p = model.initial_guesses(arr_trace, 1, rng=0)[0]
        pred = model.predict(T, p, FEED)
        assert set(pred) == {"co2", "ch4", "co"}
        for v in pred.values():
            assert v.shape == T.shape
            assert np.all(np.isfinite(v))
            assert np.all(v >= -1e-12)


# ---------------------------------------------------------------------------
# arrhenius round trip
# ---------------------------------------------------------------------------
def test_arrhenius_roundtrip_curve(arr_trace, arr_fit):
    assert arr_fit.success
    assert arr_fit.r2 > 0.98
    pred = predict_trace("arrhenius", arr_trace.temperature_K, arr_fit.params, FEED,
                         T_ref=arr_fit.T_ref)
    rmse = np.sqrt(np.mean((pred.c_ch4 - arr_trace.c_ch4) ** 2))
    assert rmse < 2.5 * NOISE


def test_arrhenius_recovers_params(arr_fit):
    # true parameters recovered within ~3 sigma (small floor guards tiny sd)
    floor = np.array([0.05, 2.0e3, 0.1, 5.0e3])
    err = np.abs(arr_fit.params - TRUE_ARR)
    tol = 3.0 * arr_fit.param_sd + floor
    assert np.all(err < tol), f"err={err}, tol={tol}"


def test_fit_without_total_concentration(arr_trace):
    feed = FeedConditions(y_co2=0.2, y_h2=0.8)  # total_concentration=None
    res = fit_trace(arr_trace, model="arrhenius", feed=feed, n_multistart=2, rng=0)
    assert res.r2 > 0.95
    # feed=None entirely also works (default stoichiometric feed)
    res2 = fit_trace(arr_trace, model="arrhenius", n_multistart=2, rng=0)
    assert res2.r2 > 0.95


# ---------------------------------------------------------------------------
# sigmoid round trip
# ---------------------------------------------------------------------------
def test_sigmoid_roundtrip(sig_trace, sig_fit):
    assert sig_fit.r2 > 0.98
    pred = predict_trace("sigmoid", sig_trace.temperature_K, sig_fit.params, FEED,
                         T_ref=sig_fit.T_ref)
    rmse = np.sqrt(np.mean((pred.c_ch4 - sig_trace.c_ch4) ** 2))
    assert rmse < 2.5 * NOISE
    # A and T50 recovered within ~3 sigma (+ floors)
    for i, floor in [(0, 0.05), (1, 5.0), (2, 5.0)]:
        assert abs(sig_fit.params[i] - TRUE_SIG[i]) < 3.0 * sig_fit.param_sd[i] + floor


# ---------------------------------------------------------------------------
# eyring + lhhw smoke fits
# ---------------------------------------------------------------------------
def test_eyring_smoke(arr_trace):
    res = fit_trace(arr_trace, model="eyring", feed=FEED, n_multistart=3, rng=0)
    assert res.r2 > 0.8
    assert np.all(np.isfinite(res.cov))
    assert np.all(np.isfinite(res.params))
    assert res.param_names == ["log_km_ref", "dH_m", "log_kr_ref", "dH_r"]


def test_lhhw_smoke():
    trace = _noisy_trace(ArrheniusFirstOrder(), TRUE_ARR, n_T=13, noise=0.003, seed=5)
    res = fit_trace(trace, model="lhhw", feed=FEED, n_multistart=2, rng=0)
    assert res.r2 > 0.8
    assert np.all(np.isfinite(res.cov))
    assert np.all(np.isfinite(res.params))
    assert res.n_multistart == 2
    # forward prediction is physical
    pred = KINETIC_MODELS["lhhw"](T_ref=res.T_ref).predict(trace.temperature_K, res.params, FEED)
    assert np.all(pred["ch4"] >= -1e-12) and np.all(pred["ch4"] <= C0 + 1e-9)


def test_lhhw_release_adsorption_params():
    model = LHHWKinetics(release_adsorption=True)
    assert model.param_names[-2:] == ["log_K_H2", "log_K_mix"]
    lo, hi = model.default_bounds(_noisy_trace(ArrheniusFirstOrder(), TRUE_ARR, n_T=5))
    assert lo.size == hi.size == 8


# ---------------------------------------------------------------------------
# model selection
# ---------------------------------------------------------------------------
def test_select_best_model(arr_trace, arr_fit, sig_fit_on_arr=None):
    sig_on_arr = fit_trace(arr_trace, model="sigmoid", feed=FEED, n_multistart=3, rng=2)
    results = {"arrhenius": arr_fit, "sigmoid": sig_on_arr}
    best = select_best_model(results, criterion="aic")
    assert results[best].aic == min(r.aic for r in results.values())
    # a deliberately terrible fit is never selected
    bad = KineticFitResult(
        model_name="bad", param_names=["a"], params=np.array([0.0]), cov=np.eye(1),
        param_sd=np.array([1.0]), T_ref=600.0, rss=1e6, dof=10, sigma2=1e5,
        aic=1e9, bic=1e9, r2=-5.0, success=True, message="fake", n_multistart=1,
        lower=np.array([-1.0]), upper=np.array([1.0]),
    )
    results["bad"] = bad
    assert select_best_model(results, criterion="aic") != "bad"
    assert select_best_model(results, criterion="bic") != "bad"
    assert select_best_model(results, criterion="r2") != "bad"
    with pytest.raises(ValueError):
        select_best_model(results, criterion="nope")
    with pytest.raises(ValueError):
        select_best_model({})


def test_fit_all_models(arr_trace):
    results = fit_all_models(arr_trace, models=["arrhenius", "sigmoid"], feed=FEED,
                             n_multistart=2, rng=0)
    assert set(results) == {"arrhenius", "sigmoid"}
    for res in results.values():
        assert isinstance(res, KineticFitResult)
        assert np.isfinite(res.aic)


# ---------------------------------------------------------------------------
# covariance / sampling / physical params
# ---------------------------------------------------------------------------
def test_cov_positive_semidefinite(arr_fit):
    cov = 0.5 * (arr_fit.cov + arr_fit.cov.T)
    evals = np.linalg.eigvalsh(cov)
    assert np.all(evals >= -1e-10 * max(evals.max(), 1.0))
    assert np.allclose(arr_fit.param_sd, np.sqrt(np.clip(np.diag(cov), 0, None)))


def test_sample_params_bounds_and_cov(arr_fit):
    draws = arr_fit.sample_params(4000, rng=5)
    assert draws.shape == (4000, arr_fit.n_params)
    assert np.all(draws >= arr_fit.lower - 1e-12)
    assert np.all(draws <= arr_fit.upper + 1e-12)
    # sample mean/std roughly match the reported params/sd (bounds not binding)
    sd = arr_fit.param_sd
    assert np.all(np.abs(draws.mean(axis=0) - arr_fit.params) < 0.2 * sd + 1e-12)
    ratio = draws.std(axis=0) / np.where(sd > 0, sd, 1.0)
    assert np.all((ratio > 0.7) & (ratio < 1.35))


def test_sample_params_deterministic(arr_fit):
    a = arr_fit.sample_params(50, rng=7)
    b = arr_fit.sample_params(50, rng=7)
    assert np.array_equal(a, b)


def test_fit_deterministic(arr_trace):
    r1 = fit_trace(arr_trace, model="arrhenius", feed=FEED, n_multistart=2, rng=42)
    r2 = fit_trace(arr_trace, model="arrhenius", feed=FEED, n_multistart=2, rng=42)
    assert np.array_equal(r1.params, r2.params)
    assert r1.aic == r2.aic and r1.rss == r2.rss


def test_physical_params(arr_fit):
    pp = physical_params(arr_fit)
    assert set(pp) == {"km_ref", "Ea_m", "kr_ref", "Ea_r"}
    km, km_sd = pp["km_ref"]
    i = arr_fit.param_names.index("log_km_ref")
    assert np.isclose(km, np.exp(arr_fit.params[i]))
    assert np.isclose(km_sd, km * arr_fit.param_sd[i])  # delta method
    # non-log params pass through
    j = arr_fit.param_names.index("Ea_m")
    assert pp["Ea_m"] == (arr_fit.params[j], arr_fit.param_sd[j])


def test_to_from_dict_roundtrip(arr_fit):
    d = arr_fit.to_dict()
    json.dumps(d)  # JSON-serializable
    back = KineticFitResult.from_dict(d)
    assert back.model_name == arr_fit.model_name
    assert back.param_names == arr_fit.param_names
    assert np.allclose(back.params, arr_fit.params)
    assert np.allclose(back.cov, arr_fit.cov)
    assert back.dof == arr_fit.dof and back.success == arr_fit.success
    # sampling behaves identically after the round trip
    assert np.array_equal(back.sample_params(20, rng=3), arr_fit.sample_params(20, rng=3))


# ---------------------------------------------------------------------------
# helpers: predict_trace, trace_band, weights
# ---------------------------------------------------------------------------
def test_predict_trace_returns_trace(arr_fit):
    T = np.linspace(450.0, 750.0, 12)
    tr = predict_trace("arrhenius", T, arr_fit.params, FEED, T_ref=arr_fit.T_ref,
                       concentration_unit="ppm")
    assert isinstance(tr, TemperatureTrace)
    assert tr.concentration_unit == "ppm"
    assert tr.n_points == 12
    assert np.allclose(tr.c_co2 + tr.c_ch4 + tr.c_co, C0)


def test_trace_band(arr_trace, arr_fit):
    T = arr_trace.temperature_K
    band = trace_band("arrhenius", T, arr_fit, FEED, n_samples=80, rng=0)
    assert set(band) == {"co2", "ch4", "co"}
    point = predict_trace("arrhenius", T, arr_fit.params, FEED, T_ref=arr_fit.T_ref)
    for ch, (mean, lo, hi) in band.items():
        assert mean.shape == lo.shape == hi.shape == T.shape
        assert np.all(np.isfinite(mean)) and np.all(lo <= hi + 1e-12)
    assert np.all(np.abs(band["ch4"][0] - point.c_ch4) < 0.05 * C0)
    # deterministic given the seed
    band2 = trace_band("arrhenius", T, arr_fit, FEED, n_samples=80, rng=0)
    assert np.array_equal(band["ch4"][0], band2["ch4"][0])


def test_channel_weights(arr_trace):
    res = fit_trace(arr_trace, model="arrhenius", feed=FEED, n_multistart=1, rng=0,
                    weights={"ch4": 2.0})
    assert res.r2 > 0.9
    with pytest.raises(ValueError, match="unknown weight channel"):
        fit_trace(arr_trace, model="arrhenius", feed=FEED, n_multistart=1, rng=0,
                  weights={"methane": 1.0})
    with pytest.raises(ValueError):
        fit_trace(arr_trace, model="arrhenius", feed=FEED, n_multistart=1, rng=0,
                  weights=[1.0, -1.0, 1.0])
