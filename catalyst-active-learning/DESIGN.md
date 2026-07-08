# Design: `zeoal` — Active learning for zeolite-supported multimetallic CO2-methanation catalysts

Subproject of the YARP repository, self-contained under `catalyst-active-learning/`.

## Purpose

An incoming dataset contains, per catalyst sample:

1. a **selected** metal composition (continuous metal fractions chosen by the experimenter / optimizer),
2. an **actual measured** metal composition (e.g. from ICP-OES; deposition never exactly matches the target),
3. a **temperature — concentration curve**: outlet CO2, CH4 and CO concentrations vs temperature
   (temperature-programmed reaction / light-off experiment).

The dataset was generated with guidance from the Doyle-lab EDBO code. This package provides, in
generic/data-agnostic form:

- kinetic fitting of the temperature–concentration curves (Arrhenius, Eyring, LHHW, empirical),
- a figure of merit (FOM), baseline: **total CH4–temperature integral**, plus an alternatives menu,
- three chained prediction "model trains" (A/B/C, below) with full uncertainty propagation,
- batch active learning (default q=6 suggestions) over the continuous metal-fraction simplex,
- switching between EDBO and other surrogate models everywhere,
- optional Schoenebeck-style noisy-data augmentation,
- a *separate* retrospective campaign-evaluation process to benchmark the model trains against a
  previous active-learning campaign,
- a Jupyter notebook driving all of the above from a single configuration cell.

## Directory layout

```
catalyst-active-learning/
├── DESIGN.md                    # this file
├── README.md                    # install + usage + data format
├── LITERATURE.md                # literature survey: traces, kinetics, FOMs, Schoenebeck, batch BO
├── requirements.txt
├── pyproject.toml               # minimal, name="zeoal"
├── zeoal/
│   ├── __init__.py              # re-export key API, __version__
│   ├── data.py                  # data model + I/O                        [WRITTEN — do not modify]
│   ├── synthetic.py             # realistic synthetic campaign generator
│   ├── kinetics.py              # kinetic models + fitting
│   ├── equilibrium.py           # CO2-methanation thermodynamics          [WRITTEN — do not modify]
│   ├── fom.py                   # figures of merit
│   ├── augment.py               # Schoenebeck noisy-data augmentation
│   ├── models.py                # surrogate registry: GP / RF / EDBO adapter
│   ├── trains.py                # model trains A, B, C with MC uncertainty propagation
│   ├── active_learning.py       # batch acquisition on the simplex
│   ├── campaign.py              # retrospective campaign replay + metrics
│   └── plotting.py              # standard figures
├── scripts/
│   └── evaluate_campaign.py     # CLI — the "separate process" for campaign evaluation
├── notebooks/
│   └── catalyst_active_learning.ipynb
└── tests/
    ├── test_data.py
    ├── test_kinetics.py
    ├── test_fom.py
    ├── test_augment.py
    ├── test_models.py
    ├── test_trains.py
    ├── test_active_learning.py
    ├── test_synthetic.py
    └── test_campaign.py
```

## Global conventions (binding for every module)

- Dependencies: numpy, scipy, pandas, scikit-learn, matplotlib only. `edbo`/`edboplus`/`torch` are
  **optional** imports guarded inside functions; the package must fully work without them.
- Temperatures **in kelvin** internally. Loaders accept `"C"`/`"K"` and convert.
- Compositions are numpy arrays of shape `(n_metals,)` or `(n, n_metals)`, non-negative,
  **summing to 1** (the simplex). `zeoal.data.project_to_simplex` enforces this.
- Every stochastic routine takes `rng: np.random.Generator | int | None` and passes it down
  (`zeoal.data.as_generator` normalizes). No global seeding.
- Uncertainties are 1-sigma standard deviations unless a name says otherwise.
- Docstrings: numpy style. Reference equations/literature by short tag (e.g. "Koschany 2016",
  see LITERATURE.md).
- Errors: raise `ValueError` with actionable messages; never silently clip data (clipping is OK when
  documented, e.g. simplex projection).
- All fitting/training must be deterministic given `rng`.

## Module specifications

### `zeoal/data.py` — WRITTEN, treat as frozen API

Key contents (see file for exact signatures):

- `KELVIN_OFFSET`, `as_generator(rng)`, `project_to_simplex(x)`, `dirichlet_sample(...)`
- `CompositionSpace(metals: list[str], min_fraction, max_fraction)` — names + box constraints on the
  simplex; `.contains`, `.random(n, rng)`, `.grid(resolution)`.
- `TemperatureTrace(temperature_K, c_co2, c_ch4, c_co, concentration_unit)` — sorted, validated;
  `.as_frame()`.
- `CatalystSample(sample_id, selected_composition, measured_composition, trace, round_index, metadata)`
  — `measured_composition` may be None (not yet characterized).
- `CatalystDataset(space, samples)` — `.selected_X()`, `.measured_X()`, `.rounds()`,
  `.filter_rounds(...)`, `.to_frame()`, `.from_frames(...)`, `.save_json/.load_json`,
  `.save_csv/.load_csv` (two tidy CSVs: samples + traces).

### `zeoal/equilibrium.py` — WRITTEN, treat as frozen API

- `delta_g_methanation(T_K)`: ΔG°(T) for CO2 + 4 H2 ⇌ CH4 + 2 H2O(g), van 't Hoff style with ΔCp
  correction fitted to standard thermochemistry.
- `keq_methanation(T_K)`, `keq_rwgs(T_K)`.
- `equilibrium_conversion(T_K, y_co2, y_h2, pressure_bar)` — solves equilibrium CO2 conversion for a
  CO2/H2/inert feed (methanation only; documented approximation), vectorized, by bisection.

### `zeoal/kinetics.py`

Purpose: fit a `TemperatureTrace` to mechanistic or empirical models and return parameters **with
covariance** so uncertainty can be propagated.

Design:

```python
KineticFitResult:  # dataclass
    model_name: str
    param_names: list[str]
    params: np.ndarray                # best fit, internal (scaled) parameterization
    cov: np.ndarray                   # covariance of params (same parameterization)
    param_sd: np.ndarray
    T_ref: float
    rss, dof, sigma2: float           # residual stats (all channels stacked)
    aic, bic, r2: float
    success: bool
    message: str
    n_multistart: int
    def sample_params(n, rng) -> (n, p) array      # MVN truncated to bounds (resample+clip)
    def to_dict() / from_dict()

KineticModel (ABC):
    name: str
    param_names: list[str]            # human-readable, physical (e.g. k_ref, Ea, ...)
    def default_bounds(trace) -> (lo, hi)
    def initial_guesses(trace, n, rng) -> (n, p)   # heuristic + jittered multistart
    def predict(T_K, params, feed) -> dict with keys "co2","ch4","co" (same conc unit as data)
    def fit(trace, feed, n_multistart=8, weights=None, rng=None) -> KineticFitResult
        # shared implementation in the ABC: scipy.optimize.least_squares (TRF, bounded),
        # residuals stacked over the three channels, optional per-channel weights,
        # covariance from J^T J pseudo-inverse * sigma2 (document rank-deficiency handling),
        # multistart keeps best RSS; params fitted in a scaled space (log-scale for rate
        # constants / prefactors) — document the transform, expose physical values via
        # `physical_params(result)`.

FeedConditions:  # dataclass — needed to convert conversion <-> concentrations
    y_co2, y_h2, y_inert: float       # inlet mole fractions
    pressure_bar: float = 1.0
    total_concentration: float = ...  # inlet CO2 concentration in the trace's unit, used to scale
```

Rate-constant parameterization (all mechanistic models): `k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))`
with `T_ref` = mid-range of the data — decorrelates prefactor and activation energy (fit `log k_ref`).

Models to implement:

1. `ArrheniusFirstOrder` (name `"arrhenius"`): two parallel pseudo-first-order channels from CO2 —
   methanation (m) and RWGS (r), integral PFR at constant space time τ (absorbed into k_ref):
   `X_tot(T) = X_eq_cap(T) * (1 - exp(-(k_m + k_r)))` with per-channel split
   `Y_ch4 = X_tot * k_m/(k_m+k_r)`, `Y_co = X_tot * k_r/(k_m+k_r)`; the equilibrium cap uses
   `equilibrium_conversion`. Params: `log_km_ref, Ea_m, log_kr_ref, Ea_r`.
   Concentrations: `c_co2 = c0*(1-X_tot)`, `c_ch4 = c0*Y_ch4`, `c_co = c0*Y_co` (c0 from feed).
   (Outlet mole-fraction renormalization from mole-number change is second-order; we fit
   concentrations proportional to yields — document this.)
2. `EyringFirstOrder` (name `"eyring"`): identical structure, but
   `k(T) = (kB*T/h) * exp(dS/R) * exp(-dH/(R*T))` reparameterized as
   `k(T) = k_ref * (T/T_ref) * exp(-dH/R * (1/T - 1/T_ref))`; params
   `log_km_ref, dH_m, log_kr_ref, dH_r` (+ document how dS is recovered).
3. `LHHWKinetics` (name `"lhhw"`): Koschany-2016-inspired rate law
   `r = k(T) * p_H2^0.5 * p_CO2^0.5 / DEN^2 * (1 - Q/Keq)`,
   `DEN = 1 + K_OH * p_H2O / p_H2^0.5 + K_H2 * p_H2^0.5 + K_mix * p_CO2^0.5`,
   adsorption constants with van 't Hoff T-dependence; integrate an isothermal ideal PFR in
   normalized space time with `scipy.integrate.solve_ivp` per temperature point (vectorize over T
   via a loop; keep it robust, stiff-safe `LSODA` or `Radau` fallback). CO via a first-order RWGS
   channel on top (same as model 1) so the three channels are all predicted. Free params (fit):
   `log_k_ref, Ea, log_K_OH_ref, dH_OH, log_kr_ref, Ea_r`; fix the weakly identified `K_H2, K_mix`
   at Koschany values by default with an option to release them. Document identifiability.
4. `SigmoidLightOff` (name `"sigmoid"`): empirical, robust fallback:
   `Y_ch4(T) = A * expit((T - T50)/w) * f_eq(T)` where `f_eq` is the normalized equilibrium decline;
   CO from a high-T sigmoid share; params `A, T50, w, A_co, T50_co, w_co`. Also yields interpretable
   `T50` directly.

Registry + helpers:

- `KINETIC_MODELS: dict[str, type[KineticModel]]`
- `fit_trace(trace, model="arrhenius", feed=..., ...) -> KineticFitResult`
- `fit_all_models(trace, models=None, ...) -> dict[str, KineticFitResult]`
- `select_best_model(results, criterion="aic") -> str`
- `predict_trace(model_name, T_K, params, feed) -> TemperatureTrace`
- `trace_band(model_name, T_K, fit_result, feed, n_samples, rng) -> (mean, lo, hi) per channel` (MC).

Tests must include: round-trip (generate from known params + noise → fit → recover within CI),
model selection sanity, covariance positive-semidefinite, sample_params respects bounds.

### `zeoal/fom.py`

```python
FOMResult: value: float, sd: float | None, name: str, details: dict

fom_from_trace(trace, name="ch4_temperature_integral", noise_sd=None, **kw) -> FOMResult
fom_from_fit(fit_result, model_name, T_grid, feed, name=..., n_samples=500, rng=None) -> FOMResult
    # MC: sample kinetic params -> predict c_ch4(T) -> integrate -> mean/sd
FOM_REGISTRY: dict[str, callable]
```

FOMs (each `f(T_K, c_ch4, c_co2, c_co, **kw) -> float`):

- `ch4_temperature_integral` — **baseline, the requested new FOM**: `∫ c_ch4 dT` (trapezoid) over an
  optional `[T_min, T_max]` window. Units: conc·K.
- `weighted_ch4_integral` — `∫ c_ch4 * w(T) dT`, weights: `"one_over_T"`, `"low_T_boltzmann"`
  (`exp(-(T-T0)/tau_w)`) — rewards low-temperature activity.
- `selectivity_weighted_integral` — `∫ c_ch4 * S_ch4(T) dT`, `S = c_ch4/(c_ch4 + c_co)` (safe/0).
- `co_penalized_integral` — `∫ (c_ch4 - lambda_co * c_co) dT`.
- `t50` — temperature where CH4 reaches 50% of its max (interpolated; negate for maximization via
  `maximize=False` flag in registry metadata `FOM_INFO`).
- `t10`, `t90`, `max_ch4`, `ch4_at_T(T_eval)`, `peak_temperature`.

`noise_sd` (per-point concentration noise) → analytic sd for linear integrals (trapezoid is linear:
`sd^2 = Σ (w_i * sd_i)^2` with trapezoid weights), MC for nonlinear FOMs (t50 etc.).

### `zeoal/augment.py` — Schoenebeck noisy-data augmentation

Based on the Schoenebeck-group approach of augmenting noisy experimental training data with
noise-perturbed copies (see LITERATURE.md for exact citation; docstring must cite it).

```python
NoiseAugmentConfig: n_copies:int=10, target_noise:"float|array|None", input_noise: float|None=None,
                    estimate_from:"residuals|replicates|value"='value', keep_original:bool=True
augment_training_data(X, y, y_sd=None, config=..., rng=None) -> (X_aug, y_aug, sd_aug)
estimate_noise_from_replicates(dataset, fom_name) -> float   # pooled sd over replicate groups
```

Rules: augmentation is applied to **training data only** (callers: `models.SurrogateModel.fit`
via config flag, and `trains.*`); augmented copies inherit the original's provenance index so CV
splits can group them (`groups` output).

### `zeoal/models.py` — surrogate registry (the EDBO ⇄ others switch)

```python
SurrogateModel (ABC):
    def fit(X, y, y_sd=None) -> self
    def predict(X) -> (mean, sd)
    def sample_y(X, n_samples, rng) -> (n_samples, n_points)
    def loo_metrics() -> dict            # leave-one-out RMSE/NLL/coverage on training data
    hyperparams_: dict                   # fitted kernel etc, for reporting

create_model(name: str, **kwargs) -> SurrogateModel
MODEL_REGISTRY = {"gp": GPModel, "rf": RandomForestModel, "edbo": EDBOModel, "linear": BayesianLinearModel}
```

- `GPModel`: sklearn `GaussianProcessRegressor`. **Hyperparameter policy (critical requirement)**:
  - candidate kernels: `C * Matern(nu=2.5)`, `C * Matern(nu=1.5)`, `C * RBF`, each iso **and** ARD,
    each `+ WhiteKernel` (noise floor bounds tied to `y_sd` if given);
  - lengthscale init = median pairwise distance of X; bounds `[1e-3, 1e3] × median` — i.e. *set from
    the data*, not hard-coded absolutes;
  - y standardized internally (`normalize_y`); heteroscedastic noise via `alpha = y_sd**2` when given;
  - `n_restarts_optimizer` scales with dim (e.g. `max(5, 2*d)`);
  - kernel chosen by LOO log predictive density (analytic LOO from Cholesky, or
    sklearn's `log_marginal_likelihood` + BIC penalty — pick one, document);
  - option `kernel="auto"` (default) or explicit (`"matern52"`, ...), `ard=True|False|"auto"`.
  - optional `input_transform="ilr"` for compositional inputs (isometric log-ratio with zero-handling
    via multiplicative replacement; document).
- `RandomForestModel`: sklearn RF; predictive sd from std over trees (document bias); supports
  `sample_y` by sampling trees.
- `BayesianLinearModel`: sklearn `BayesianRidge` on polynomial features (degree 2 default) — cheap
  baseline with analytic predictive sd.
- `EDBOModel`: adapter for Doyle-lab EDBO. Import guarded:
  `try: import edbo ... except ImportError: raise ImportError(<install hint>)`. Maps our
  `fit/predict/sample_y` onto edbo's `BO` object API. Because edbo pins old deps, ALSO implement
  `EDBOLikeGPModel` (name `"edbo_like"`, always available): reproduces EDBO's documented modeling
  choices with our stack — Matern 5/2 kernel, ARD, standardized y, GP noise hyperparameter — so
  users without edbo installed can "switch to EDBO(-like)" and get the same modeling behavior.
  `create_model("edbo")` falls back to `EDBOLikeGPModel` with a loud warning if edbo is missing
  (behavior flag `strict=True` to raise instead).
- `MultiOutputModel`: wraps a `create_model` factory into independent per-output models with the
  same API (used for composition→composition and composition→kinetic-params stages); `predict`
  returns `(mean (n,k), sd (n,k))`, `sample_y -> (n_samples, n, k)`.
- All models accept `augment: NoiseAugmentConfig | None` in `fit`.

### `zeoal/trains.py` — the three prediction trains

```python
TrainPrediction: mean, sd: (n,) arrays; samples: (n_samples, n); per_stage: dict (diagnostics)

ModelTrain (ABC):
    name: str
    def fit(dataset: CatalystDataset, fom="ch4_temperature_integral", **cfg) -> self
    def predict_fom(X_selected, n_samples=1000, rng=None) -> TrainPrediction
    def cross_validate(dataset, k=5 or LOO, rng) -> dict of metrics (rmse, mae, nll, coverage95, spearman)
create_train(name, model_backend="gp", **kw); TRAIN_REGISTRY = {"A": TrainA, "B": TrainB, "C": TrainC}
```

- `TrainA` (`"A"`): selected composition → surrogate → FOM. Single stage.
- `TrainB` (`"B"`): stage 1: selected → measured composition (MultiOutputModel; targets are the
  measured fractions; predictions projected to simplex); stage 2: measured → FOM (surrogate trained
  on measured compositions). Prediction: MC — sample stage-1 outputs, project to simplex, push
  each sample through stage-2 `sample_y` (one sample per propagated point), aggregate.
- `TrainC` (`"C"`): stage 1: selected → measured (as B); stage 2: measured → kinetic parameters
  (MultiOutputModel trained on per-sample `KineticFitResult.params`, with per-sample fit sd as
  heteroscedastic noise `y_sd`); stage 3 (deterministic physics): params → kinetic model → predicted
  temperature trace on the dataset's common T grid → FOM integral. MC through all stages; kinetic-fit
  parameter uncertainty enters via stage-2 noise. Config: `kinetic_model="arrhenius"` etc.
  Cache per-sample fits on the train instance (`self.fits_`); samples whose kinetic fit fails are
  dropped from stage 2 with a warning (recorded in `fit_report_`).
- Uncertainty propagation is **always by Monte Carlo sampling** through stages (documented); sd of
  the FOM = sd over samples; also report decomposition: predictive variance per stage (law of total
  variance estimate) in `per_stage`.
- Every stage takes the surrogate backend from `model_backend` (so EDBO/GP/RF switching applies to
  the whole train), overridable per stage (`stage_backends={"measured": "gp", ...}`).
- `augment` config forwarded to all stages.

### `zeoal/active_learning.py`

```python
Suggestion: composition (array), acquisition_value, pred_mean, pred_sd; DataFrame conversion.
suggest_batch(train, space, dataset, q=6, acquisition="ei", strategy="kriging_believer",
              n_candidates=4000, candidate_method="mixed", xi=0.01, kappa=2.0,
              min_distance=None, rng=None) -> list[Suggestion]
```

- Candidate generation on the simplex (`candidate_method`): `"dirichlet"` (uniform, alpha=1),
  `"mixed"` = uniform + concentrated Dirichlet around current top-m points + grid corners/edges,
  respecting `space.min_fraction/max_fraction` (rejection); dedupe vs existing data within tol.
- Acquisition from **MC samples** (works for non-Gaussian train predictions):
  `ei` (E[max(s - best, 0)] over samples), `ucb` (mean + kappa*sd), `pi`, `thompson` (one draw),
  `mean` (exploit), `sd` (explore). `best` = current best observed FOM in dataset.
- Batch strategies for q points: `"kriging_believer"` (greedy: pick argmax, add fantasy = pred mean,
  refit cheap stage-2 or use in-sample conditioning — document: we refit the train's final stage on
  augmented data), `"constant_liar"` (lie = min/mean/max of observed y), `"local_penalization"`
  (distance-based penalty with lengthscale from train's GP if available else median heuristic),
  `"thompson"` (q independent posterior draws, argmax each). Enforce `min_distance` between batch
  members (default: 0.5 * median nearest-neighbor distance of data).
- Returns compositions on the simplex rounded to a configurable precision (default 3 decimals,
  re-projected).

### `zeoal/synthetic.py`

Ground-truth generator for demos/tests AND for validating campaign evaluation:

```python
SyntheticGroundTruth(space, rng, ...):
    def measured_from_selected(Xsel, rng) -> Xmeas       # per-metal deposition bias + Dirichlet-ish noise
    def kinetic_params(Xmeas) -> params                  # smooth nonlinear map w/ synergy (e.g. Ni-Ru), one optimum inside simplex
    def trace(Xmeas, T_grid, noise_sd, rng) -> TemperatureTrace   # via kinetics.ArrheniusFirstOrder forward model + noise
    def true_fom(X, fom_name) -> float                   # noiseless, via dense T grid
make_synthetic_dataset(space=None (default: Ni,Ru,Co on SSZ-13 flavor text), n_rounds, samples_per_round,
                       selection="edbo_like"|"random"|"lhs", T_range=(423,773), n_T=36, noise_sd=...,
                       rng) -> (CatalystDataset, SyntheticGroundTruth)
    # selection="edbo_like": rounds after the first are chosen by running our own TrainA-GP+EI on the
    # accumulated synthetic data — emulates an EDBO-guided historical campaign with round_index set.
```

Traces must look like real methanation light-off data: CH4 sigmoidal rise from ~200 °C, peak
400–450 °C, equilibrium/RWGS decline above; CO appearing at high T. (Justify in LITERATURE.md.)

### `zeoal/campaign.py` + `scripts/evaluate_campaign.py`

Retrospective evaluation of a *previous* campaign (the "separate process"):

```python
replay_campaign(dataset, trains: dict[name->factory kwargs], fom=..., start_round=0,
                acquisition="ei", q=6, n_samples=..., rng=None) -> CampaignReport
CampaignReport:
    per_round: DataFrame   # train, round, n_train, rmse, mae, nll, coverage95, spearman on next round
    ranking: DataFrame     # would-have-chosen analysis: for each round, acquisition rank of the
                           # actually-best next-round sample; top-q hit rate; simulated regret
    summary() -> DataFrame; to_markdown(path); plots via plotting module
simulate_campaign(ground_truth, space, train_cfgs, n_rounds, q, seed_size, rng) -> DataFrame
    # forward simulation on synthetic truth: best-FOM trajectory per train (regret curves)
```

CLI `scripts/evaluate_campaign.py`:
`python evaluate_campaign.py --data campaign.json --trains A B C --backend gp --fom ch4_temperature_integral --q 6 --augment/--no-augment --simulate-synthetic --outdir results/` —
loads dataset (JSON/CSV), runs `replay_campaign` (and optional `simulate_campaign` when
`--simulate-synthetic`), writes `report.md`, `per_round.csv`, `ranking.csv`, figures (PNG).
Must run end-to-end on synthetic data via `--demo`.

### `zeoal/plotting.py`

Matplotlib helpers, all returning `(fig, axes)`, no plt.show():
`plot_traces(dataset, color_by="round")`, `plot_fit(trace, fit_result, feed, band=True)`,
`plot_model_comparison(fit_results)` (AIC/BIC bars), `plot_parity(y_true, y_pred, y_sd)`,
`plot_calibration(y_true, mean, sd)` (reliability of predictive intervals),
`plot_fom_landscape(train, space, resolution)` (ternary heatmap for 3 metals via barycentric
projection; pairwise-slices grid for >3), `plot_suggestions(space, dataset, suggestions)`,
`plot_campaign_report(report)`, `plot_regret(simulate_df)`.

### Notebook `notebooks/catalyst_active_learning.ipynb`

Single CONFIG cell at top:

```python
CONFIG = dict(
    data_path=None,                  # None -> synthetic demo campaign
    model_backend="gp",              # "gp" | "edbo" | "edbo_like" | "rf" | "linear"
    train="C",                       # "A" | "B" | "C"
    kinetic_model="arrhenius",       # "arrhenius" | "eyring" | "lhhw" | "sigmoid"
    fom="ch4_temperature_integral",  # any FOM_REGISTRY key
    augment=False, augment_copies=10,
    q=6, acquisition="ei", batch_strategy="kriging_believer",
    seed=7,
)
```

Sections: 1 setup; 2 load data (or synthetic) + trace gallery; 3 kinetic fitting (all models on an
example trace, AIC comparison, uncertainty bands; batch-fit all samples); 4 FOM analysis (baseline
integral + menu comparison table, uncertainties); 5 model trains (fit A/B/C, LOO/CV metrics table,
parity + calibration plots, stage diagnostics for C); 6 active learning (suggest 6 compositions,
table + simplex plot, export `suggestions.csv`); 7 pointer to campaign-evaluation CLI. Every section
reads CONFIG — switching backend/train/FOM/kinetics requires editing only that cell.

## Testing bar

- Unit tests per module as listed; fast (<~2 min total): small n, `n_multistart<=4`, coarse grids.
- One end-to-end smoke test (`test_campaign.py::test_end_to_end_demo`): synthetic 3-round campaign →
  fit train C → suggest 6 → replay_campaign on trains A/B/C → report builds.
- Determinism: same rng seed → same suggestions.

## Error-propagation summary (must hold everywhere)

measurement noise → (a) trace noise → kinetic fit covariance (`KineticFitResult.cov`) →
heteroscedastic `y_sd` for stage-2 GPs in train C, and (b) FOM sd via linear-integral error or MC →
`y_sd` for FOM-target GPs in trains A/B. GP predictive variances propagate between stages by MC
sampling; final FOM uncertainty = MC sd; calibration checked via coverage metrics in
`cross_validate` and campaign replay.
