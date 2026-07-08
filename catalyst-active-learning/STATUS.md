# Project status — zeoal (snapshot)

Work was stopped mid-build at the user's request. This file records exactly what is complete and
tested, and what remains, so the build can be continued locally. **`DESIGN.md` is the authoritative
blueprint**: every remaining module has its full interface contract, algorithms and test plan
specified there — implementing the remainder is a matter of filling in the specified files.

## Complete and tested (48/48 unit tests passing, `python3 -m pytest tests/ -q`)

| File | What it is | Tests |
|---|---|---|
| `zeoal/data.py` | Composition simplex + box-constrained `CompositionSpace`, `TemperatureTrace` (K, sorted, validated), `CatalystSample` (selected + measured composition + trace + round index + replicate group), `CatalystDataset` with JSON/CSV round-trip I/O, common temperature grid, round filtering; simplex utilities (`project_to_simplex`, Dirichlet sampling), rng normalization | exercised via other tests |
| `zeoal/equilibrium.py` | CO2-methanation thermodynamics: ΔG(T) with ΔCp correction, `keq_methanation` (verified: Kp≈1.5e3 bar⁻² @ 673 K, K=1 crossover ≈ 870 K, matches literature), `keq_rwgs`, `equilibrium_conversion` (bisection, vectorized) — provides the equilibrium cap that gives CH4 traces their high-T decline | exercised via kinetics tests |
| `zeoal/kinetics.py` | Full kinetic-fitting engine per DESIGN.md: `FeedConditions`, `KineticFitResult` (covariance, `sample_params`, AIC/BIC, serialization), `KineticModel` ABC with bounded multistart `least_squares` fit + covariance from Jacobian; models: `arrhenius` (two parallel first-order channels CH4/CO + equilibrium cap), `eyring` (transition-state form), `lhhw` (Koschany-2016-style LHHW with PFR integration via `solve_ivp`), `sigmoid` (empirical light-off × equilibrium decline); `fit_trace`, `fit_all_models`, `select_best_model`, `predict_trace`, `trace_band` (MC uncertainty bands), `physical_params` | 22 passing |
| `zeoal/fom.py` | FOM registry + `FOMResult`: **`ch4_temperature_integral` (the requested new FOM)**, `weighted_ch4_integral` (1/T and low-T Boltzmann weights), `selectivity_weighted_integral`, `co_penalized_integral`, `t50/t10/t90`, `max_ch4`, `peak_temperature`, `ch4_at_T`; temperature windows as exact linear stencils; **analytic error propagation** for linear integral FOMs (trapezoid-weight variance), MC for nonlinear ones; `fom_from_fit` propagates kinetic-fit covariance → FOM mean/sd by sampling `KineticFitResult.sample_params` | 26 passing |
| `zeoal/augment.py` | Schoenebeck-style noisy-data augmentation: `NoiseAugmentConfig` (n_copies=10 default, absolute/relative target noise, optional input noise, keep-original), `augment_training_data` (returns provenance `groups` for leak-free group CV), `estimate_noise_from_replicates` (pooled replicate sd) | exercised via config validation; unit tests specified in DESIGN.md |
| `DESIGN.md` | Binding architecture + interface contract for **all** modules incl. the unimplemented ones | — |
| `docs/research_findings.json` | Raw structured findings of six literature researchers (see below) | — |
| `pyproject.toml`, `requirements.txt`, `.gitignore`, `zeoal/__init__.py` (lazy exports) | packaging; `pip install -e .` works | — |

## Not yet implemented (spec ready in DESIGN.md)

1. `zeoal/models.py` — surrogate registry (`gp` with data-driven kernel search/ARD/heteroscedastic
   noise, `rf`, `linear`, `edbo` adapter + always-available `edbo_like` fallback,
   `MultiOutputModel`). See DESIGN.md §models — the GP hyperparameter policy (kernel grid, median-
   distance lengthscale init/bounds, LOO log-predictive-density selection) is fully specified.
2. `zeoal/trains.py` — model trains A (selected→FOM), B (selected→measured→FOM),
   C (selected→measured→kinetic params→trace→FOM) with Monte-Carlo error propagation and
   per-stage variance decomposition. See DESIGN.md §trains.
3. `zeoal/active_learning.py` — batch suggestions (default q=6) on the simplex: MC-sample EI/UCB/PI/
   Thompson acquisitions + kriging-believer / constant-liar / local-penalization batch strategies.
   See DESIGN.md §active_learning.
4. `zeoal/synthetic.py` — synthetic ground truth + EDBO-like emulated historical campaign
   (needed for demos and for validating the campaign evaluator). See DESIGN.md §synthetic.
5. `zeoal/campaign.py` + `scripts/evaluate_campaign.py` — the separate retrospective
   campaign-evaluation process (replay rounds, per-train RMSE/NLL/coverage/Spearman +
   would-have-chosen ranking analysis + forward simulation regret curves). See DESIGN.md §campaign.
6. `zeoal/plotting.py` — trace/fit/parity/calibration/ternary-landscape/suggestion/campaign plots.
7. `notebooks/catalyst_active_learning.ipynb` — single CONFIG cell switching backend
   (EDBO/GP/RF/linear), train (A/B/C), kinetic model, FOM, augmentation; sections per DESIGN.md.
8. `LITERATURE.md` — to be written from `docs/research_findings.json`.

## Literature research (complete, raw)

`docs/research_findings.json` holds structured summaries, explicit equations, design
recommendations, caveats and full citations from six parallel researchers:

- `traces` — typical CO2-methanation light-off/volcano traces, equilibrium ceiling numbers,
  zeolite-supported examples, synthetic-curve recipes (incl. per-metal T50/selectivity anchors).
- `kinetics` — power-law/Arrhenius light-off fitting, Eyring, the Koschany et al. 2016 LHHW rate law
  with exact parameter values, reparameterization and weighted-least-squares practice.
- `edbo` — EDBO/EDBO+ model internals (Matern-5/2 ARD GP, EI/TS, batch via fantasies), API shapes,
  install caveats, and exactly what an `edbo_like` fallback must replicate.
- `fom` — FOM menu (T50, yield@T, STY, AUC-style integrals, weighted variants), critiques of a
  single integral FOM and recommended complements.
- `schoenebeck` — the Schoenebeck-group noisy-data augmentation papers and the concrete
  augment-with-Gaussian-copies algorithm + defaults (already implemented in `zeoal/augment.py`).
- `bo_gp` — batch BO on the simplex (Dirichlet candidates, ilr transforms), GP small-data
  hyperparameter practice, q-batch strategies without BoTorch, MC uncertainty propagation through
  chained models, calibration metrics.

## How to continue

```bash
cd catalyst-active-learning
pip install -e . && python3 -m pytest tests/ -q   # 48 passing now
# then implement, in order (each unblocks the next):
#   zeoal/models.py -> zeoal/trains.py -> zeoal/active_learning.py -> zeoal/synthetic.py
#   -> zeoal/campaign.py + scripts/evaluate_campaign.py -> zeoal/plotting.py -> notebook
```
