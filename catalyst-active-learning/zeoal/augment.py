"""Noisy-data augmentation for training surrogates on noisy experiments.

Implements the Schoenebeck-group strategy of augmenting a noisy experimental
training set with noise-perturbed copies of each point: every training sample
is replicated ``n_copies`` times with Gaussian noise (scaled to the estimated
experimental uncertainty) added to the target — and optionally to the inputs —
which regularizes small-data models and communicates the noise floor to
learners that lack an explicit noise model (random forests in particular).
See LITERATURE.md ("Schoenebeck noisy-data augmentation") for the exact
citation, reported benefits and caveats.

Rules of use (enforced by callers, see DESIGN.md):

* augmentation is applied to **training folds only** — never to held-out data;
* augmented copies carry the index of their parent sample (``groups``) so that
  cross-validation can split by *original* sample and never leak copies of a
  test point into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import as_generator

__all__ = [
    "NoiseAugmentConfig",
    "augment_training_data",
    "estimate_noise_from_replicates",
]


@dataclass
class NoiseAugmentConfig:
    """Configuration of Schoenebeck-style noisy-data augmentation.

    Parameters
    ----------
    n_copies:
        Number of noise-perturbed copies generated per training point
        (default 10; the original point is kept in addition when
        ``keep_original``).
    target_noise:
        1-sigma noise added to targets.  ``None`` means "use the per-point
        ``y_sd`` passed to :func:`augment_training_data`"; a scalar applies
        uniformly; an array gives per-point sigmas (overrides ``y_sd``).
    input_noise:
        Optional 1-sigma Gaussian noise added to the *inputs* of the copies
        (e.g. compositional uncertainty).  ``None`` disables input noise.
    keep_original:
        Keep the unperturbed point alongside its copies (recommended).
    relative:
        If True, ``target_noise`` is interpreted as a fraction of ``|y|``
        (with ``min_abs_noise`` as floor) instead of an absolute sigma.
    min_abs_noise:
        Absolute noise floor used when ``relative`` is True.
    """

    n_copies: int = 10
    target_noise: float | np.ndarray | None = None
    input_noise: float | None = None
    keep_original: bool = True
    relative: bool = False
    min_abs_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.n_copies < 0:
            raise ValueError("n_copies must be >= 0")
        if self.n_copies == 0 and not self.keep_original:
            raise ValueError("n_copies=0 with keep_original=False leaves no data")


def _resolve_sigma(
    y: np.ndarray, y_sd: np.ndarray | None, config: NoiseAugmentConfig
) -> np.ndarray:
    if config.target_noise is None:
        if y_sd is None:
            raise ValueError(
                "no noise scale available: pass y_sd or set NoiseAugmentConfig.target_noise"
            )
        sigma = np.asarray(y_sd, dtype=float)
    elif np.isscalar(config.target_noise):
        base = float(config.target_noise)  # type: ignore[arg-type]
        sigma = (
            np.maximum(np.abs(y) * base, config.min_abs_noise)
            if config.relative
            else np.full_like(y, base, dtype=float)
        )
    else:
        sigma = np.asarray(config.target_noise, dtype=float)
    sigma = np.broadcast_to(sigma, y.shape).astype(float)
    if np.any(sigma < 0) or np.any(~np.isfinite(sigma)):
        raise ValueError("noise sigmas must be finite and non-negative")
    return sigma


def augment_training_data(
    X: np.ndarray,
    y: np.ndarray,
    y_sd: np.ndarray | None = None,
    config: NoiseAugmentConfig | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Augment a training set with noise-perturbed copies.

    Parameters
    ----------
    X:
        Inputs, shape ``(n, d)``.
    y:
        Targets, shape ``(n,)``.
    y_sd:
        Optional per-point 1-sigma target uncertainties, shape ``(n,)``.
    config:
        Augmentation settings (defaults: 10 copies + original, noise from
        ``y_sd``).
    rng:
        Random source (Generator, seed or None).

    Returns
    -------
    ``(X_aug, y_aug, sd_aug, groups)`` where ``groups[i]`` is the index of the
    original sample each augmented row derives from (for group-aware CV) and
    ``sd_aug`` carries the per-row sigma (originals keep their input ``y_sd``
    or the resolved sigma; copies get the same sigma — the added noise does
    not reduce the stated uncertainty).
    """
    config = config or NoiseAugmentConfig()
    gen = as_generator(rng)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but y has {n}")
    sigma = _resolve_sigma(y, np.asarray(y_sd, dtype=float).ravel() if y_sd is not None else None, config)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    idx = np.arange(n)
    if config.keep_original:
        xs.append(X)
        ys.append(y)
        sds.append(sigma)
        gs.append(idx)
    for _ in range(config.n_copies):
        y_pert = y + gen.normal(0.0, 1.0, size=n) * sigma
        X_pert = X.copy()
        if config.input_noise is not None and config.input_noise > 0:
            X_pert = X + gen.normal(0.0, config.input_noise, size=X.shape)
        xs.append(X_pert)
        ys.append(y_pert)
        sds.append(sigma)
        gs.append(idx)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys),
        np.concatenate(sds),
        np.concatenate(gs),
    )


def estimate_noise_from_replicates(values: np.ndarray, groups: np.ndarray) -> float:
    """Pooled replicate standard deviation of a quantity (e.g. a FOM).

    Parameters
    ----------
    values:
        Measured values, shape ``(n,)``.
    groups:
        Replicate-group labels aligned with ``values``; groups with a single
        member contribute nothing.

    Returns
    -------
    Pooled 1-sigma estimate ``sqrt(sum_g (n_g - 1) s_g^2 / sum_g (n_g - 1))``.
    Raises ``ValueError`` when no group has replication.
    """
    values = np.asarray(values, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if values.size != groups.size:
        raise ValueError("values and groups must align")
    num = 0.0
    dof = 0
    for g in np.unique(groups):
        v = values[groups == g]
        if v.size >= 2:
            num += (v.size - 1) * float(np.var(v, ddof=1))
            dof += v.size - 1
    if dof == 0:
        raise ValueError(
            "no replicated groups found; provide replicate_group labels or set "
            "an explicit noise level"
        )
    return float(np.sqrt(num / dof))
