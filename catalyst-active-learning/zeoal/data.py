"""Core data model for zeolite-supported multimetallic catalyst campaigns.

This module is the shared foundation of :mod:`zeoal`: composition spaces on the
metal-fraction simplex, temperature-concentration traces, catalyst samples and
campaign datasets, plus small numerical utilities used across the package.

Conventions (see DESIGN.md):

* temperatures are stored in kelvin,
* compositions are non-negative arrays summing to 1 (the simplex),
* uncertainties are 1-sigma standard deviations,
* every stochastic routine accepts ``rng`` (``np.random.Generator``, int seed
  or None) normalized through :func:`as_generator`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

KELVIN_OFFSET = 273.15

__all__ = [
    "KELVIN_OFFSET",
    "as_generator",
    "project_to_simplex",
    "dirichlet_sample",
    "CompositionSpace",
    "TemperatureTrace",
    "CatalystSample",
    "CatalystDataset",
]


def as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Normalize ``rng`` into a :class:`numpy.random.Generator`."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Project composition(s) onto the probability simplex.

    Negative entries are clipped to zero and rows renormalized to sum to 1.
    Rows that are entirely non-positive are replaced by the uniform
    composition.  Accepts shape ``(d,)`` or ``(n, d)`` and preserves shape.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    X = np.atleast_2d(x).copy()
    X = np.clip(X, 0.0, None)
    totals = X.sum(axis=1)
    dead = totals <= 0
    if np.any(dead):
        X[dead] = 1.0 / X.shape[1]
        totals[dead] = 1.0
    X = X / totals[:, None]
    return X[0] if single else X


def dirichlet_sample(
    n: int,
    d: int,
    alpha: float | Sequence[float] = 1.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Draw ``n`` compositions from a Dirichlet on the ``d``-simplex."""
    gen = as_generator(rng)
    alpha_vec = np.full(d, float(alpha)) if np.isscalar(alpha) else np.asarray(alpha, dtype=float)
    if alpha_vec.shape != (d,):
        raise ValueError(f"alpha must be scalar or length {d}, got shape {alpha_vec.shape}")
    return gen.dirichlet(alpha_vec, size=n)


@dataclass
class CompositionSpace:
    """Search space of continuous metal fractions on the simplex.

    Parameters
    ----------
    metals:
        Ordered metal names, e.g. ``["Ni", "Ru", "Co"]``.
    min_fraction, max_fraction:
        Optional per-metal box constraints (scalar or per-metal arrays)
        applied on top of the simplex constraint.
    support:
        Free-text description of the support (e.g. ``"SSZ-13"``); metadata only.
    """

    metals: list[str]
    min_fraction: np.ndarray = field(default=None)  # type: ignore[assignment]
    max_fraction: np.ndarray = field(default=None)  # type: ignore[assignment]
    support: str = ""

    def __post_init__(self) -> None:
        self.metals = list(self.metals)
        if len(self.metals) < 2:
            raise ValueError("CompositionSpace needs at least two metals")
        if len(set(self.metals)) != len(self.metals):
            raise ValueError(f"duplicate metal names in {self.metals}")
        d = self.n_metals
        lo = 0.0 if self.min_fraction is None else self.min_fraction
        hi = 1.0 if self.max_fraction is None else self.max_fraction
        self.min_fraction = np.broadcast_to(np.asarray(lo, dtype=float), (d,)).copy()
        self.max_fraction = np.broadcast_to(np.asarray(hi, dtype=float), (d,)).copy()
        if np.any(self.min_fraction < 0) or np.any(self.max_fraction > 1):
            raise ValueError("fraction bounds must lie in [0, 1]")
        if np.any(self.min_fraction > self.max_fraction):
            raise ValueError("min_fraction exceeds max_fraction for some metal")
        if self.min_fraction.sum() > 1.0 + 1e-12:
            raise ValueError("sum of min_fraction exceeds 1; simplex is empty")
        if self.max_fraction.sum() < 1.0 - 1e-12:
            raise ValueError("sum of max_fraction below 1; simplex is empty")

    @property
    def n_metals(self) -> int:
        return len(self.metals)

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> np.ndarray | bool:
        """Whether composition(s) satisfy simplex + box constraints."""
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        X = np.atleast_2d(x)
        if X.shape[1] != self.n_metals:
            raise ValueError(f"expected {self.n_metals} fractions, got {X.shape[1]}")
        ok = (
            (np.abs(X.sum(axis=1) - 1.0) <= max(tol, 1e-6))
            & np.all(X >= self.min_fraction - tol, axis=1)
            & np.all(X <= self.max_fraction + tol, axis=1)
        )
        return bool(ok[0]) if single else ok

    def random(
        self,
        n: int,
        alpha: float | Sequence[float] = 1.0,
        rng: np.random.Generator | int | None = None,
        max_tries: int = 200,
    ) -> np.ndarray:
        """Sample ``n`` compositions uniformly-ish (Dirichlet) respecting bounds.

        Rejection sampling against the box constraints; falls back to clipped
        + renormalized draws if acceptance is very low (documented clipping).
        """
        gen = as_generator(rng)
        out: list[np.ndarray] = []
        for _ in range(max_tries):
            draw = dirichlet_sample(max(n, 64), self.n_metals, alpha, gen)
            keep = draw[self.contains(draw)]
            if len(keep):
                out.append(keep)
            if sum(len(o) for o in out) >= n:
                break
        got = np.concatenate(out)[:n] if out else np.empty((0, self.n_metals))
        if len(got) < n:  # low acceptance: clip into the box and re-project
            draw = dirichlet_sample(n - len(got), self.n_metals, alpha, gen)
            clipped = project_to_simplex(
                np.clip(draw, self.min_fraction, self.max_fraction)
            )
            got = np.concatenate([got, np.atleast_2d(clipped)])
        return got

    def grid(self, resolution: int = 10) -> np.ndarray:
        """Regular barycentric grid with spacing ``1/resolution`` (filtered to bounds)."""
        if resolution < 1:
            raise ValueError("resolution must be >= 1")
        d = self.n_metals

        def compositions(total: int, dims: int) -> Iterable[tuple[int, ...]]:
            if dims == 1:
                yield (total,)
                return
            for i in range(total + 1):
                for rest in compositions(total - i, dims - 1):
                    yield (i, *rest)

        pts = np.array(list(compositions(resolution, d)), dtype=float) / resolution
        return pts[self.contains(pts)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metals": self.metals,
            "min_fraction": self.min_fraction.tolist(),
            "max_fraction": self.max_fraction.tolist(),
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompositionSpace":
        return cls(
            metals=d["metals"],
            min_fraction=np.asarray(d.get("min_fraction", 0.0)),
            max_fraction=np.asarray(d.get("max_fraction", 1.0)),
            support=d.get("support", ""),
        )


@dataclass
class TemperatureTrace:
    """Outlet concentrations of CO2, CH4 and CO versus temperature.

    Arrays are sorted by temperature on construction.  Concentration unit is
    generic (``"mol_frac"``, ``"ppm"``, ...) and only used for labeling; all
    three channels must share it.
    """

    temperature_K: np.ndarray
    c_co2: np.ndarray
    c_ch4: np.ndarray
    c_co: np.ndarray
    concentration_unit: str = "mol_frac"

    def __post_init__(self) -> None:
        self.temperature_K = np.asarray(self.temperature_K, dtype=float).ravel()
        n = self.temperature_K.size
        if n < 3:
            raise ValueError("a trace needs at least 3 temperature points")
        for name in ("c_co2", "c_ch4", "c_co"):
            arr = np.asarray(getattr(self, name), dtype=float).ravel()
            if arr.size != n:
                raise ValueError(f"{name} has {arr.size} points, expected {n}")
            setattr(self, name, arr)
        if np.any(~np.isfinite(self.temperature_K)):
            raise ValueError("non-finite temperatures in trace")
        if np.any(self.temperature_K <= 0):
            raise ValueError("temperatures must be in kelvin (> 0); convert before constructing")
        order = np.argsort(self.temperature_K)
        self.temperature_K = self.temperature_K[order]
        self.c_co2 = self.c_co2[order]
        self.c_ch4 = self.c_ch4[order]
        self.c_co = self.c_co[order]

    @classmethod
    def from_celsius(
        cls,
        temperature_C: np.ndarray,
        c_co2: np.ndarray,
        c_ch4: np.ndarray,
        c_co: np.ndarray,
        concentration_unit: str = "mol_frac",
    ) -> "TemperatureTrace":
        return cls(
            np.asarray(temperature_C, dtype=float) + KELVIN_OFFSET,
            c_co2,
            c_ch4,
            c_co,
            concentration_unit,
        )

    @property
    def n_points(self) -> int:
        return self.temperature_K.size

    @property
    def temperature_C(self) -> np.ndarray:
        return self.temperature_K - KELVIN_OFFSET

    def channel(self, name: str) -> np.ndarray:
        key = {"co2": "c_co2", "ch4": "c_ch4", "co": "c_co"}.get(name.lower())
        if key is None:
            raise ValueError(f"unknown channel {name!r}; use 'co2', 'ch4' or 'co'")
        return getattr(self, key)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "temperature_K": self.temperature_K,
                "c_co2": self.c_co2,
                "c_ch4": self.c_ch4,
                "c_co": self.c_co,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_K": self.temperature_K.tolist(),
            "c_co2": self.c_co2.tolist(),
            "c_ch4": self.c_ch4.tolist(),
            "c_co": self.c_co.tolist(),
            "concentration_unit": self.concentration_unit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TemperatureTrace":
        return cls(
            np.asarray(d["temperature_K"]),
            np.asarray(d["c_co2"]),
            np.asarray(d["c_ch4"]),
            np.asarray(d["c_co"]),
            d.get("concentration_unit", "mol_frac"),
        )


@dataclass
class CatalystSample:
    """One synthesized + tested catalyst.

    ``measured_composition`` may be ``None`` when the sample has not been
    characterized (e.g. suggestions not yet synthesized).  ``round_index``
    orders active-learning rounds (0 = initial seed batch).
    """

    sample_id: str
    selected_composition: np.ndarray
    measured_composition: np.ndarray | None = None
    trace: TemperatureTrace | None = None
    round_index: int = 0
    replicate_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selected_composition = np.asarray(self.selected_composition, dtype=float).ravel()
        if self.measured_composition is not None:
            self.measured_composition = np.asarray(self.measured_composition, dtype=float).ravel()
            if self.measured_composition.shape != self.selected_composition.shape:
                raise ValueError(
                    f"sample {self.sample_id}: measured composition has shape "
                    f"{self.measured_composition.shape}, selected has "
                    f"{self.selected_composition.shape}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "selected_composition": self.selected_composition.tolist(),
            "measured_composition": None
            if self.measured_composition is None
            else self.measured_composition.tolist(),
            "trace": None if self.trace is None else self.trace.to_dict(),
            "round_index": int(self.round_index),
            "replicate_group": self.replicate_group,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CatalystSample":
        return cls(
            sample_id=str(d["sample_id"]),
            selected_composition=np.asarray(d["selected_composition"]),
            measured_composition=None
            if d.get("measured_composition") is None
            else np.asarray(d["measured_composition"]),
            trace=None if d.get("trace") is None else TemperatureTrace.from_dict(d["trace"]),
            round_index=int(d.get("round_index", 0)),
            replicate_group=d.get("replicate_group"),
            metadata=d.get("metadata", {}),
        )


class CatalystDataset:
    """An ordered collection of :class:`CatalystSample` sharing one space."""

    def __init__(self, space: CompositionSpace, samples: Sequence[CatalystSample] = ()):
        self.space = space
        self.samples: list[CatalystSample] = []
        for s in samples:
            self.add(s)

    # -- construction ------------------------------------------------------
    def add(self, sample: CatalystSample) -> None:
        if sample.selected_composition.size != self.space.n_metals:
            raise ValueError(
                f"sample {sample.sample_id}: {sample.selected_composition.size} fractions, "
                f"space has {self.space.n_metals} metals {self.space.metals}"
            )
        if any(s.sample_id == sample.sample_id for s in self.samples):
            raise ValueError(f"duplicate sample_id {sample.sample_id!r}")
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, i: int) -> CatalystSample:
        return self.samples[i]

    # -- views -------------------------------------------------------------
    def selected_X(self) -> np.ndarray:
        """(n, d) array of selected compositions."""
        if not self.samples:
            return np.empty((0, self.space.n_metals))
        return np.vstack([s.selected_composition for s in self.samples])

    def measured_X(self, allow_missing: bool = False) -> np.ndarray:
        """(n, d) measured compositions; missing rows fall back to selected if allowed."""
        rows = []
        for s in self.samples:
            if s.measured_composition is not None:
                rows.append(s.measured_composition)
            elif allow_missing:
                rows.append(s.selected_composition)
            else:
                raise ValueError(
                    f"sample {s.sample_id} has no measured composition; "
                    "pass allow_missing=True to fall back to selected"
                )
        if not rows:
            return np.empty((0, self.space.n_metals))
        return np.vstack(rows)

    def with_traces(self) -> "CatalystDataset":
        return CatalystDataset(self.space, [s for s in self.samples if s.trace is not None])

    def rounds(self) -> list[int]:
        return sorted({s.round_index for s in self.samples})

    def filter_rounds(self, max_round: int | None = None, rounds: Iterable[int] | None = None) -> "CatalystDataset":
        if (max_round is None) == (rounds is None):
            raise ValueError("pass exactly one of max_round / rounds")
        if max_round is not None:
            keep = [s for s in self.samples if s.round_index <= max_round]
        else:
            wanted = set(rounds)  # type: ignore[arg-type]
            keep = [s for s in self.samples if s.round_index in wanted]
        return CatalystDataset(self.space, keep)

    def common_temperature_grid(self, n: int | None = None) -> np.ndarray:
        """Temperature grid spanning the overlap of all traces (median density)."""
        traced = [s.trace for s in self.samples if s.trace is not None]
        if not traced:
            raise ValueError("dataset has no traces")
        lo = max(t.temperature_K.min() for t in traced)
        hi = min(t.temperature_K.max() for t in traced)
        if not hi > lo:
            raise ValueError("traces have no overlapping temperature range")
        if n is None:
            n = int(np.median([t.n_points for t in traced]))
        return np.linspace(lo, hi, max(int(n), 3))

    # -- pandas / disk -----------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """One row per sample: id, round, selected_*/measured_* fractions."""
        rows = []
        for s in self.samples:
            row: dict[str, Any] = {
                "sample_id": s.sample_id,
                "round_index": s.round_index,
                "replicate_group": s.replicate_group,
                "has_trace": s.trace is not None,
            }
            for m, v in zip(self.space.metals, s.selected_composition):
                row[f"selected_{m}"] = v
            for i, m in enumerate(self.space.metals):
                row[f"measured_{m}"] = (
                    np.nan if s.measured_composition is None else s.measured_composition[i]
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def traces_frame(self) -> pd.DataFrame:
        """Tidy long frame of all traces: sample_id, temperature_K, c_co2, c_ch4, c_co."""
        frames = []
        for s in self.samples:
            if s.trace is None:
                continue
            f = s.trace.as_frame()
            f.insert(0, "sample_id", s.sample_id)
            frames.append(f)
        if not frames:
            return pd.DataFrame(columns=["sample_id", "temperature_K", "c_co2", "c_ch4", "c_co"])
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def from_frames(
        cls,
        space: CompositionSpace,
        samples: pd.DataFrame,
        traces: pd.DataFrame | None = None,
        concentration_unit: str = "mol_frac",
        temperature_unit: str = "K",
    ) -> "CatalystDataset":
        """Build a dataset from the tidy frames produced by :meth:`to_frame` /
        :meth:`traces_frame` (or user CSVs with the same columns)."""
        if temperature_unit not in ("K", "C"):
            raise ValueError("temperature_unit must be 'K' or 'C'")
        ds = cls(space)
        trace_groups = (
            {k: g for k, g in traces.groupby("sample_id", sort=False)} if traces is not None else {}
        )
        for _, row in samples.iterrows():
            sid = str(row["sample_id"])
            sel = np.array([row[f"selected_{m}"] for m in space.metals], dtype=float)
            meas_cols = [f"measured_{m}" for m in space.metals]
            meas: np.ndarray | None
            if all(c in row.index for c in meas_cols):
                meas = np.array([row[c] for c in meas_cols], dtype=float)
                if np.any(~np.isfinite(meas)):
                    meas = None
            else:
                meas = None
            trace = None
            if sid in trace_groups:
                g = trace_groups[sid]
                temp_col = "temperature_K" if "temperature_K" in g else "temperature_C"
                T = np.asarray(g[temp_col], dtype=float)
                if temp_col == "temperature_C" or temperature_unit == "C":
                    if temp_col == "temperature_C":
                        T = T + KELVIN_OFFSET
                    elif temperature_unit == "C":
                        T = T + KELVIN_OFFSET
                trace = TemperatureTrace(
                    T,
                    np.asarray(g["c_co2"], dtype=float),
                    np.asarray(g["c_ch4"], dtype=float),
                    np.asarray(g["c_co"], dtype=float),
                    concentration_unit,
                )
            rg = row.get("replicate_group")
            ds.add(
                CatalystSample(
                    sample_id=sid,
                    selected_composition=sel,
                    measured_composition=meas,
                    trace=trace,
                    round_index=int(row.get("round_index", 0)),
                    replicate_group=None if pd.isna(rg) else str(rg),
                )
            )
        return ds

    def save_json(self, path: str | Path) -> None:
        payload = {
            "space": self.space.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load_json(cls, path: str | Path) -> "CatalystDataset":
        payload = json.loads(Path(path).read_text())
        space = CompositionSpace.from_dict(payload["space"])
        return cls(space, [CatalystSample.from_dict(d) for d in payload["samples"]])

    def save_csv(self, samples_path: str | Path, traces_path: str | Path) -> None:
        self.to_frame().to_csv(samples_path, index=False)
        self.traces_frame().to_csv(traces_path, index=False)

    @classmethod
    def load_csv(
        cls,
        space: CompositionSpace,
        samples_path: str | Path,
        traces_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "CatalystDataset":
        samples = pd.read_csv(samples_path)
        traces = pd.read_csv(traces_path) if traces_path is not None else None
        return cls.from_frames(space, samples, traces, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CatalystDataset({len(self)} samples, metals={self.space.metals}, "
            f"rounds={self.rounds()})"
        )
