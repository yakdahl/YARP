"""zeoal — active learning for zeolite-supported multimetallic CO2-methanation catalysts.

Lazy re-exports: submodules are imported on first attribute access so that the
package can be used piecemeal (e.g. data handling without matplotlib).
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_ATTRS = {
    # data
    "CompositionSpace": "zeoal.data",
    "TemperatureTrace": "zeoal.data",
    "CatalystSample": "zeoal.data",
    "CatalystDataset": "zeoal.data",
    "project_to_simplex": "zeoal.data",
    "as_generator": "zeoal.data",
    # equilibrium
    "keq_methanation": "zeoal.equilibrium",
    "equilibrium_conversion": "zeoal.equilibrium",
    # kinetics
    "KINETIC_MODELS": "zeoal.kinetics",
    "FeedConditions": "zeoal.kinetics",
    "fit_trace": "zeoal.kinetics",
    "fit_all_models": "zeoal.kinetics",
    "select_best_model": "zeoal.kinetics",
    "predict_trace": "zeoal.kinetics",
    "trace_band": "zeoal.kinetics",
    # fom
    "FOM_REGISTRY": "zeoal.fom",
    "FOM_INFO": "zeoal.fom",
    "fom_from_trace": "zeoal.fom",
    "fom_from_fit": "zeoal.fom",
    # augmentation
    "NoiseAugmentConfig": "zeoal.augment",
    "augment_training_data": "zeoal.augment",
    "estimate_noise_from_replicates": "zeoal.augment",
    # models
    "MODEL_REGISTRY": "zeoal.models",
    "create_model": "zeoal.models",
    "MultiOutputModel": "zeoal.models",
    # trains
    "TRAIN_REGISTRY": "zeoal.trains",
    "create_train": "zeoal.trains",
    # active learning
    "suggest_batch": "zeoal.active_learning",
    # synthetic + campaign
    "make_synthetic_dataset": "zeoal.synthetic",
    "replay_campaign": "zeoal.campaign",
    "simulate_campaign": "zeoal.campaign",
}

__all__ = ["__version__", *sorted(_LAZY_ATTRS)]


def __getattr__(name: str) -> Any:  # PEP 562 lazy loading
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'zeoal' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
