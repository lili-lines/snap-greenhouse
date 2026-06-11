import importlib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

_ALIASES = {
    "LinearRegression":          "sklearn.linear_model.LinearRegression",
    "Ridge":                     "sklearn.linear_model.Ridge",
    "Lasso":                     "sklearn.linear_model.Lasso",
    "RandomForestRegressor":     "sklearn.ensemble.RandomForestRegressor",
    "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
    "PolynomialFeatures":        "sklearn.preprocessing.PolynomialFeatures",
    "LGBMRegressor":             "lightgbm.LGBMRegressor",
}


def _resolve(cls_path: str):
    """
    a class name -> a Python class.

    accepts either a short alias from _ALIASES, or full path `module.submodule.Class`
    import is done lazily -> no model needs to be listed nor hard-imported here
    """
    path = _ALIASES.get(cls_path, cls_path)
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(
            f"unknown cls '{cls_path}': give a full path "
            f"(e.g. sklearn.svm.SVR) or add an alias in _ALIASES."
        )
    return getattr(importlib.import_module(module_name), class_name)


def make_model(*steps):
    """
    standardScaler on X + steps, with target 'y' normalisation
    """
    pipe = make_pipeline(StandardScaler(), *steps)
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def _build_step(step: dict):
    """
    instantiate a {cls, params} param step from yaml
    """
    return _resolve(step["cls"])(**step.get("params", {}))


def build_models(cfg_models: dict) -> dict:
    """
    build {name: estimator} from yaml
    type 'sklearn' -> StandardScaler(X) + normalised target 'y'
    type 'tree'    -> bare estimator, NO scaling
    other types ignored handled elsewhere
    """
    models = {}
    for name, spec in cfg_models.items():
        t = spec.get("type")
        if t not in ("sklearn", "tree"):
            continue
        steps = [_build_step(s) for s in spec["steps"]]
        if t == "sklearn":
            models[name] = make_model(*steps)
        else:  # tree: no scaling
            models[name] = steps[0] if len(steps) == 1 else make_pipeline(*steps)
    return models


def baseline_horizon(cfg_models: dict, default: int = 24) -> int:
    """
    get baseline horizon from yaml
    """
    for spec in cfg_models.values():
        if spec.get("type") == "baseline":
            return spec.get("horizon", default)
    return default


def river_configs(cfg_models: dict) -> dict:
    """
    get all online configs: {name: params}
    river variants (lr, optim...) be evaluated in loop
    """
    return {name: spec.get("params", {})
            for name, spec in cfg_models.items()
            if spec.get("type") == "river"}
