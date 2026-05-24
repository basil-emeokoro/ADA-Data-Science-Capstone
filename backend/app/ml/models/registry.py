from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


def get_model_registry(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=120, random_state=random_state),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state),
        "Support Vector Regressor": Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
    }
    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=0,
        )
    if CatBoostRegressor is not None:
        models["CatBoost Regressor"] = CatBoostRegressor(
            iterations=120,
            learning_rate=0.08,
            depth=4,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    return models
