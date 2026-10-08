"""
SHL Grammar Scoring Engine - Modeling & Feature Fusion Module
Implements feature fusion, strict leakage-free preprocessing, regression models, ensemble blending, and tuning.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict, RandomizedSearchCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fuse_features(
    audio_df: pd.DataFrame,
    text_df: pd.DataFrame,
    target_df: Optional[pd.DataFrame] = None,
    filename_col: str = "filename",
    target_col: Optional[str] = "grammar_score"
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Fuse acoustic, linguistic, and grammar features into a unified feature matrix.

    Returns:
        (X, y, feature_names)
    """
    # Merge audio and text features
    fused_df = pd.merge(audio_df, text_df, on=filename_col, how="inner")

    # If target dataframe is provided, merge it
    y = None
    if target_df is not None and target_col is not None and target_col in target_df.columns:
        fused_df = pd.merge(fused_df, target_df[[filename_col, target_col]], on=filename_col, how="left")
        y = fused_df[target_col].copy()

    # Drop non-feature identifier columns
    drop_cols = [filename_col]
    if target_col and target_col in fused_df.columns:
        drop_cols.append(target_col)
    if "transcript" in fused_df.columns:
        drop_cols.append("transcript")

    feature_cols = [c for c in fused_df.columns if c not in drop_cols]
    X = fused_df[feature_cols].copy()

    # Clean non-numeric, infs, NaNs
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute missing values with column median
    X = X.fillna(X.median())
    # If any column is still all NaN, fill with 0
    X = X.fillna(0.0)

    # Remove zero-variance (constant) columns
    std_devs = X.std()
    non_constant_cols = std_devs[std_devs > 1e-8].index.tolist()
    X = X[non_constant_cols]

    logger.info(f"Fused feature matrix constructed: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, non_constant_cols


def clip_predictions(predictions: np.ndarray, min_val: float = 0.0, max_val: float = 5.0) -> np.ndarray:
    """
    Constrain predictions to the valid domain [0.0, 5.0] matching the official scoring rubric.
    """
    return np.clip(predictions, min_val, max_val)


class BlendRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble blend of multiple diverse regression estimators.
    """
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], weights: Optional[List[float]] = None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        from sklearn.base import clone
        self.models_ = []
        for name, est in self.estimators:
            model = clone(est).fit(X, y)
            self.models_.append((name, model))

        if self.weights is None:
            self.normalized_weights_ = [1.0 / len(self.models_)] * len(self.models_)
        else:
            total = sum(self.weights)
            self.normalized_weights_ = [w / total for w in self.weights]
        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        for (name, model), w in zip(self.models_, self.normalized_weights_):
            preds += w * model.predict(X)
        return clip_predictions(preds)


def get_baseline_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Initialize standard baseline and competitive regression models.
    """
    models: Dict[str, BaseEstimator] = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=3, random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=random_state
        )
    }

    # Add LightGBM if installed
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1
        )
    except ImportError:
        pass

    # Add XGBoost if installed
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            verbosity=0
        )
    except ImportError:
        pass

    # Add CatBoost if installed
    try:
        import catboost as cb
        models["CatBoost"] = cb.CatBoostRegressor(
            iterations=150,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=0
        )
    except ImportError:
        pass

    return models


def tune_lightgbm_or_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lightgbm",
    cv: int = 5,
    random_state: int = 42
) -> BaseEstimator:
    """
    Perform careful hyperparameter search on 5-fold CV to prevent overfitting on ~769 samples.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            base_estimator = lgb.LGBMRegressor(random_state=random_state, verbose=-1)
            param_grid = {
                "n_estimators": [60, 100, 150, 200],
                "learning_rate": [0.02, 0.04, 0.06, 0.1],
                "max_depth": [3, 4, 5],
                "num_leaves": [7, 15, 25],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 5.0]
            }
            search = RandomizedSearchCV(
                base_estimator,
                param_distributions=param_grid,
                n_iter=20,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=random_state,
                n_jobs=-1
            )
            search.fit(X_scaled, y)
            logger.info(f"Best LightGBM params: {search.best_params_}, CV RMSE: {-search.best_score_:.4f}")
            return search.best_estimator_
        except ImportError:
            model_type = "ridge"

    if model_type == "ridge":
        base_estimator = Ridge(random_state=random_state)
        param_grid = {"alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]}
        search = RandomizedSearchCV(
            base_estimator,
            param_distributions=param_grid,
            n_iter=9,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )
        search.fit(X_scaled, y)
        logger.info(f"Best Ridge params: {search.best_params_}, CV RMSE: {-search.best_score_:.4f}")
        return search.best_estimator_

    return GradientBoostingRegressor(random_state=random_state)
