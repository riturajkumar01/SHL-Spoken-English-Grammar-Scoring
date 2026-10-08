"""
SHL Grammar Scoring Engine - Evaluation & Visualization Module
Implements cross-validation protocols, Pearson correlation, RMSE, visual diagnostics, and submission generator.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson Correlation Coefficient (r).
    Handles edge cases like zero variance gracefully.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if np.std(y_pred_arr) < 1e-8 or np.std(y_true_arr) < 1e-8:
        return 0.0
    r_val, _ = pearsonr(y_true_arr, y_pred_arr)
    return float(r_val) if not np.isnan(r_val) else 0.0


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard competition metrics (RMSE, Pearson r)."""
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "pearson": compute_pearson(y_true, y_pred)
    }


def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    groups: Optional[pd.Series] = None,
    scale_features: bool = True,
    random_state: int = 42
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate a model using K-Fold (or GroupKFold if groups provided).
    Strictly fits scalers and transformations inside each training fold to prevent leakage.

    Returns:
        (metrics_dict, out_of_fold_predictions)
    """
    X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_vec = y.values if isinstance(y, pd.Series) else np.asarray(y)

    oof_preds = np.zeros(len(y_vec))
    fold_rmses = []
    fold_pearsons = []

    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        split_gen = cv.split(X_mat, y_vec, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_gen = cv.split(X_mat, y_vec)

    for fold_idx, (train_idx, val_idx) in enumerate(split_gen):
        X_train_f, y_train_f = X_mat[train_idx], y_vec[train_idx]
        X_val_f, y_val_f = X_mat[val_idx], y_vec[val_idx]

        if scale_features:
            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train_f)
            X_val_f = scaler.transform(X_val_f)

        from sklearn.base import clone
        fold_model = clone(model) if hasattr(model, "fit") else model
        fold_model.fit(X_train_f, y_train_f)
        val_pred = np.clip(fold_model.predict(X_val_f), 0.0, 5.0)

        oof_preds[val_idx] = val_pred
        fold_rmses.append(compute_rmse(y_val_f, val_pred))
        fold_pearsons.append(compute_pearson(y_val_f, val_pred))

    overall_rmse = compute_rmse(y_vec, oof_preds)
    overall_pearson = compute_pearson(y_vec, oof_preds)

    metrics = {
        "mean_cv_rmse": float(np.mean(fold_rmses)),
        "std_cv_rmse": float(np.std(fold_rmses)),
        "mean_cv_pearson": float(np.mean(fold_pearsons)),
        "std_cv_pearson": float(np.std(fold_pearsons)),
        "overall_oof_rmse": overall_rmse,
        "overall_oof_pearson": overall_pearson
    }

    return metrics, oof_preds


def compare_models(
    models_dict: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    groups: Optional[pd.Series] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Run consistent 5-fold CV across all candidate models and return summary table.
    """
    summary_records = []
    all_oof_preds = {}

    for name, model in models_dict.items():
        logger.info(f"Cross-validating model: {name}...")
        metrics, oof = evaluate_model_cv(
            model=model,
            X=X,
            y=y,
            n_splits=n_splits,
            groups=groups,
            scale_features=True,
            random_state=random_state
        )
        all_oof_preds[name] = oof
        summary_records.append({
            "Model": name,
            "CV RMSE (Mean)": round(metrics["mean_cv_rmse"], 4),
            "CV RMSE (Std)": round(metrics["std_cv_rmse"], 4),
            "CV Pearson (Mean)": round(metrics["mean_cv_pearson"], 4),
            "CV Pearson (Std)": round(metrics["std_cv_pearson"], 4),
            "OOF RMSE": round(metrics["overall_oof_rmse"], 4),
            "OOF Pearson": round(metrics["overall_oof_pearson"], 4),
        })

    summary_df = pd.DataFrame(summary_records).sort_values("CV RMSE (Mean)").reset_index(drop=True)
    return summary_df, all_oof_preds


def plot_target_distribution(y: pd.Series, save_path: Optional[str] = None):
    """Plot distribution histogram and boxplot for target grammar scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(y, kde=True, ax=axes[0], color="#2b5c8f", bins=15)
    axes[0].set_title("Grammar Score Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Grammar Score (0-5)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    sns.boxplot(x=y, ax=axes[1], color="#e07a5f")
    axes[1].set_title("Grammar Score Boxplot & Spread", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Grammar Score (0-5)", fontsize=11)
    axes[1].grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Grammar Scores",
    save_path: Optional[str] = None
):
    """Scatter plot of actual vs predicted scores with ideal diagonal line."""
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="#1f77b4", edgecolors="k", s=40)
    min_val, max_val = 0.0, 5.0
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")

    rmse_val = compute_rmse(y_true, y_pred)
    r_val = compute_pearson(y_true, y_pred)

    plt.title(f"{title}\nRMSE: {rmse_val:.4f} | Pearson r: {r_val:.4f}", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Grammar Score", fontsize=11)
    plt.ylabel("Predicted Grammar Score", fontsize=11)
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 5.2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """Plot residual distribution and residuals vs predicted values."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals, kde=True, ax=axes[0], color="#e76f51", bins=20)
    axes[0].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[0].set_title("Residual Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Residual", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].scatter(y_pred, residuals, alpha=0.6, color="#2a9d8f", edgecolors="k", s=35)
    axes[1].axhline(0, color="r", linestyle="--", lw=1.5)
    axes[1].set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Score", fontsize=11)
    axes[1].set_ylabel("Residual", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """Plot top N feature importances."""
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title(f"Top {top_n} Most Informative Features", fontsize=13, fontweight="bold")
    plt.xlabel("Feature Importance", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def create_submission_file(
    test_df: pd.DataFrame,
    test_predictions: np.ndarray,
    sample_sub_df: Optional[pd.DataFrame] = None,
    filename_col: str = "filename",
    target_col: str = "grammar_score",
    output_path: str = "outputs/submission.csv"
) -> pd.DataFrame:
    """
    Format, validate, and write the Kaggle submission CSV.
    """
    # Enforce valid 0-5 range clipping
    clipped_preds = np.clip(test_predictions, 0.0, 5.0)

    # Determine column names based on sample_submission if present
    out_id_col = filename_col
    out_target_col = target_col

    if sample_sub_df is not None:
        sub_cols = list(sample_sub_df.columns)
        if len(sub_cols) >= 2:
            out_id_col = sub_cols[0]
            out_target_col = sub_cols[1]

    submission_df = pd.DataFrame({
        out_id_col: test_df[filename_col].values,
        out_target_col: clipped_preds
    })

    # If sample submission is available, align exact row ordering
    if sample_sub_df is not None and out_id_col in sample_sub_df.columns:
        submission_df = sample_sub_df[[out_id_col]].merge(submission_df, on=out_id_col, how="left")
        # Check for any missing values after merge
        if submission_df[out_target_col].isnull().any():
            submission_df[out_target_col] = submission_df[out_target_col].fillna(3.0)

    # Validation assertions
    assert not submission_df[out_target_col].isnull().any(), "Submission contains null predictions!"
    assert (submission_df[out_target_col] >= 0.0).all() and (submission_df[out_target_col] <= 5.0).all(), \
        "Predictions exceed valid [0, 5] range!"
    if sample_sub_df is not None:
        assert len(submission_df) == len(sample_sub_df), \
            f"Submission length ({len(submission_df)}) does not match sample submission ({len(sample_sub_df)})!"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Successfully generated validated submission: {output_path} with shape {submission_df.shape}")
    return submission_df
