"""
SHL Grammar Scoring Engine - End-to-End Pipeline Runner
Executes the full preprocessing, feature engineering, model benchmarking, validation, and submission generation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Import our custom modules
from src.preprocessing import discover_dataset_paths, detect_columns, inspect_dataframe, load_audio_file
from src.audio_features import extract_audio_features_dataset
from src.text_features import transcribe_dataset, extract_all_text_features
from src.model import fuse_features, get_baseline_models, BlendRegressor, tune_lightgbm_or_ridge
from src.evaluation import (
    compare_models,
    evaluate_model_cv,
    compute_rmse,
    compute_pearson,
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    create_submission_file
)
from scripts.generate_sample_dataset import generate_mock_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SHL_Pipeline")


def run_pipeline():
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    print("=" * 70)
    print("      SHL SPOKEN ENGLISH GRAMMAR SCORING ENGINE - PIPELINE EXECUTION      ")
    print("=" * 70)

    # 1. Discover Paths
    paths = discover_dataset_paths()
    train_csv_path = paths.get("train_csv")
    test_csv_path = paths.get("test_csv")
    sub_csv_path = paths.get("sample_submission_csv")

    if not train_csv_path or not os.path.exists(train_csv_path):
        logger.info("Dataset not detected in default directories. Generating local benchmark dataset...")
        generate_mock_dataset("data")
        paths = discover_dataset_paths(["./data"])
        train_csv_path = paths["train_csv"]
        test_csv_path = paths["test_csv"]
        sub_csv_path = paths["sample_submission_csv"]

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path) if test_csv_path and os.path.exists(test_csv_path) else None
    sample_sub_df = pd.read_csv(sub_csv_path) if sub_csv_path and os.path.exists(sub_csv_path) else None

    train_cols = detect_columns(train_df, is_train=True)
    filename_col = train_cols["filename"]
    target_col = train_cols["target"]

    train_audio_dir = paths.get("train_audio_dir") or "data/train_audio"
    test_audio_dir = paths.get("test_audio_dir") or "data/test_audio"

    logger.info(f"Dataset Loaded: {len(train_df)} train samples, {len(test_df) if test_df is not None else 0} test samples.")
    logger.info(f"Filename col: '{filename_col}', Target col: '{target_col}'")

    # 2. Extract Acoustic Features
    os.makedirs("outputs", exist_ok=True)
    train_audio_feats = extract_audio_features_dataset(
        df=train_df,
        audio_dir=train_audio_dir,
        filename_col=filename_col,
        cache_path="outputs/audio_features_train.csv"
    )

    test_audio_feats = extract_audio_features_dataset(
        df=test_df,
        audio_dir=test_audio_dir,
        filename_col=filename_col,
        cache_path="outputs/audio_features_test.csv"
    ) if test_df is not None else None

    # 3. Speech-to-Text Transcriptions
    train_transcripts = transcribe_dataset(
        df=train_df,
        audio_dir=train_audio_dir,
        filename_col=filename_col,
        cache_path="outputs/transcripts_train.csv"
    )

    test_transcripts = transcribe_dataset(
        df=test_df,
        audio_dir=test_audio_dir,
        filename_col=filename_col,
        cache_path="outputs/transcripts_test.csv"
    ) if test_df is not None else None

    # 4. Linguistic & Grammar Features
    train_durations = dict(zip(train_audio_feats[filename_col], train_audio_feats["audio_duration"]))
    train_text_feats = extract_all_text_features(
        transcripts_df=train_transcripts,
        filename_col=filename_col,
        audio_durations=train_durations
    )

    test_durations = dict(zip(test_audio_feats[filename_col], test_audio_feats["audio_duration"])) if test_audio_feats is not None else None
    test_text_feats = extract_all_text_features(
        transcripts_df=test_transcripts,
        filename_col=filename_col,
        audio_durations=test_durations
    ) if test_transcripts is not None else None

    # 5. Feature Fusion
    X_train, y_train, feature_cols = fuse_features(
        audio_df=train_audio_feats,
        text_df=train_text_feats,
        target_df=train_df,
        filename_col=filename_col,
        target_col=target_col
    )

    if test_audio_feats is not None and test_text_feats is not None:
        X_test, _, _ = fuse_features(
            audio_df=test_audio_feats,
            text_df=test_text_feats,
            target_df=None,
            filename_col=filename_col,
            target_col=None
        )
        X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)
    else:
        X_test = None

    # 6. Baseline & Model Benchmarking
    logger.info("Benchmarking candidate regression models across 5-Fold Cross-Validation...")
    candidate_models = get_baseline_models(random_state=RANDOM_STATE)
    summary_df, oof_predictions = compare_models(
        models_dict=candidate_models,
        X=X_train,
        y=y_train,
        n_splits=5,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 70)
    print("                     MODEL BENCHMARK RESULTS                     ")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # 7. Final Model & Ensembling
    logger.info("Building and evaluating final blended ensemble model...")
    ensemble = BlendRegressor(
        estimators=[
            ("ridge", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
            ("grad_boost", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE)),
            ("rf", candidate_models["Random Forest"])
        ],
        weights=[0.4, 0.35, 0.25]
    )

    final_cv_metrics, final_oof = evaluate_model_cv(
        model=ensemble,
        X=X_train,
        y=y_train,
        n_splits=5,
        scale_features=True,
        random_state=RANDOM_STATE
    )

    # 8. Train on 100% Training Data & Report COMPULSORY Training RMSE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ensemble.fit(X_train_scaled, y_train)

    train_preds = np.clip(ensemble.predict(X_train_scaled), 0.0, 5.0)
    train_rmse = compute_rmse(y_train.values, train_preds)
    train_pearson = compute_pearson(y_train.values, train_preds)
    val_rmse = final_cv_metrics["overall_oof_rmse"]
    val_pearson = final_cv_metrics["overall_oof_pearson"]

    print("\n" + "=" * 70)
    print("                    FINAL MODEL PERFORMANCE REPORT                    ")
    print("=" * 70)
    print(f"  Training RMSE (COMPULSORY):     {train_rmse:.4f}")
    print(f"  Training Pearson Correlation:   {train_pearson:.4f}")
    print(f"  Validation RMSE (5-Fold OOF):   {val_rmse:.4f}")
    print(f"  Validation Pearson Correlation: {val_pearson:.4f}")
    print("=" * 70 + "\n")

    # 9. Plots
    plot_target_distribution(y_train, save_path="outputs/target_distribution.png")
    plot_actual_vs_predicted(y_train.values, final_oof, save_path="outputs/val_actual_vs_predicted.png")
    plot_residuals(y_train.values, final_oof, save_path="outputs/val_residuals.png")

    # Feature Importance Plot
    tree_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    tree_model.fit(X_train_scaled, y_train)
    plot_feature_importance(feature_cols, tree_model.feature_importances_, top_n=15, save_path="outputs/feature_importance.png")

    # 10. Generate Test Predictions and Submission CSV
    if X_test is not None and test_df is not None:
        X_test_scaled = scaler.transform(X_test)
        test_preds = np.clip(ensemble.predict(X_test_scaled), 0.0, 5.0)

        sub_df = create_submission_file(
            test_df=test_df,
            test_predictions=test_preds,
            sample_sub_df=sample_sub_df,
            filename_col=filename_col,
            target_col=target_col,
            output_path="outputs/submission.csv"
        )
        print("Submission preview:")
        print(sub_df.head(10).to_string(index=False))

    return {
        "train_rmse": train_rmse,
        "train_pearson": train_pearson,
        "val_rmse": val_rmse,
        "val_pearson": val_pearson,
        "summary_df": summary_df
    }


if __name__ == "__main__":
    run_pipeline()

