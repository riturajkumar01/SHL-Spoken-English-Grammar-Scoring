"""
SHL Grammar Scoring Engine - Preprocessing & Dataset Discovery Module
Provides robust path discovery, column identification, audio loading, and data integrity checks.
"""

import os
import glob
import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def discover_dataset_paths(base_search_paths: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """
    Robustly locate dataset files (train.csv, test.csv, sample_submission.csv, and audio folders)
    across standard Kaggle directories (/kaggle/input/...) or local paths.

    Returns:
        Dict mapping dataset keys to resolved file/directory paths.
    """
    if base_search_paths is None:
        base_search_paths = [
            "/kaggle/input",
            "/kaggle/input/*",
            "./data",
            "../data",
            "./dataset",
            ".",
            ".."
        ]

    resolved_paths: Dict[str, Optional[str]] = {
        "train_csv": None,
        "test_csv": None,
        "sample_submission_csv": None,
        "train_audio_dir": None,
        "test_audio_dir": None,
        "audio_dir": None
    }

    candidate_dirs = []
    for pattern in base_search_paths:
        expanded = glob.glob(pattern)
        candidate_dirs.extend([p for p in expanded if os.path.isdir(p)])
    candidate_dirs = list(dict.fromkeys(candidate_dirs))

    logger.info(f"Scanning {len(candidate_dirs)} candidate directories for dataset assets...")

    # Search for CSVs
    for root_dir in candidate_dirs:
        for root, dirs, files in os.walk(root_dir):
            # Exclude virtual environments and cache directories
            dirs[:] = [d for d in dirs if d not in {".venv", "venv", "env", "site-packages", "__pycache__", ".git"}]
            for file in files:
                f_lower = file.lower()
                f_path = os.path.join(root, file)
                if "site-packages" in f_path:
                    continue

                if "train" in f_lower and f_lower.endswith(".csv") and resolved_paths["train_csv"] is None:
                    resolved_paths["train_csv"] = f_path
                    logger.info(f"Found train CSV: {f_path}")
                elif "test" in f_lower and f_lower.endswith(".csv") and resolved_paths["test_csv"] is None:
                    resolved_paths["test_csv"] = f_path
                    logger.info(f"Found test CSV: {f_path}")
                elif ("sample" in f_lower or "submission" in f_lower) and f_lower.endswith(".csv") and resolved_paths["sample_submission_csv"] is None:
                    resolved_paths["sample_submission_csv"] = f_path
                    logger.info(f"Found sample submission CSV: {f_path}")

    # Search for Audio directories
    for root_dir in candidate_dirs:
        for root, dirs, _ in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in {".venv", "venv", "env", "site-packages", "__pycache__", ".git"}]
            for d in dirs:
                if "site-packages" in root:
                    continue
                d_lower = d.lower()
                d_path = os.path.join(root, d)
                # Check if directory contains wav files
                wav_count = len(glob.glob(os.path.join(d_path, "*.wav")) + glob.glob(os.path.join(d_path, "*.WAV")) + glob.glob(os.path.join(d_path, "*.mp3")))
                if wav_count > 0:
                    if "train" in d_lower and resolved_paths["train_audio_dir"] is None:
                        resolved_paths["train_audio_dir"] = d_path
                        logger.info(f"Found train audio directory ({wav_count} files): {d_path}")
                    elif "test" in d_lower and resolved_paths["test_audio_dir"] is None:
                        resolved_paths["test_audio_dir"] = d_path
                        logger.info(f"Found test audio directory ({wav_count} files): {d_path}")
                    elif resolved_paths["audio_dir"] is None:
                        resolved_paths["audio_dir"] = d_path
                        logger.info(f"Found general audio directory ({wav_count} files): {d_path}")

    # Fallback assignment if single audio folder is used
    if resolved_paths["train_audio_dir"] is None and resolved_paths["audio_dir"] is not None:
        resolved_paths["train_audio_dir"] = resolved_paths["audio_dir"]
    if resolved_paths["test_audio_dir"] is None and resolved_paths["audio_dir"] is not None:
        resolved_paths["test_audio_dir"] = resolved_paths["audio_dir"]

    return resolved_paths


def detect_columns(df: pd.DataFrame, is_train: bool = True) -> Dict[str, Optional[str]]:
    """
    Detect filename, target, and speaker ID columns without hardcoding.
    """
    col_mapping: Dict[str, Optional[str]] = {
        "filename": None,
        "target": None,
        "speaker_id": None
    }

    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    # Filename candidate matches
    filename_candidates = ["filename", "file_name", "audio_filename", "audio_file", "audio_path", "audio", "id", "file_id", "wav_file"]
    for cand in filename_candidates:
        if cand in cols_lower:
            idx = cols_lower.index(cand)
            col_mapping["filename"] = cols[idx]
            break

    # If not found, check string columns with .wav extensions in rows
    if col_mapping["filename"] is None:
        for col in cols:
            if df[col].dtype == object and df[col].astype(str).str.contains(r"\.wav|\.mp3|\.flac", case=False).any():
                col_mapping["filename"] = col
                break

    # Fallback for filename: first column
    if col_mapping["filename"] is None and len(cols) > 0:
        col_mapping["filename"] = cols[0]

    # Target candidate matches (for train set)
    if is_train:
        target_candidates = ["grammar_score", "grammar", "score", "label", "target", "overall_score", "grammar_rating", "rating"]
        for cand in target_candidates:
            if cand in cols_lower:
                idx = cols_lower.index(cand)
                col_mapping["target"] = cols[idx]
                break

        # If not found, look for numeric columns with values between 0 and 5
        if col_mapping["target"] is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != col_mapping["filename"]:
                    val_min = df[col].min()
                    val_max = df[col].max()
                    if 0 <= val_min and val_max <= 10:
                        col_mapping["target"] = col
                        break

    # Speaker ID candidate matches
    speaker_candidates = ["speaker_id", "speaker", "speakerid", "user_id", "subject_id"]
    for cand in speaker_candidates:
        if cand in cols_lower:
            idx = cols_lower.index(cand)
            col_mapping["speaker_id"] = cols[idx]
            break

    return col_mapping


def load_audio_file(
    audio_path: str,
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[Optional[np.ndarray], int]:
    """
    Robust audio loading with format standardization.
    Converts multi-channel to mono, resamples to target_sr if needed, and normalizes amplitude.

    Returns:
        (signal, sample_rate) or (None, target_sr) on error.
    """
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return None, target_sr

    try:
        data, sr = sf.read(audio_path, dtype="float32")

        # Handle multi-channel audio (convert stereo/surround to mono)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if needed using scipy.signal
        if sr != target_sr:
            from scipy import signal
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples)
            sr = target_sr

        # Normalization
        if normalize and len(data) > 0:
            max_val = np.max(np.abs(data))
            if max_val > 1e-6:
                data = data / max_val

        return data.astype(np.float32), sr

    except Exception as e:
        logger.warning(f"Failed to load audio file {audio_path}: {e}")
        return None, target_sr


def inspect_dataframe(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Compute key EDA summary statistics for a dataframe.
    """
    summary = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary
