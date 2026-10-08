"""
SHL Spoken English Grammar Scoring Engine - Advanced Acoustic Feature Engineering
Extracts comprehensive acoustic, prosodic, pitch, spectral, energy, and temporal rhythm descriptors.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.preprocessing import load_audio_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_audio_features(
    audio_path: str,
    target_sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 40
) -> Dict[str, float]:
    """
    Extract fixed-length acoustic descriptors capturing prosodic fluency, intonation, and rhythm.

    Features extracted:
    - Temporal Dynamics: Audio duration, active speech duration, phonation time ratio.
    - Energy & Pauses: RMS energy (mean, std, max, min, skewness), silence ratio, pause frequency.
    - Zero-Crossing Dynamics: ZCR (mean, std, peak variability).
    - Spectral Properties: Centroid, Bandwidth, Rolloff (85% & 95%), Spectral Contrast, Spectral Flux.
    - Pitch & Prosody (F0): Fundamental frequency mean, std, range, and voiced frame ratio.
    - Cepstral Descriptors: 13 MFCCs (mean, std, min, max).
    - Chromagram: 12 Chroma pitch classes (mean, std).
    - Mel Spectrogram: Mean and standard deviation of log-mel energy.

    Returns:
        Dict mapping feature name to scalar float.
    """
    feature_dict: Dict[str, float] = {}

    y, sr = load_audio_file(audio_path, target_sr=target_sr, normalize=True)

    if y is None or len(y) == 0:
        return _get_default_audio_features(n_mfcc=n_mfcc, n_mels=n_mels)

    duration = float(len(y) / sr)
    feature_dict["audio_duration"] = duration

    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms

    try:
        import librosa

        # 1. RMS Energy & Silence Analysis
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        feature_dict["rms_mean"] = mean_rms
        feature_dict["rms_std"] = std_rms
        feature_dict["rms_max"] = float(np.max(rms))
        feature_dict["rms_min"] = float(np.min(rms))
        feature_dict["rms_skew"] = float(((rms - mean_rms) ** 3).mean() / (std_rms ** 3 + 1e-6))

        # Phonation Time Ratio & Pause Heuristic
        silence_threshold = 0.10 * (mean_rms + 1e-6)
        silence_mask = rms < silence_threshold
        silence_ratio = float(np.mean(silence_mask))
        feature_dict["silence_ratio"] = silence_ratio
        feature_dict["phonation_time_ratio"] = float(1.0 - silence_ratio)

        # Estimate pause transitions (consecutive silent frames > 200ms = 20 frames)
        pause_count = 0
        current_silent_run = 0
        for is_silent in silence_mask:
            if is_silent:
                current_silent_run += 1
            else:
                if current_silent_run >= 20:
                    pause_count += 1
                current_silent_run = 0
        if current_silent_run >= 20:
            pause_count += 1
        feature_dict["pause_count"] = float(pause_count)
        feature_dict["pause_rate_per_sec"] = float(pause_count / max(duration, 0.1))

        # 2. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        feature_dict["zcr_mean"] = float(np.mean(zcr))
        feature_dict["zcr_std"] = float(np.std(zcr))
        feature_dict["zcr_max"] = float(np.max(zcr))

        # 3. Spectral Descriptors
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        feature_dict["spectral_centroid_mean"] = float(np.mean(cent))
        feature_dict["spectral_centroid_std"] = float(np.std(cent))

        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        feature_dict["spectral_bandwidth_mean"] = float(np.mean(bw))
        feature_dict["spectral_bandwidth_std"] = float(np.std(bw))

        rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]
        feature_dict["spectral_rolloff_mean"] = float(np.mean(rolloff_85))
        feature_dict["spectral_rolloff_std"] = float(np.std(rolloff_85))

        rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.95)[0]
        feature_dict["spectral_rolloff_95_mean"] = float(np.mean(rolloff_95))

        # Spectral Flux (frame-to-frame difference of spectrogram)
        stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        spectral_flux = np.sqrt(np.mean(np.diff(stft, axis=1) ** 2, axis=0))
        feature_dict["spectral_flux_mean"] = float(np.mean(spectral_flux)) if len(spectral_flux) > 0 else 0.0
        feature_dict["spectral_flux_std"] = float(np.std(spectral_flux)) if len(spectral_flux) > 0 else 0.0

        # 4. Pitch & Intonation Dynamics (F0 estimation via YIN)
        try:
            fmin = 65.0   # ~C2
            fmax = 500.0  # ~B4
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=1024, frame_length=2048)
            f0_voiced = f0[(f0 >= fmin) & (f0 <= fmax)]
            if len(f0_voiced) > 5:
                feature_dict["f0_mean"] = float(np.mean(f0_voiced))
                feature_dict["f0_std"] = float(np.std(f0_voiced))
                feature_dict["f0_min"] = float(np.min(f0_voiced))
                feature_dict["f0_max"] = float(np.max(f0_voiced))
                feature_dict["f0_range"] = float(np.max(f0_voiced) - np.min(f0_voiced))
                feature_dict["voiced_ratio"] = float(len(f0_voiced) / max(len(f0), 1))
            else:
                feature_dict["f0_mean"] = 150.0
                feature_dict["f0_std"] = 20.0
                feature_dict["f0_min"] = 100.0
                feature_dict["f0_max"] = 250.0
                feature_dict["f0_range"] = 150.0
                feature_dict["voiced_ratio"] = 0.5
        except Exception:
            feature_dict["f0_mean"] = 150.0
            feature_dict["f0_std"] = 20.0
            feature_dict["f0_min"] = 100.0
            feature_dict["f0_max"] = 250.0
            feature_dict["f0_range"] = 150.0
            feature_dict["voiced_ratio"] = 0.5

        # 5. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        for i in range(n_mfcc):
            feature_dict[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i, :]))
            feature_dict[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i, :]))
            feature_dict[f"mfcc_{i+1}_min"] = float(np.min(mfccs[i, :]))
            feature_dict[f"mfcc_{i+1}_max"] = float(np.max(mfccs[i, :]))

        # 6. Chroma STFT (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        for i in range(chroma.shape[0]):
            feature_dict[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i, :]))
            feature_dict[f"chroma_{i+1}_std"] = float(np.std(chroma[i, :]))

        # 7. Mel Spectrogram Summary
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feature_dict["mel_db_mean"] = float(np.mean(mel_db))
        feature_dict["mel_db_std"] = float(np.std(mel_db))

        # 8. Spectral Contrast
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
            feature_dict["spectral_contrast_mean"] = float(np.mean(contrast))
            feature_dict["spectral_contrast_std"] = float(np.std(contrast))
        except Exception:
            feature_dict["spectral_contrast_mean"] = 0.0
            feature_dict["spectral_contrast_std"] = 0.0

    except ImportError:
        # High-performance NumPy / SciPy fallback
        from scipy import signal

        mean_rms = float(np.sqrt(np.mean(y ** 2)))
        std_rms = float(np.std(np.abs(y)))
        feature_dict["rms_mean"] = mean_rms
        feature_dict["rms_std"] = std_rms
        feature_dict["rms_max"] = float(np.max(np.abs(y)))
        feature_dict["rms_min"] = float(np.min(np.abs(y)))
        feature_dict["rms_skew"] = 0.0

        silence_ratio = float(np.mean(np.abs(y) < (0.05 * np.max(np.abs(y) + 1e-6))))
        feature_dict["silence_ratio"] = silence_ratio
        feature_dict["phonation_time_ratio"] = float(1.0 - silence_ratio)
        feature_dict["pause_count"] = 2.0
        feature_dict["pause_rate_per_sec"] = 0.1

        zcr_val = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0)
        feature_dict["zcr_mean"] = zcr_val
        feature_dict["zcr_std"] = 0.0
        feature_dict["zcr_max"] = zcr_val

        freqs, psd = signal.welch(y, fs=sr, nperseg=frame_length)
        psd_sum = np.sum(psd) + 1e-12
        centroid = float(np.sum(freqs * psd) / psd_sum)
        feature_dict["spectral_centroid_mean"] = centroid
        feature_dict["spectral_centroid_std"] = 0.0
        feature_dict["spectral_bandwidth_mean"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
        feature_dict["spectral_bandwidth_std"] = 0.0
        feature_dict["spectral_rolloff_mean"] = float(freqs[np.searchsorted(np.cumsum(psd) / psd_sum, 0.85)])
        feature_dict["spectral_rolloff_std"] = 0.0
        feature_dict["spectral_rolloff_95_mean"] = float(freqs[np.searchsorted(np.cumsum(psd) / psd_sum, 0.95)])
        feature_dict["spectral_flux_mean"] = 0.0
        feature_dict["spectral_flux_std"] = 0.0
        feature_dict["f0_mean"] = 150.0
        feature_dict["f0_std"] = 20.0
        feature_dict["f0_min"] = 100.0
        feature_dict["f0_max"] = 250.0
        feature_dict["f0_range"] = 150.0
        feature_dict["voiced_ratio"] = 0.5

        for i in range(n_mfcc):
            feature_dict[f"mfcc_{i+1}_mean"] = 0.0
            feature_dict[f"mfcc_{i+1}_std"] = 0.0
            feature_dict[f"mfcc_{i+1}_min"] = 0.0
            feature_dict[f"mfcc_{i+1}_max"] = 0.0
        for i in range(12):
            feature_dict[f"chroma_{i+1}_mean"] = 0.0
            feature_dict[f"chroma_{i+1}_std"] = 0.0
        feature_dict["mel_db_mean"] = 0.0
        feature_dict["mel_db_std"] = 0.0
        feature_dict["spectral_contrast_mean"] = 0.0
        feature_dict["spectral_contrast_std"] = 0.0

    return feature_dict


def _get_default_audio_features(n_mfcc: int = 13, n_mels: int = 40) -> Dict[str, float]:
    """Provide default zero-filled acoustic feature dict."""
    feats: Dict[str, float] = {
        "audio_duration": 0.0,
        "rms_mean": 0.0,
        "rms_std": 0.0,
        "rms_max": 0.0,
        "rms_min": 0.0,
        "rms_skew": 0.0,
        "silence_ratio": 0.0,
        "phonation_time_ratio": 0.0,
        "pause_count": 0.0,
        "pause_rate_per_sec": 0.0,
        "zcr_mean": 0.0,
        "zcr_std": 0.0,
        "zcr_max": 0.0,
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_std": 0.0,
        "spectral_bandwidth_mean": 0.0,
        "spectral_bandwidth_std": 0.0,
        "spectral_rolloff_mean": 0.0,
        "spectral_rolloff_std": 0.0,
        "spectral_rolloff_95_mean": 0.0,
        "spectral_flux_mean": 0.0,
        "spectral_flux_std": 0.0,
        "f0_mean": 0.0,
        "f0_std": 0.0,
        "f0_min": 0.0,
        "f0_max": 0.0,
        "f0_range": 0.0,
        "voiced_ratio": 0.0,
        "mel_db_mean": 0.0,
        "mel_db_std": 0.0,
        "spectral_contrast_mean": 0.0,
        "spectral_contrast_std": 0.0
    }
    for i in range(n_mfcc):
        feats[f"mfcc_{i+1}_mean"] = 0.0
        feats[f"mfcc_{i+1}_std"] = 0.0
        feats[f"mfcc_{i+1}_min"] = 0.0
        feats[f"mfcc_{i+1}_max"] = 0.0
    for i in range(12):
        feats[f"chroma_{i+1}_mean"] = 0.0
        feats[f"chroma_{i+1}_std"] = 0.0
    return feats


def extract_audio_features_dataset(
    df: pd.DataFrame,
    audio_dir: str,
    filename_col: str,
    cache_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract acoustic features for an entire dataset with persistent disk caching.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached audio features from: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info(f"Extracting enhanced acoustic features for {len(df)} files from {audio_dir}...")
    records = []

    for idx, row in df.iterrows():
        fname = str(row[filename_col])
        if os.path.isabs(fname) and os.path.exists(fname):
            full_path = fname
        elif os.path.exists(os.path.join(audio_dir, fname)):
            full_path = os.path.join(audio_dir, fname)
        elif not fname.lower().endswith(".wav") and os.path.exists(os.path.join(audio_dir, f"{fname}.wav")):
            full_path = os.path.join(audio_dir, f"{fname}.wav")
        else:
            full_path = os.path.join(audio_dir, fname)

        feats = extract_audio_features(full_path)
        feats[filename_col] = row[filename_col]
        records.append(feats)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            logger.info(f"Extracted acoustic features for {idx + 1}/{len(df)} samples.")

    features_df = pd.DataFrame(records)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        features_df.to_csv(cache_path, index=False)
        logger.info(f"Saved audio features cache to: {cache_path}")

    return features_df
