"""
SHL Grammar Scoring Engine - Unit Tests & Sanity Checks
Tests path discovery, audio extraction, NLP feature engineering, model fusion, and metrics.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.preprocessing import discover_dataset_paths, detect_columns, load_audio_file
from src.audio_features import extract_audio_features, _get_default_audio_features
from src.text_features import extract_linguistic_features, extract_grammar_features
from src.model import fuse_features, clip_predictions, BlendRegressor
from src.evaluation import compute_rmse, compute_pearson, create_submission_file


class TestGrammarScoringPipeline(unittest.TestCase):

    def test_column_detection(self):
        df = pd.DataFrame({
            "audio_file": ["sample1.wav", "sample2.wav"],
            "grammar_score": [3.5, 4.0]
        })
        cols = detect_columns(df, is_train=True)
        self.assertEqual(cols["filename"], "audio_file")
        self.assertEqual(cols["target"], "grammar_score")

    def test_linguistic_features(self):
        sample_text = "Um, actually, because technology is evolving rapidly, we need better education, you know."
        feats = extract_linguistic_features(sample_text, duration=5.0)

        self.assertGreater(feats["word_count"], 5)
        self.assertGreater(feats["filler_word_count"], 0)
        self.assertGreater(feats["complex_connective_count"], 0)
        self.assertGreater(feats["lexical_diversity"], 0)
        self.assertGreater(feats["words_per_second"], 0)

    def test_grammar_features(self):
        bad_grammar = "He do not understand why she have a apple."
        feats = extract_grammar_features(bad_grammar)

        self.assertGreater(feats["grammar_error_count"], 0)
        self.assertGreater(feats["grammar_error_rate"], 0)

    def test_rmse_and_pearson(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        rmse = compute_rmse(y_true, y_pred)
        r = compute_pearson(y_true, y_pred)

        self.assertLess(rmse, 0.3)
        self.assertGreater(r, 0.95)

    def test_clipping(self):
        preds = np.array([-1.0, 0.5, 3.0, 5.8, 6.5])
        clipped = clip_predictions(preds)
        self.assertTrue((clipped >= 0.0).all())
        self.assertTrue((clipped <= 5.0).all())

    def test_feature_fusion(self):
        audio_df = pd.DataFrame({
            "filename": ["f1.wav", "f2.wav"],
            "rms_mean": [0.1, 0.2],
            "zcr_mean": [0.05, 0.06]
        })
        text_df = pd.DataFrame({
            "filename": ["f1.wav", "f2.wav"],
            "word_count": [20, 35],
            "grammar_error_rate": [5.0, 2.0]
        })
        target_df = pd.DataFrame({
            "filename": ["f1.wav", "f2.wav"],
            "grammar_score": [3.0, 4.5]
        })

        X, y, cols = fuse_features(audio_df, text_df, target_df, "filename", "grammar_score")
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)
        self.assertIn("word_count", cols)
        self.assertIn("rms_mean", cols)


if __name__ == "__main__":
    unittest.main()

