"""
SHL Spoken English Grammar Scoring Engine - Advanced Text & Linguistic Engineering
Extracts multidimensional lexical richness, syntactic complexity, grammar error taxonomy, and disfluency profiles.
"""

import os
import re
import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Disfluency Taxonomy
FILLED_PAUSES = {"um", "uh", "er", "ah", "eh", "hmm", "em"}
DISCURSIVE_FILLERS = {
    "like", "basically", "actually", "literally", "honestly", "well",
    "right", "okay", "yeah", "anyway", "totally", "seriously"
}
FILLER_PHRASES = [
    "you know", "i mean", "sort of", "kind of", "as in", "you see",
    "what i mean is", "how to say", "and all that"
]

# Syntactic Complexity Connectives & Subordination
SUBORDINATING_CONNECTIVES = {
    "because", "although", "though", "even though", "since", "while", "whereas",
    "if", "unless", "provided that", "assuming", "despite", "in spite of",
    "as long as", "in order that", "so that", "whether"
}
COORDINATING_CONNECTIVES = {"and", "but", "or", "nor", "for", "yet", "so"}
TRANSITIONAL_CONNECTIVES = {
    "furthermore", "moreover", "nevertheless", "nonetheless", "consequently",
    "therefore", "thus", "hence", "accordingly", "meanwhile", "in addition",
    "on the other hand", "in contrast", "as a result"
}

# Modal Auxiliaries (Indicators of epistemic hedging & syntactic nuance)
MODAL_VERBS = {"can", "could", "may", "might", "shall", "should", "will", "would", "must"}

# Grammar Error Taxonomy - Subject-Verb Agreement (SVA)
SVA_PATTERNS = [
    r"\b(he|she|it|this|that|everyone|everybody|anyone|nobody)\s+(do|have|are|were|go|say|make|take|know|think|see)\b",
    r"\b(i|we|they|these|those|many|people)\s+(is|was|has|does|goes|says|makes|takes|knows|thinks|sees)\b",
    r"\b(he|she|it|this|that)\s+don't\b",
    r"\b(they|we|you|these|those)\s+doesn't\b",
    r"\bthere\s+is\s+(many|several|two|three|four|few|multiple|people|children|cars|problems|reasons)\b",
    r"\bthere\s+are\s+(a|an|one|each|every)\b"
]

# Determiner / Article Misuse
ARTICLE_PATTERNS = [
    r"\ba\s+(apple|orange|egg|elephant|hour|honest|idea|option|issue|event|item|opportunity|error|accident|answer|example|organization|industry|expert|individual)\b",
    r"\ban\s+(car|house|dog|cat|man|woman|book|tree|computer|person|university|european|one|uniform|table|friend|school|student|government|company)\b",
    r"\bmany\s+(information|advice|furniture|luggage|homework|equipment|money|research|evidence)\b",
    r"\bmuch\s+(people|students|cars|problems|books|countries|options|years|items)\b"
]

# Preposition Collocation Errors
PREPOSITION_ERRORS = [
    r"\bdepend\s+(of|at|to|in)\b",          # Correct: depend on
    r"\binterested\s+(for|at|to|on)\b",      # Correct: interested in
    r"\bdiscuss\s+about\b",                  # Correct: discuss
    r"\bmarried\s+with\b",                   # Correct: married to
    r"\bgood\s+in\s+(english|math|science|speaking)\b", # Correct: good at
    r"\bpay\s+attention\s+(for|at|on)\b",    # Correct: pay attention to
    r"\bprefer\s+\w+\s+than\b",             # Correct: prefer ... to
    r"\blisten\s+(the|music|people|me)\b"    # Correct: listen to
]

# Double Negatives & Structural Irregularities
DOUBLE_NEGATIVE_PATTERNS = [
    r"\b(don't|didn't|can't|cannot|won't|couldn't|never|not)\s+\w*\s*(no|nothing|nobody|never|nowhere|neither)\b"
]


def transcribe_audio_file(
    audio_path: str,
    whisper_model: Optional[Any] = None,
    model_name: str = "base"
) -> str:
    """
    Transcribe a single audio file to text using offline Whisper ASR.
    Falls back gracefully if Whisper is not installed.
    """
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return ""

    try:
        import whisper
        if whisper_model is None:
            whisper_model = whisper.load_model(model_name)
        result = whisper_model.transcribe(audio_path, language="en", fp16=False)
        return str(result.get("text", "")).strip()

    except ImportError:
        try:
            from transformers import pipeline
            asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
            res = asr_pipeline(audio_path)
            return str(res.get("text", "")).strip()
        except Exception:
            return ""
    except Exception as e:
        logger.warning(f"ASR transcription failed on {audio_path}: {e}")
        return ""


def transcribe_dataset(
    df: pd.DataFrame,
    audio_dir: str,
    filename_col: str,
    cache_path: Optional[str] = None,
    whisper_model_name: str = "base"
) -> pd.DataFrame:
    """
    Batch transcription with persistent caching to transcripts.csv.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading existing transcripts from cache: {cache_path}")
        cached_df = pd.read_csv(cache_path)
        if filename_col in cached_df.columns and "transcript" in cached_df.columns:
            return cached_df

    logger.info(f"Transcribing {len(df)} speech audio files...")
    records = []

    whisper_model = None
    try:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(whisper_model_name, device=device)
        logger.info(f"Loaded Whisper model '{whisper_model_name}' on {device}.")
    except Exception:
        pass

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

        text = transcribe_audio_file(full_path, whisper_model=whisper_model)
        records.append({
            filename_col: row[filename_col],
            "transcript": text
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            logger.info(f"Transcribed {idx + 1}/{len(df)} files.")

    transcripts_df = pd.DataFrame(records)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        transcripts_df.to_csv(cache_path, index=False)
        logger.info(f"Saved transcripts to: {cache_path}")

    return transcripts_df


def extract_linguistic_features(text: str, duration: Optional[float] = None) -> Dict[str, float]:
    """
    Extract comprehensive lexical, structural, syntactic, and disfluency descriptors.
    """
    features: Dict[str, float] = {}

    if not isinstance(text, str) or len(text.strip()) == 0:
        return _get_default_linguistic_features()

    cleaned_text = text.strip()
    words = re.findall(r"\b[A-Za-z']+\b", cleaned_text)
    words_lower = [w.lower() for w in words]
    total_words = len(words_lower)
    unique_words = set(words_lower)
    num_unique = len(unique_words)

    # 1. Structural Counts
    features["char_count"] = float(len(cleaned_text))
    features["word_count"] = float(total_words)
    features["unique_word_count"] = float(num_unique)

    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned_text) if len(s.strip()) > 0]
    num_sentences = max(len(sentences), 1)
    features["sentence_count"] = float(num_sentences)

    if total_words > 0:
        features["avg_word_length"] = float(np.mean([len(w) for w in words]))
        features["avg_sentence_length"] = float(total_words / num_sentences)
        features["long_word_ratio"] = float(sum(1 for w in words if len(w) >= 7) / total_words)

        # Advanced Lexical Diversity Indices
        # Type-Token Ratio
        ttr = num_unique / total_words
        features["lexical_diversity"] = float(ttr)
        # Guiraud's Index R = V / sqrt(N)
        features["guiraud_index"] = float(num_unique / math.sqrt(total_words))
        # Herdan's C = log(V) / log(N)
        features["herdan_c"] = float(math.log(num_unique + 1e-6) / math.log(total_words + 1e-6))
        # Dugast's U = log^2(N) / (log(N) - log(V))
        denom = math.log(total_words + 1e-6) - math.log(num_unique + 1e-6)
        features["dugast_u"] = float((math.log(total_words + 1e-6) ** 2) / (denom if abs(denom) > 1e-4 else 1e-4))
        # Maas Index a^2 = (log(N) - log(V)) / log^2(N)
        features["maas_index"] = float(denom / ((math.log(total_words + 1e-6) ** 2) + 1e-6))
    else:
        features["avg_word_length"] = 0.0
        features["avg_sentence_length"] = 0.0
        features["long_word_ratio"] = 0.0
        features["lexical_diversity"] = 0.0
        features["guiraud_index"] = 0.0
        features["herdan_c"] = 0.0
        features["dugast_u"] = 0.0
        features["maas_index"] = 0.0

    # 2. Speech Rate & Temporal Dynamics
    if duration and duration > 0 and total_words > 0:
        features["words_per_second"] = float(total_words / duration)
        features["chars_per_second"] = float(len(cleaned_text) / duration)
        # Estimated syllable rate (~1.4 syllables per English word)
        features["syllables_per_second"] = float((total_words * 1.4) / duration)
    else:
        features["words_per_second"] = 0.0
        features["chars_per_second"] = 0.0
        features["syllables_per_second"] = 0.0

    # 3. Disfluency Profiling
    filled_pauses_count = sum(1 for w in words_lower if w in FILLED_PAUSES)
    discursive_fillers_count = sum(1 for w in words_lower if w in DISCURSIVE_FILLERS)
    phrase_fillers_count = sum(len(re.findall(rf"\b{re.escape(p)}\b", cleaned_text.lower())) for p in FILLER_PHRASES)
    total_fillers = filled_pauses_count + discursive_fillers_count + phrase_fillers_count

    features["filled_pause_count"] = float(filled_pauses_count)
    features["discursive_filler_count"] = float(discursive_fillers_count)
    features["filler_word_count"] = float(total_fillers)
    features["filler_ratio"] = float(total_fillers / max(total_words, 1))

    # Repetition Detection
    repeated_words = 0
    for i in range(len(words_lower) - 1):
        if words_lower[i] == words_lower[i + 1]:
            repeated_words += 1
    features["repeated_word_count"] = float(repeated_words)
    features["repetition_ratio"] = float(repeated_words / max(total_words, 1))

    # 4. Syntactic Architecture & Subordination
    sub_count = sum(1 for w in words_lower if w in SUBORDINATING_CONNECTIVES)
    coord_count = sum(1 for w in words_lower if w in COORDINATING_CONNECTIVES)
    trans_count = sum(1 for w in words_lower if w in TRANSITIONAL_CONNECTIVES)
    for phrase in ["even though", "in spite of", "as a result", "in order to", "on the other hand", "as long as"]:
        if phrase in cleaned_text.lower():
            sub_count += 1
            trans_count += 1

    features["subordinate_connective_count"] = float(sub_count)
    features["coordinate_connective_count"] = float(coord_count)
    features["transitional_connective_count"] = float(trans_count)
    features["complex_connective_count"] = float(sub_count + trans_count)
    features["complex_connective_ratio"] = float((sub_count + trans_count) / max(total_words, 1))
    features["subordination_ratio"] = float(sub_count / max(coord_count, 1))

    # Modal Auxiliaries
    modal_count = sum(1 for w in words_lower if w in MODAL_VERBS)
    features["modal_verb_count"] = float(modal_count)
    features["modal_verb_ratio"] = float(modal_count / max(total_words, 1))

    # Passive Voice heuristic ("is/was/were/been + ed/en")
    passive_matches = len(re.findall(r"\b(is|are|was|were|been|being)\s+\w+(ed|en)\b", cleaned_text.lower()))
    features["passive_voice_count"] = float(passive_matches)
    features["passive_voice_ratio"] = float(passive_matches / max(num_sentences, 1))

    return features


def extract_grammar_features(text: str) -> Dict[str, float]:
    """
    Extract multi-category grammar error metrics, syntactic violations, and rubric baseline score.
    """
    features: Dict[str, float] = {}

    if not isinstance(text, str) or len(text.strip()) == 0:
        return _get_default_grammar_features()

    cleaned_text = text.strip()
    words = re.findall(r"\b[A-Za-z']+\b", cleaned_text)
    total_words = max(len(words), 1)
    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned_text) if len(s.strip()) > 0]
    num_sentences = max(len(sentences), 1)

    sva_errors = 0
    article_errors = 0
    prep_errors = 0
    double_negatives = 0
    structural_errors = 0

    # 1. Subject-Verb Agreement Checks
    for pat in SVA_PATTERNS:
        sva_errors += len(re.findall(pat, cleaned_text, flags=re.IGNORECASE))

    # 2. Article Misuse Checks
    for pat in ARTICLE_PATTERNS:
        article_errors += len(re.findall(pat, cleaned_text, flags=re.IGNORECASE))

    # 3. Preposition Collocation Checks
    for pat in PREPOSITION_ERRORS:
        prep_errors += len(re.findall(pat, cleaned_text, flags=re.IGNORECASE))

    # 4. Double Negatives
    for pat in DOUBLE_NEGATIVE_PATTERNS:
        double_negatives += len(re.findall(pat, cleaned_text, flags=re.IGNORECASE))

    # 5. Structural / Sentence Length Anomalies
    for s in sentences:
        if s and s[0].islower():
            structural_errors += 0.5
        words_in_sent = len(s.split())
        if words_in_sent > 45:  # Run-on sentence
            structural_errors += 1.0
        elif words_in_sent < 3 and len(sentences) > 1:  # Fragment
            structural_errors += 0.5

    total_grammar_errors = sva_errors + article_errors + prep_errors + double_negatives + structural_errors

    features["sva_error_count"] = float(sva_errors)
    features["article_error_count"] = float(article_errors)
    features["preposition_error_count"] = float(prep_errors)
    features["double_negative_count"] = float(double_negatives)
    features["structural_error_count"] = float(structural_errors)
    features["grammar_error_count"] = float(total_grammar_errors)
    features["spelling_error_count"] = 0.0

    # Error Ratios
    features["grammar_error_rate"] = float((total_grammar_errors / total_words) * 100.0)  # Errors per 100 words
    features["sentence_error_rate"] = float(total_grammar_errors / num_sentences)
    
    # Rule-calibrated baseline score [0, 5]
    rubric_baseline = 5.0 - (total_grammar_errors / max(num_sentences, 1)) * 0.9
    features["grammar_accuracy_score"] = float(np.clip(rubric_baseline, 0.5, 5.0))

    return features


def _get_default_linguistic_features() -> Dict[str, float]:
    return {
        "char_count": 0.0,
        "word_count": 0.0,
        "unique_word_count": 0.0,
        "sentence_count": 0.0,
        "avg_word_length": 0.0,
        "avg_sentence_length": 0.0,
        "long_word_ratio": 0.0,
        "lexical_diversity": 0.0,
        "guiraud_index": 0.0,
        "herdan_c": 0.0,
        "dugast_u": 0.0,
        "maas_index": 0.0,
        "words_per_second": 0.0,
        "chars_per_second": 0.0,
        "syllables_per_second": 0.0,
        "filled_pause_count": 0.0,
        "discursive_filler_count": 0.0,
        "filler_word_count": 0.0,
        "filler_ratio": 0.0,
        "repeated_word_count": 0.0,
        "repetition_ratio": 0.0,
        "subordinate_connective_count": 0.0,
        "coordinate_connective_count": 0.0,
        "transitional_connective_count": 0.0,
        "complex_connective_count": 0.0,
        "complex_connective_ratio": 0.0,
        "subordination_ratio": 0.0,
        "modal_verb_count": 0.0,
        "modal_verb_ratio": 0.0,
        "passive_voice_count": 0.0,
        "passive_voice_ratio": 0.0
    }


def _get_default_grammar_features() -> Dict[str, float]:
    return {
        "sva_error_count": 0.0,
        "article_error_count": 0.0,
        "preposition_error_count": 0.0,
        "double_negative_count": 0.0,
        "structural_error_count": 0.0,
        "grammar_error_count": 0.0,
        "spelling_error_count": 0.0,
        "grammar_error_rate": 0.0,
        "sentence_error_rate": 0.0,
        "grammar_accuracy_score": 0.0
    }


def extract_all_text_features(
    transcripts_df: pd.DataFrame,
    filename_col: str,
    audio_durations: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Extract complete linguistic + grammar feature matrix from transcripts.
    """
    records = []
    for _, row in transcripts_df.iterrows():
        fname = row[filename_col]
        text = str(row["transcript"]) if pd.notnull(row.get("transcript")) else ""
        dur = audio_durations.get(fname, None) if audio_durations else None

        ling_feats = extract_linguistic_features(text, duration=dur)
        gram_feats = extract_grammar_features(text)

        combined = {**ling_feats, **gram_feats}
        combined[filename_col] = fname
        records.append(combined)

    return pd.DataFrame(records)
