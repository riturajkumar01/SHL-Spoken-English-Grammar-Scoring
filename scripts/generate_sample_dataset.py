"""
SHL Grammar Scoring Engine - Dataset Simulator for Local Testing & Verification
Creates synthetic audio files and realistic transcripts matching the exact SHL challenge specifications:
- 769 training samples with grammar scores [0.0, 5.0]
- 216 test samples with placeholder labels
- WAV audio files with varied acoustic properties
- Transcripts reflecting CEFR / SHL 5-point grammar rubrics
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf

np.random.seed(42)
random.seed(42)

# Sample transcripts tailored across 5 grammar rubric proficiency levels
TRANSCRIPTS_BY_LEVEL = {
    1: [
        "he go to store yesterday and he buy no foods because money is not have. very difficult life.",
        "i is wanting speak english good but english hard. she do not know why car broken.",
        "they was playing and then accident happen. no anybody help. we is scared.",
        "yesterday i goes work and boss say bad thing. me no happy with this job."
    ],
    2: [
        "yesterday i go to the market with friend and we buy some vegetable. he don't like vegetable so he eat apple.",
        "i think education is very important for all people because it help to get good job and make money.",
        "in my country many people drives car but traffic is very terrible. we need more bus.",
        "last year i visit my uncle house and we was having big dinner. the food is delicious."
    ],
    3: [
        "In my opinion, technology has changed how we communicate every day. Although it makes life faster, sometimes people don't talk face to face.",
        "I have been studying computer science for two years. It is interesting but quite challenging, especially when we do programming assignments.",
        "Many people prefer living in big cities because of better job opportunities, even though the cost of living is noticeably higher.",
        "During my last holiday, I traveled to the mountains with my family. The weather was cold, but the scenery was truly beautiful."
    ],
    4: [
        "Technological advancements have significantly reshaped contemporary communication. While social media facilitates global networking, it often reduces the depth of interpersonal relationships.",
        "Higher education provides students with essential analytical skills and practical knowledge, enabling them to navigate complex professional landscapes effectively.",
        "Urbanization offers substantial economic advantages, yet municipal authorities must proactively address infrastructural strain and environmental degradation.",
        "Sustainable development requires a balanced integration of renewable energy policies, economic incentives, and widespread public awareness."
    ],
    5: [
        "The rapid evolution of artificial intelligence necessitates comprehensive regulatory frameworks to mitigate systemic biases while fostering innovative research across diverse domains.",
        "Notwithstanding the undeniable economic benefits of globalization, regional disparities underscore the imperative for nuanced socioeconomic policies that promote equitable resource distribution.",
        "Effective leadership entails not only visionary strategic foresight but also the cultivation of an inclusive environment wherein interdisciplinary collaboration can flourish.",
        "Environmental sustainability demands a paradigm shift in industrial consumption, integrating circular economy principles and advanced renewable infrastructure."
    ]
}


def create_synthetic_audio(file_path: str, duration_sec: float = 2.0, sr: int = 16000, level: int = 3):
    """
    Generate synthetic WAV audio containing speech-like harmonic formants and pause intervals.
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental frequency based on proficiency/variability
    f0 = 120.0 + (level * 15.0) + np.sin(2 * np.pi * 0.5 * t) * 10.0
    signal = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)

    # Add envelope and pause intervals
    envelope = (np.sin(2 * np.pi * 1.5 * t) > -0.2).astype(float)
    noise = np.random.normal(0, 0.02, len(t))
    audio = (signal * envelope + noise).astype(np.float32)
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    sf.write(file_path, audio, sr)


def generate_mock_dataset(base_dir: str = "data"):
    """
    Generate mock dataset locally for testing and reproducible verification.
    """
    train_audio_dir = os.path.join(base_dir, "train_audio")
    test_audio_dir = os.path.join(base_dir, "test_audio")
    os.makedirs(train_audio_dir, exist_ok=True)
    os.makedirs(test_audio_dir, exist_ok=True)

    print("Generating synthetic dataset (769 train, 216 test)...")

    # Generate Train
    train_records = []
    for i in range(1, 770):
        fname = f"train_speech_{i:04d}.wav"
        fpath = os.path.join(train_audio_dir, fname)

        # Discrete proficiency level 1 to 5 with some continuous noise
        base_level = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]
        score = float(np.clip(base_level + np.random.normal(0, 0.25), 0.5, 5.0))
        transcript_sample = random.choice(TRANSCRIPTS_BY_LEVEL[base_level])

        # Create lightweight audio representation
        create_synthetic_audio(fpath, duration_sec=1.5, sr=16000, level=base_level)

        train_records.append({
            "filename": fname,
            "grammar_score": round(score, 2),
            "transcript_text": transcript_sample
        })

    train_df = pd.DataFrame(train_records)
    train_df[["filename", "grammar_score"]].to_csv(os.path.join(base_dir, "train.csv"), index=False)
    print(f"Train dataset created: {len(train_df)} rows.")

    # Generate Test
    test_records = []
    sample_sub_records = []
    for i in range(1, 217):
        fname = f"test_speech_{i:04d}.wav"
        fpath = os.path.join(test_audio_dir, fname)

        base_level = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]
        create_synthetic_audio(fpath, duration_sec=1.5, sr=16000, level=base_level)

        test_records.append({
            "filename": fname,
            "dummy_label": 0.0
        })
        sample_sub_records.append({
            "filename": fname,
            "grammar_score": 3.0
        })

    test_df = pd.DataFrame(test_records)
    test_df.to_csv(os.path.join(base_dir, "test.csv"), index=False)

    sample_sub_df = pd.DataFrame(sample_sub_records)
    sample_sub_df.to_csv(os.path.join(base_dir, "sample_submission.csv"), index=False)
    print(f"Test dataset created: {len(test_df)} rows. Sample submission created.")


if __name__ == "__main__":
    generate_mock_dataset("data")

