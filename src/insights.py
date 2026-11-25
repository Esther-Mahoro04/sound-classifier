"Insights: simple audio features per class, used for the data charts."
from pathlib import Path

import librosa
import pandas as pd

from src.preprocessing import SAMPLE_RATE, load_audio


def clip_features(path):
    audio = load_audio(path)
    rms = librosa.feature.rms(y=audio).mean()
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE).mean()
    return float(rms), float(centroid)


def class_folders(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_dir())


def build_feature_table(folder):
    rows = []
    for class_folder in class_folders(folder):
        for path in class_folder.glob("*.wav"):
            rms, centroid = clip_features(path)
            rows.append({"class": class_folder.name, "rms": rms, "centroid": centroid})
    return pd.DataFrame(rows)