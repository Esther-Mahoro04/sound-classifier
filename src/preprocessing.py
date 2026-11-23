"""Preprocessing: turn audio clips into mel spectrograms the model can read."""
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 64
NUM_SAMPLES = SAMPLE_RATE * DURATION


def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    return librosa.util.fix_length(audio, size=NUM_SAMPLES)


def audio_to_spectrogram(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (mel_db + 80) / 80


def file_to_spectrogram(path):
    return audio_to_spectrogram(load_audio(path))


def get_class_names(folder):
    return sorted(p.name for p in Path(folder).iterdir() if p.is_dir())


def load_folder(folder):
    class_names = get_class_names(folder)
    spectrograms = []
    labels = []
    for label, name in enumerate(class_names):
        for path in sorted((Path(folder) / name).glob("*.wav")):
            spectrograms.append(file_to_spectrogram(path))
            labels.append(label)
    x = np.array(spectrograms, dtype=np.float32)[:, np.newaxis, :, :]
    return x, np.array(labels), class_names