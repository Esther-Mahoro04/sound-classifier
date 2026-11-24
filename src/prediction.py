"Prediction: run the ONNX model on audio files."
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.preprocessing import file_to_spectrogram

MODEL_PATH = Path("models/sound_model.onnx")
CLASS_NAMES_PATH = Path("models/class_names.json")


def load_session(model_path=MODEL_PATH):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def load_class_names(path=CLASS_NAMES_PATH):
    with open(path) as file:
        return json.load(file)


def stack_spectrograms(spectrograms):
    return np.array(spectrograms, dtype=np.float32)[:, np.newaxis, :, :]


def run_model(session, batch):
    return session.run(None, {"spectrogram": batch})[0]


def softmax(scores):
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def describe(probabilities, class_names):
    best = int(probabilities.argmax())
    return {"label": class_names[best], "confidence": round(float(probabilities[best]), 3)}


def predict_files(session, class_names, paths):
    spectrograms = [file_to_spectrogram(path) for path in paths]
    probabilities = softmax(run_model(session, stack_spectrograms(spectrograms)))
    return [describe(row, class_names) for row in probabilities]


def predict_file(session, class_names, path):
    return predict_files(session, class_names, [path])[0]