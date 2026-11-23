"Retraining: add an uploaded zip to the training data and train a new model."
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch

from src.evaluation import dict_report, save_report
from src.model import export_onnx, predict_labels, save_class_names, train_model
from src.preprocessing import get_class_names, load_folder

TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def extract_zip(zip_path, target_folder):
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)


def find_wav_files(folder):
    files = Path(folder).rglob("*.wav")
    return [f for f in files if "__MACOSX" not in f.parts and not f.name.startswith("._")]


def check_wav_files(wav_files, known_classes):
    if not wav_files:
        raise ValueError("The zip contains no .wav files.")
    unknown = {f.parent.name for f in wav_files} - set(known_classes)
    if unknown:
        raise ValueError(f"Unknown class folders in zip: {sorted(unknown)}")


def copy_into_training(wav_files):
    for path in wav_files:
        shutil.copy(path, TRAIN_DIR / path.parent.name / path.name)
    return len(wav_files)


def add_zip_to_training_data(zip_path):
    known_classes = get_class_names(TRAIN_DIR)
    with tempfile.TemporaryDirectory() as temp_folder:
        extract_zip(zip_path, temp_folder)
        wav_files = find_wav_files(temp_folder)
        check_wav_files(wav_files, known_classes)
        return copy_into_training(wav_files)


def replace_onnx_model(model):
    temp_path = MODELS_DIR / "new_model.onnx"
    export_onnx(model, temp_path)
    os.replace(temp_path, MODELS_DIR / "sound_model.onnx")


def save_new_model(model, class_names):
    replace_onnx_model(model)
    torch.save(model.state_dict(), MODELS_DIR / "sound_model.pt")
    save_class_names(class_names, MODELS_DIR / "class_names.json")


def retrain(epochs=40):
    x_train, y_train, class_names = load_folder(TRAIN_DIR)
    x_test, y_test, _ = load_folder(TEST_DIR)
    model, _ = train_model(x_train, y_train, len(class_names), epochs, x_test, y_test)
    report = dict_report(y_test, predict_labels(model, x_test), class_names)
    save_new_model(model, class_names)
    save_report(report, REPORTS_DIR / "metrics.json")
    return {"accuracy": round(report["accuracy"], 3), "train_clips": len(y_train)}