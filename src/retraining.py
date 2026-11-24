"""Retraining: fine-tune the saved model on the training folder and swap it in."""
import os
from pathlib import Path

import torch

from src.evaluation import dict_report, save_report
from src.model import (
    export_onnx, load_pretrained, predict_labels, save_class_names, train_model,
)
from src.preprocessing import load_folder

TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "15"))
PRETRAINED_PATH = MODELS_DIR / "sound_model.pt"
FINE_TUNE_LR = 0.0003


def replace_onnx_model(model):
    temp_path = MODELS_DIR / "new_model.onnx"
    export_onnx(model, temp_path)
    os.replace(temp_path, MODELS_DIR / "sound_model.onnx")


def save_new_model(model, class_names):
    replace_onnx_model(model)
    torch.save(model.state_dict(), MODELS_DIR / "sound_model.pt")
    save_class_names(class_names, MODELS_DIR / "class_names.json")


def retrain(epochs=EPOCHS):
    x_train, y_train, class_names = load_folder(TRAIN_DIR)
    x_test, y_test, _ = load_folder(TEST_DIR)
    pretrained = load_pretrained(len(class_names), PRETRAINED_PATH)
    model, _ = train_model(
        x_train, y_train, len(class_names), epochs, x_test, y_test,
        model=pretrained, learning_rate=FINE_TUNE_LR,
    )
    report = dict_report(y_test, predict_labels(model, x_test), class_names)
    save_new_model(model, class_names)
    save_report(report, REPORTS_DIR / "metrics.json")
    return {"accuracy": round(report["accuracy"], 3), "train_clips": len(y_train)}