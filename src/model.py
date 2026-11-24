"""Model: a small CNN, plus training, evaluation and saving."""
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import N_MELS

TIME_STEPS = 216
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def build_model(num_classes):
    return nn.Sequential(
        conv_block(1, 16),
        conv_block(16, 32),
        conv_block(32, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )


def make_loader(x, y, shuffle):
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def predict_labels(model, x):
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x))
    return scores.argmax(dim=1).numpy()


def accuracy(model, x, y):
    return float((predict_labels(model, x) == y).mean())


def make_record(model, loss, x_train, y_train, x_val, y_val):
    record = {"loss": loss, "train_acc": accuracy(model, x_train, y_train)}
    if x_val is not None:
        record["val_acc"] = accuracy(model, x_val, y_val)
    return record


def print_record(epoch, record):
    numbers = ", ".join(f"{name}={value:.3f}" for name, value in record.items())
    print(f"epoch {epoch}: {numbers}")


def train_model(
    x_train, y_train, num_classes, epochs=30, x_val=None, y_val=None,
    model=None, learning_rate=LEARNING_RATE,
):
    if model is None:
        model = build_model(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(x_train, y_train, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, loss_fn)
        scheduler.step()
        record = make_record(model, loss, x_train, y_train, x_val, y_val)
        history.append(record)
        print_record(epoch, record)
    return model, history


def load_pretrained(num_classes, path):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def export_onnx(model, path):
    model.eval()
    example = torch.zeros(1, 1, N_MELS, TIME_STEPS)
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["spectrogram"],
        output_names=["scores"],
        dynamic_axes={"spectrogram": {0: "batch"}, "scores": {0: "batch"}},
        external_data=False,
    )


def save_class_names(class_names, path):
    with open(path, "w") as file:
        json.dump(class_names, file)