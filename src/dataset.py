"Dataset: add an uploaded zip of .wav files to the training data."
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.preprocessing import get_class_names

TRAIN_DIR = Path("data/train")


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