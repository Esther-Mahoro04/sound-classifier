"""Cleaning: keep the 10 ESC-10 classes and split them into train/test folders."""
import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("raw_esc50")
CSV_PATH = RAW_DIR / "meta" / "esc50.csv"
AUDIO_DIR = RAW_DIR / "audio"
DATA_DIR = Path("data")
TEST_FOLD = 5


def load_metadata():
    return pd.read_csv(CSV_PATH)


def keep_esc10(df):
    return df[df["esc10"]]


def split_train_test(df):
    test = df[df["fold"] == TEST_FOLD]
    train = df[df["fold"] != TEST_FOLD]
    return train, test


def copy_clip(filename, category, split_name):
    folder = DATA_DIR / split_name / category
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(AUDIO_DIR / filename, folder / filename)


def copy_split(df, split_name):
    for row in df.itertuples():
        copy_clip(row.filename, row.category, split_name)


def main():
    df = keep_esc10(load_metadata())
    train, test = split_train_test(df)
    copy_split(train, "train")
    copy_split(test, "test")
    print(f"Done: {len(train)} train clips, {len(test)} test clips")


if __name__ == "__main__":
    main()