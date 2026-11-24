"Streamlit UI: samples, prediction, retraining and status."
import datetime
import json
import os
from pathlib import Path
import io
import tempfile
import zipfile

import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from src.preprocessing import SAMPLE_RATE, audio_to_spectrogram, load_audio

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
METRICS_PATH = Path("reports/metrics.json")


def call_api(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
    except requests.RequestException:
        st.error("Cannot reach the API. Is it running?")
        return None
    if response.status_code != 200:
        st.error(response.json().get("detail", "The request failed."))
        return None
    return response.json()


def class_folders(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_dir())


def spectrogram_figure(path):
    spectrogram = audio_to_spectrogram(load_audio(path))
    figure, axis = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(
        spectrogram, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=axis
    )
    return figure


def show_about():
    st.write(
        "This app listens to a 5-second clip and tells you which of 10 "
        "environmental sounds it is, such as a dog, rain or a chainsaw."
    )
    if not TEST_DIR.exists():
        st.info("Sample clips are not available here.")
        return
    name = st.selectbox("Pick a sound class", [p.name for p in class_folders(TEST_DIR)])
    clip = sorted((TEST_DIR / name).glob("*.wav"))[0]
    st.audio(str(clip))
    st.pyplot(spectrogram_figure(clip))


def show_single_prediction():
    clip = st.file_uploader("Upload one .wav clip", type="wav", key="single")
    if clip and st.button("Predict"):
        st.audio(clip)
        result = call_api("POST", "/predict", files={"file": (clip.name, clip.getvalue())})
        if result:
            st.success(f"{result['label']} ({result['confidence']:.0%} confident)")


def show_batch_prediction():
    clips = st.file_uploader(
        "Upload several .wav clips", type="wav", accept_multiple_files=True, key="batch"
    )
    if clips and st.button("Predict all"):
        files = [("files", (c.name, c.getvalue())) for c in clips]
        result = call_api("POST", "/predict-batch", files=files)
        if result:
            table = pd.DataFrame(result["predictions"])
            table.insert(0, "file", [c.name for c in clips])
            st.dataframe(table)


def show_predict():
    st.subheader("One clip")
    show_single_prediction()
    st.subheader("Many clips")
    show_batch_prediction()


def show_training_status():
    status = call_api("GET", "/retrain/status")
    if status:
        st.write(f"Training status: **{status['status']}**")
        if status["result"]:
            st.json(status["result"])


def show_pipeline_steps():
    st.markdown(
        "**How retraining works**\n"
        "1. Upload a zip of new clips. They are saved and logged in the database.\n"
        "2. Each clip is cut to 5 seconds and turned into a mel spectrogram.\n"
        "3. The current model is loaded as a starting point (pre-trained).\n"
        "4. It is fine-tuned on the old and new clips, then replaces the old model."
    )


def preview_first_clip(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n]
        if not names:
            return
        with tempfile.TemporaryDirectory() as folder:
            path = archive.extract(names[0], folder)
            st.write(f"Preprocessing preview of {names[0]}:")
            st.pyplot(spectrogram_figure(path))


def show_history():
    history = call_api("GET", "/history")
    if history:
        st.write("Uploads saved in the database")
        st.dataframe(pd.DataFrame(history["uploads"]))
        st.write("Retraining runs")
        st.dataframe(pd.DataFrame(history["retrainings"]))


def show_retrain():
    show_pipeline_steps()
    zip_file = st.file_uploader("Dataset zip", type="zip")
    if zip_file and st.button("Upload dataset"):
        files = {"file": (zip_file.name, zip_file.getvalue())}
        result = call_api("POST", "/upload", files=files)
        if result:
            st.success(f"{result['clips_added']} clips saved to the training data and logged.")
            preview_first_clip(zip_file.getvalue())
    if st.button("Retrain model") and call_api("POST", "/retrain"):
        st.info("Retraining started. Press Refresh status in a minute.")
    st.button("Refresh status")
    show_training_status()
    show_history()


def show_class_counts():
    counts = {p.name: len(list(p.glob("*.wav"))) for p in class_folders(TRAIN_DIR)}
    st.write("Training clips per class")
    st.bar_chart(pd.Series(counts))


def show_f1_scores():
    report = json.loads(METRICS_PATH.read_text())
    names = [p.name for p in class_folders(TRAIN_DIR)]
    st.write("F1 score per class on the test clips")
    st.bar_chart(pd.Series({name: report[name]["f1-score"] for name in names}))


def show_status():
    health = call_api("GET", "/health")
    if health:
        uptime = str(datetime.timedelta(seconds=health["uptime_seconds"]))
        first, second, third = st.columns(3)
        first.metric("API", health["status"])
        second.metric("Uptime", uptime)
        third.metric("Training", health["training"])
    if TRAIN_DIR.exists() and METRICS_PATH.exists():
        show_class_counts()
        show_f1_scores()


def main():
    st.title("Environmental Sound Classifier")
    about, predict, retrain, status = st.tabs(["About", "Predict", "Retrain", "Status"])
    with about:
        show_about()
    with predict:
        show_predict()
    with retrain:
        show_retrain()
    with status:
        show_status()


main()