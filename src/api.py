"API: health, prediction, upload and retraining endpoints."
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from src.prediction import load_class_names, load_session, predict_files
from src.retraining import add_zip_to_training_data, retrain

app = FastAPI(title="Sound Classifier API")

START_TIME = time.time()
state = {"session": load_session(), "class_names": load_class_names()}
training = {"status": "idle", "result": None}


def check_extension(upload, extension):
    if not upload.filename.lower().endswith(extension):
        raise HTTPException(status_code=400, detail=f"Please upload a {extension} file.")


def save_upload(upload, folder, index=0):
    path = Path(folder) / f"{index}_{Path(upload.filename).name}"
    with open(path, "wb") as file:
        shutil.copyfileobj(upload.file, file)
    return path


def predict_uploads(uploads):
    for upload in uploads:
        check_extension(upload, ".wav")
    with tempfile.TemporaryDirectory() as folder:
        paths = [save_upload(u, folder, i) for i, u in enumerate(uploads)]
        try:
            return predict_files(state["session"], state["class_names"], paths)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read the audio file.")


def run_retraining():
    training["status"] = "running"
    try:
        training["result"] = retrain()
        state["session"] = load_session()
        state["class_names"] = load_class_names()
        training["status"] = "done"
    except Exception as error:
        training["result"] = {"error": str(error)}
        training["status"] = "failed"


@app.get("/health")
def health():
    uptime = round(time.time() - START_TIME)
    return {"status": "ok", "uptime_seconds": uptime, "training": training["status"]}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    return predict_uploads([file])[0]


@app.post("/predict-batch")
def predict_batch(files: list[UploadFile] = File(...)):
    return {"predictions": predict_uploads(files)}


@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    check_extension(file, ".zip")
    with tempfile.TemporaryDirectory() as folder:
        zip_path = save_upload(file, folder)
        try:
            count = add_zip_to_training_data(zip_path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
    return {"clips_added": count}


@app.post("/retrain")
def start_retraining(background_tasks: BackgroundTasks):
    if training["status"] == "running":
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    background_tasks.add_task(run_retraining)
    return {"status": "started"}


@app.get("/retrain/status")
def retraining_status():
    return training