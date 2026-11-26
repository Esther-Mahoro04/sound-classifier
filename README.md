# Environmental Sound Classifier

Classifies 5-second audio clips into 10 environmental sounds (dog, rooster, rain, sea waves, crying baby, clock tick, sneezing, helicopter, chainsaw, crackling fire) using mel spectrograms and a small CNN.

- **Video demo:** https://youtu.be/sqgn3LbzVt4
- **Live app (Azure VM):** http://102.37.14.131:8501
- **Live API docs:** http://102.37.14.131:8000/docs

The live links use http because i have no certificate)

## Project description
Data: the ESC-10 subset of ESC-50 (400 clips, 320 train / 80 test). Each clip is resampled, cut to 5 seconds and turned into a 64-band mel spectrogram. A CNN (3 convolution blocks with batch normalization, dropout, Adam, cosine learning-rate schedule) is trained in `notebook/sound_classifier.ipynb`, then exported to ONNX for fast prediction. Test accuracy is about 0.81 (macro F1 0.81).

Model files: `models/sound_model.pt` (PyTorch weights, used as the starting point for retraining), `models/sound_model.onnx` (used for prediction) and `models/class_names.json`.

## Features
- Predict one clip or many clips (UI and API)
- Visualizations with written interpretations (Insights tab)
- Upload a zip of new clips (one folder per class, `.wav` files inside), stored and logged in SQLite
- Retrain: loads the saved model and fine-tunes it on the old and new clips, then replaces it
- Status tab: API health, uptime, class counts, F1 per class

## API
| Endpoint | Purpose |
|---|---|
| GET /health | Status and uptime |
| POST /predict | One `.wav` file |
| POST /predict-batch | Several `.wav` files |
| POST /upload | Zip of new training clips |y|
| POST /retrain | Start retraining |
| GET /retrain/status | Training status |
| GET /history | Uploads and retraining runs |

A ready-made test zip is in `samples/sample_upload.zip`.

## Setup
Local (Python 3.12):

    git clone https://github.com/Esther-Mahoro04/sound-classifier.git
    cd sound-classifier
    pip install -r requirements.txt
    uvicorn src.api:app --port 8000
    streamlit run app.py

Docker (API behind nginx, plus the UI):

    docker compose up -d --build --scale api=2

Load test:

    pip install -r requirements-loadtest.txt
    locust -f locustfile.py --host http://127.0.0.1:8000 --users 50 --spawn-rate 10 --run-time 60s --headless

## Flood test results (Locust, POST /predict)
| Setup | Containers | Failures | Median (ms) | Requests/s |
|---|---|---|---|---|
| Laptop, default threads | 1 / 2 / 3 | 0 | 960 / 1100 / 1200 | 35.6 / 30.9 / 29.8 |
| Laptop, 1 thread per container | 1 / 2 / 3 | 0 | 420 / 970 / 790 | 66.2 / 37.7 / 37.0 |
| Azure VM (2 vCPUs), 10 users | 1 | 0 | 2300 | 2.67 |

On the laptop, extra containers did not help because they share the same CPU cores. The Azure run was tested over the internet from Kigali, so its times include network travel and are not comparable with the laptop runs. Each configuration was run once. Raw results are in `reports/`.

## Deployment
Azure for Students VM (Standard B2als_v2, 2 vCPUs, 4 GiB, South Africa North) running `docker compose`.

## Limitations
- Retraining replaces the model files on that server only. With several API containers, only the one that ran the retraining switches to the new model immediately.
- Uploaded zips must use the 10 existing class folder names.
- The Azure student credit is limited, so the VM may be stopped after grading.

## Repository structure
    notebook/   src/   data/   models/   reports/   samples/
    app.py   Dockerfile   docker-compose.yml   nginx.conf   locustfile.py   requirements.txt
