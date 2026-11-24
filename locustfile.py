"""Locust load test: many users send a clip to /predict."""
from pathlib import Path

from locust import HttpUser, between, task

CLIP = sorted(Path("data/test/dog").glob("*.wav"))[0]
AUDIO = CLIP.read_bytes()


class SoundUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", files={"file": (CLIP.name, AUDIO, "audio/wav")})