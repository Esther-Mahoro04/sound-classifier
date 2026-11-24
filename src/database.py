"Database: a small SQLite log of uploads and retraining runs."
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/app.db")
UPLOAD_COLUMNS = ("filename", "clips_added", "uploaded_at")
RETRAINING_COLUMNS = ("accuracy", "train_clips", "finished_at")


def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS uploads "
        "(id INTEGER PRIMARY KEY, filename TEXT, clips_added INTEGER, uploaded_at TEXT)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retrainings "
        "(id INTEGER PRIMARY KEY, accuracy REAL, train_clips INTEGER, finished_at TEXT)"
    )
    return connection


def run(query, params=()):
    connection = connect()
    rows = connection.execute(query, params).fetchall()
    connection.commit()
    connection.close()
    return rows


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_upload(filename, clips_added):
    run(
        "INSERT INTO uploads (filename, clips_added, uploaded_at) VALUES (?, ?, ?)",
        (filename, clips_added, now()),
    )


def log_retraining(accuracy, train_clips):
    run(
        "INSERT INTO retrainings (accuracy, train_clips, finished_at) VALUES (?, ?, ?)",
        (accuracy, train_clips, now()),
    )


def list_uploads():
    rows = run("SELECT filename, clips_added, uploaded_at FROM uploads ORDER BY id DESC LIMIT 20")
    return [dict(zip(UPLOAD_COLUMNS, row)) for row in rows]


def list_retrainings():
    rows = run("SELECT accuracy, train_clips, finished_at FROM retrainings ORDER BY id DESC LIMIT 20")
    return [dict(zip(RETRAINING_COLUMNS, row)) for row in rows]