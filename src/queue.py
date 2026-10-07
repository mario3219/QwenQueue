import sqlite3
import time
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image
import os

from src.qwen import *

DB_FILE = "jobs.db"

def connect():
    """Connect to the queue database and create it if necessary."""
    db = sqlite3.connect(DB_FILE)

    db.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            prompt TEXT NOT NULL,
            img_input TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            steps INTEGER NOT NULL
        )
    """)

    db.commit()
    return db

def submit_job(prompt, img_input, width=1024, height=1024, steps=30):
    """Add a new job to the queue."""

    if img_input != "":
        img_path = os.path.abspath(os.path.join(os.getcwd(),"input",img_input))
        if not os.path.exists(img_path):
            print(f"{img_input} doesn't exist")
            return

    db = connect()

    cursor = db.execute(
        """
        INSERT INTO jobs (
            status,
            prompt,
            img_input,
            width,
            height,
            steps
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "queued",
            prompt,
            img_input,
            width,
            height,
            steps
        )
    )

    db.commit()

    job_id = cursor.lastrowid

    db.close()

    print(f"Job {job_id} queued.")

def cancel_job():
    """Cancel the currently running job."""
    db = connect()

    job = db.execute(
        """
        SELECT id
        FROM jobs
        WHERE status = 'running'
        LIMIT 1
        """
    ).fetchone()

    if job is None:
        print("No running job.")
        db.close()
        return

    job_id = job[0]

    db.execute(
        """
        UPDATE jobs
        SET status = 'cancelled'
        WHERE id = ?
        """,
        (job_id,)
    )

    db.commit()
    db.close()

    print(f"Cancellation requested for job {job_id}.")

def is_cancelled(db, job_id):
    """Check whether a job has been cancelled."""
    result = db.execute(
        """
        SELECT status
        FROM jobs
        WHERE id = ?
        """,
        (job_id,)
    ).fetchone()

    if result is None:
        return False

    return result[0] == "cancelled"

def run_job(db, job_id, pipe, prompt, img_input, width, height, steps):

    print(f"Running job {job_id}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {steps}")

    if is_cancelled(db, job_id):
        print(f"Job {job_id} cancelled.")
        return False

    if img_input == "":
        image = qwen(pipe, prompt, width, height, steps)
    else:
        image = qwen_edit(pipe, prompt, img_input, steps)
    
    output_dir = os.path.abspath(os.path.join(os.getcwd(),"output"))
    img_idx = 0
    img_savepath = os.path.join(output_dir,f"{img_idx}.png")
    while (os.path.exists(img_savepath)):
        img_idx += 1
        img_savepath = os.path.join(output_dir,f"{img_idx}.png")
    image.save(img_savepath)
    print(f"Image saved to {img_savepath}")

    return True

def worker():
    """Continuously process queued jobs."""
    if (os.path.exists(os.path.join(os.getcwd(),"jobs.db"))):
        os.remove("jobs.db")
    db = connect()

    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()

    print("Worker started.")

    try:
        while True:

            job = db.execute(
                """
                SELECT id, prompt, img_input, width, height, steps
                FROM jobs
                WHERE status = 'queued'
                ORDER BY id
                LIMIT 1
                """
            ).fetchone()

            if job is None:
                time.sleep(1)
                continue

            job_id, prompt, img_input, width, height, steps = job

            db.execute(
                """
                UPDATE jobs
                SET status = 'running'
                WHERE id = ?
                """,
                (job_id,)
            )

            db.commit()

            completed = run_job(
                db,
                job_id,
                pipe,
                prompt,
                img_input,
                width,
                height,
                steps
            )

            if completed:
                db.execute(
                    """
                    UPDATE jobs
                    SET status = 'finished'
                    WHERE id = ?
                    """,
                    (job_id,)
                )

                db.commit()

    except KeyboardInterrupt:
        print("\nWorker stopped.")

    finally:
        db.close()
