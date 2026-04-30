"""
checkpoint_service/server.py
-----------------------------
Run this on your always-on machine (laptop or normal VM).
Receives checkpoint files from the spot instance and stores them safely.

Run with:
    python -m checkpoint_service.server
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import uvicorn

app = FastAPI()

SAVE_DIR = "./received_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


@app.post("/upload_checkpoint")
async def upload_checkpoint(file: UploadFile = File(...)):
    """Receives a checkpoint zip from the spot instance and saves it."""
    save_path = os.path.join(SAVE_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print(f"[SERVER] Received and saved: {file.filename}")
    return {"message": "Checkpoint saved successfully", "filename": file.filename}


@app.get("/download_checkpoint/{filename}")
async def download_checkpoint(filename: str):
    """Returns a stored checkpoint zip to the spot instance."""
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return FileResponse(path, filename=filename)


@app.get("/list_checkpoints")
async def list_checkpoints():
    """Lists all stored checkpoints. Open in browser to verify."""
    files = os.listdir(SAVE_DIR)
    return {"checkpoints": files}


@app.get("/")
async def root():
    return {"status": "Checkpoint server is running!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
