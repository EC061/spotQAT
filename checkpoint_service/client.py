"""
checkpoint_service/client.py
----------------------------
Runs on the spot instance alongside train_and_qat.py.
Sends checkpoints to the external server and downloads them back on restart.

Usage in train_and_qat.py:
    from checkpoint_service import send_checkpoint, download_checkpoint
"""

import os
import zipfile
import requests

# ⚠️ Change this to your server's IP address before running experiments
SERVER_URL = "http://localhost:8000"


def zip_folder(folder_path: str, zip_name: str) -> str:
    """Zips a folder and returns the zip file path."""
    zip_path = zip_name if zip_name.endswith(".zip") else zip_name + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=os.path.dirname(folder_path))
                zf.write(full_path, arcname)
    print(f"[CLIENT] Zipped checkpoint to: {zip_path}")
    return zip_path


def unzip_file(zip_path: str, extract_to: str):
    """Unzips a file to a folder."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[CLIENT] Unzipped to: {extract_to}")


def send_checkpoint(folder_path: str, checkpoint_name: str):
    """
    Zips the checkpoint folder and uploads it to the external server.
    Call this after saving a checkpoint locally during training.
    """
    zip_path = zip_folder(folder_path, checkpoint_name)
    filename = os.path.basename(zip_path)

    with open(zip_path, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/upload_checkpoint",
            files={"file": (filename, f, "application/zip")}
        )

    if response.status_code == 200:
        print(f"[CLIENT] Successfully sent checkpoint: {filename}")
    else:
        print(f"[CLIENT] ERROR sending checkpoint: {response.text}")

    os.remove(zip_path)


def download_checkpoint(checkpoint_name: str, extract_to: str = ".") -> bool:
    """
    Downloads a checkpoint zip from the server and unzips it.
    Call this at the start of training to resume if a checkpoint exists on the server.
    Returns True if successful, False if no checkpoint was found.
    """
    filename = checkpoint_name if checkpoint_name.endswith(".zip") else checkpoint_name + ".zip"
    response = requests.get(f"{SERVER_URL}/download_checkpoint/{filename}", stream=True)

    if response.status_code == 200:
        zip_path = filename
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[CLIENT] Downloaded checkpoint: {filename}")
        unzip_file(zip_path, extract_to)
        os.remove(zip_path)
        return True
    else:
        print(f"[CLIENT] No checkpoint found on server. Starting fresh.")
        return False
