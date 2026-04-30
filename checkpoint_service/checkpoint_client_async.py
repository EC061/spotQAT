import os
import shutil
import queue
import threading
import requests

# Update this to the checkpoint server host if it is not on the same machine.
SERVER_URL = os.environ.get("CHECKPOINT_SERVER_URL", "http://localhost:8000")


class AsyncCheckpointClient:
    """Background uploader for checkpoint files."""

    def __init__(self, server_url: str = SERVER_URL, queue_size: int = 2):
        self.server_url = server_url.rstrip("/")
        self.tasks = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.tasks.empty():
            try:
                item = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.tasks.task_done()
                break

            file_path, remote_name, delete_after = item
            try:
                send_checkpoint_file(
                    file_path=file_path,
                    remote_name=remote_name,
                    server_url=self.server_url,
                )
                if delete_after and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as exc:
                print(f"[CLIENT] Upload error for {remote_name or os.path.basename(file_path)}: {exc}")
            finally:
                self.tasks.task_done()

    def enqueue_file(self, file_path: str, remote_name: str | None = None, delete_after: bool = False):
        payload = (file_path, remote_name, delete_after)
        if self.tasks.full():
            try:
                dropped = self.tasks.get_nowait()
                self.tasks.task_done()
                if dropped is not None:
                    dropped_path, dropped_remote_name, dropped_delete_after = dropped
                    print(
                        "[CLIENT] Upload queue full; dropping oldest pending upload: "
                        f"{dropped_remote_name or os.path.basename(dropped_path)}"
                    )
                    if dropped_delete_after and os.path.exists(dropped_path):
                        os.remove(dropped_path)
            except queue.Empty:
                pass
        self.tasks.put(payload)

    def flush(self):
        self.tasks.join()

    def close(self):
        self.flush()
        self.stop_event.set()
        self.tasks.put(None)
        self.worker.join()


def send_checkpoint_file(file_path: str, remote_name: str | None = None, server_url: str = SERVER_URL):
    """Synchronously uploads a single checkpoint file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {file_path}")

    upload_name = remote_name or os.path.basename(file_path)
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{server_url.rstrip('/')}/upload_checkpoint",
            files={"file": (upload_name, f, "application/octet-stream")},
            timeout=300,
        )

    if response.status_code != 200:
        raise RuntimeError(f"Upload failed: {response.status_code} {response.text}")

    print(f"[CLIENT] Successfully sent checkpoint: {upload_name}")


def download_checkpoint_file(
    remote_name: str,
    destination_path: str,
    server_url: str = SERVER_URL,
) -> bool:
    """Downloads a checkpoint file to destination_path using a temp file + atomic replace."""
    url = f"{server_url.rstrip('/')}/download_checkpoint/{remote_name}"
    response = requests.get(url, stream=True, timeout=300)

    if response.status_code != 200:
        print("[CLIENT] No checkpoint found on server (starting fresh).")
        return False

    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    temp_path = destination_path + ".download"
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(temp_path, destination_path)
    print(f"[CLIENT] Downloaded checkpoint: {remote_name} -> {destination_path}")
    return True


def stage_file_copy(file_path: str, staging_dir: str, staged_name: str | None = None) -> str:
    """Creates a stable copy for background upload so the source file can be overwritten later."""
    os.makedirs(staging_dir, exist_ok=True)
    target_name = staged_name or os.path.basename(file_path)
    staged_path = os.path.join(staging_dir, target_name)
    shutil.copy2(file_path, staged_path)
    return staged_path
