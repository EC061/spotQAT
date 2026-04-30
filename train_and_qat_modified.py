import math
import os
import csv
import time
import statistics
import copy
import queue
import threading
import shutil
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
import bitsandbytes as bnb
import gc
from prettytable import PrettyTable
from tqdm import tqdm

# TorchAO QAT
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig, PerGroup
from torchao.quantization.qat import QATConfig

from checkpoint_service.checkpoint_client_async import (
    AsyncCheckpointClient,
    download_checkpoint_file,
    send_checkpoint_file,
    stage_file_copy,
)

# Configure Envs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# -----------------------------
# Configuration
# -----------------------------
MODELS = [
    "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
]

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SEQ_LEN = 256
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS_FP = 1
NUM_EPOCHS_QAT = 1
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0

SAVE_EVERY_N_STEPS = 1000
SAVE_EVERY_N_SECONDS = -1

#   1) "fixed_interval" -> save on a fixed step/time interval in the training thread
#                           and synchronously upload to the checkpoint server
#   2) "async"          -> save on a fixed step/time interval in a background thread,
#                           then upload in a second background thread
CHECKPOINT_MODE = "async"  # options: "fixed_interval", "async"
ASYNC_CHECKPOINT_QUEUE_SIZE = 2
ASYNC_UPLOAD_QUEUE_SIZE = 2

SEED = 42

BASE_OUTPUT_DIR = "./qat_experiment_out"
BASE_CHECKPOINT_DIR = "./spot_checkpoints"
REMOTE_CHECKPOINT_FILENAME = "latest_spot_checkpoint.pt"


class AsyncCheckpointWriter:
    def __init__(
        self,
        checkpoint_path,
        upload_staging_dir,
        checkpoint_times,
        record_timing_fn,
        upload_client: AsyncCheckpointClient | None,
    ):
        self.checkpoint_path = checkpoint_path
        self.upload_staging_dir = upload_staging_dir
        self.checkpoint_times = checkpoint_times
        self.record_timing_fn = record_timing_fn
        self.upload_client = upload_client
        self.tasks = queue.Queue(maxsize=ASYNC_CHECKPOINT_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _atomic_save_checkpoint(self, payload):
        temp_path = self.checkpoint_path + ".tmp"
        torch.save(payload, temp_path)
        os.replace(temp_path, self.checkpoint_path)

    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.tasks.empty():
            try:
                item = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.tasks.task_done()
                break

            payload, epoch_idx, step_idx, phase = item
            t0 = time.time()
            self._atomic_save_checkpoint(payload)
            dt = time.time() - t0
            self.checkpoint_times.append(dt)
            self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_async_write", dt)

            if self.upload_client is not None:
                stage_name = f"upload_{phase}_epoch{epoch_idx}_step{step_idx}.pt"
                staged_path = stage_file_copy(
                    self.checkpoint_path,
                    self.upload_staging_dir,
                    staged_name=stage_name,
                )
                self.upload_client.enqueue_file(
                    file_path=staged_path,
                    remote_name=REMOTE_CHECKPOINT_FILENAME,
                    delete_after=True,
                )
                self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_async_upload_enqueue", 0.0)

            self.tasks.task_done()

    def enqueue(self, payload, epoch_idx, step_idx, phase):
        if self.tasks.full():
            try:
                dropped = self.tasks.get_nowait()
                if dropped is not None:
                    _, drop_epoch, drop_step, drop_phase = dropped
                    self.record_timing_fn(
                        drop_phase,
                        drop_epoch,
                        drop_step,
                        "checkpoint_async_drop",
                        0.0,
                    )
                self.tasks.task_done()
            except queue.Empty:
                pass
        self.tasks.put((payload, epoch_idx, step_idx, phase))

    def flush(self):
        self.tasks.join()

    def close(self):
        self.flush()
        self.stop_event.set()
        self.tasks.put(None)
        self.worker.join()


def _to_cpu_state_dict(state_dict):
    cpu_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = copy.deepcopy(value)
    return cpu_state


def run_training_pipeline(model_name, run_type):
    """
    run_type: "spot" or "baseline"
    Returns timing statistics for the run.
    """
    print(f"\n{'=' * 50}")
    print(f"Starting pipeline for model: {model_name} | run_type: {run_type}")
    print(f"{'=' * 50}\n")

    torch.manual_seed(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = (
        torch.bfloat16
        if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    model_short_name = model_name.split("/")[-1]

    suffix = "_spot" if run_type == "spot" else "_baseline"
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_short_name + suffix)
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, model_short_name + suffix)
    upload_staging_dir = os.path.join(checkpoint_dir, "upload_staging")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(upload_staging_dir, exist_ok=True)

    # -----------------------------
    # Data prep
    # -----------------------------
    print(f"Loading dataset and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def tokenize_fn(batch):
        texts = [t for t in batch["text"] if t and not t.isspace()]
        return tokenizer(texts, truncation=False)

    tokenized = raw.map(
        tokenize_fn, batched=True, remove_columns=raw["train"].column_names
    )

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // SEQ_LEN) * SEQ_LEN
        concatenated = concatenated[:total_length]
        input_ids = [
            concatenated[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)
        ]
        attention_mask = [[1] * SEQ_LEN for _ in range(len(input_ids))]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [x[:] for x in input_ids],
        }

    lm_ds = tokenized.map(group_texts, batched=True)
    train_loader = DataLoader(
        lm_ds["train"],
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    eval_loader = DataLoader(
        lm_ds["validation"],
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    # -----------------------------
    # Model
    # -----------------------------
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    model.config.use_cache = False
    model.to(DEVICE)

    # -----------------------------
    # Optimizer / scheduler
    # -----------------------------
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps = (NUM_EPOCHS_FP + NUM_EPOCHS_QAT) * num_update_steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    def record_timing(phase, epoch, step, action, duration):
        csv_path = os.path.join(BASE_OUTPUT_DIR, "timing_log.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_name", "run_type", "phase", "epoch", "step", "action", "duration"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "model_name": model_name,
                    "run_type": run_type,
                    "phase": phase,
                    "epoch": epoch,
                    "step": step,
                    "action": action,
                    "duration": duration,
                }
            )

    def move_batch(batch, device):
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate(model, dataloader):
        model.eval()
        losses = []
        for batch in dataloader:
            batch = move_batch(batch, DEVICE)
            with torch.autocast(
                device_type="cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")
            ):
                out = model(**batch)
            losses.append(out.loss.detach().float().item())
        mean_loss = sum(losses) / max(len(losses), 1)
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
        return mean_loss, ppl

    epoch_times = []
    checkpoint_times = []
    checkpoint_path = os.path.join(checkpoint_dir, REMOTE_CHECKPOINT_FILENAME)

    upload_client = None
    async_checkpoint_writer = None
    if run_type == "spot":
        if CHECKPOINT_MODE == "async":
            upload_client = AsyncCheckpointClient(queue_size=ASYNC_UPLOAD_QUEUE_SIZE)
            async_checkpoint_writer = AsyncCheckpointWriter(
                checkpoint_path=checkpoint_path,
                upload_staging_dir=upload_staging_dir,
                checkpoint_times=checkpoint_times,
                record_timing_fn=record_timing,
                upload_client=upload_client,
            )

    def build_checkpoint_payload(epoch_idx, step_idx, phase):
        return {
            "epoch": epoch_idx,
            "step": step_idx,
            "phase": phase,
            "model_state_dict": _to_cpu_state_dict(model.state_dict()),
            "optimizer_state_dict": _to_cpu_state_dict(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        }

    def atomic_save_checkpoint(payload):
        temp_path = checkpoint_path + ".tmp"
        torch.save(payload, temp_path)
        os.replace(temp_path, checkpoint_path)

    def maybe_restore_remote_checkpoint():
        if os.path.exists(checkpoint_path):
            return True
        if run_type != "spot":
            return False
        print(f"No local spot checkpoint found for {model_name}; trying remote server...")
        return download_checkpoint_file(REMOTE_CHECKPOINT_FILENAME, checkpoint_path)

    def load_spot_checkpoint_if_present():
        start_epoch = 1
        start_step = 0
        current_phase = "fp"

        if run_type == "spot" and maybe_restore_remote_checkpoint() and os.path.exists(checkpoint_path):
            print(f"Found spot checkpoint at {checkpoint_path} for {model_name}. Resuming...")
            chkpt = torch.load(checkpoint_path, map_location="cpu")
            start_epoch = chkpt["epoch"]
            start_step = chkpt.get("step", 0)
            current_phase = chkpt["phase"]

            if current_phase == "qat":
                print(
                    "Resuming directly into QAT phase, setting up fake quantize modules BEFORE loading state dict..."
                )
                base_config = Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=torch.int4,
                    weight_granularity=PerGroup(32),
                )
                quantize_(model, QATConfig(base_config, step="prepare"))
                model.to(DEVICE)

            model.load_state_dict(chkpt["model_state_dict"])
            model.to(DEVICE)
            optimizer.load_state_dict(chkpt["optimizer_state_dict"])
            scheduler.load_state_dict(chkpt["scheduler_state_dict"])
        else:
            if run_type == "spot":
                print("No spot checkpoint found. Starting from scratch.")

        return start_epoch, start_step, current_phase

    start_epoch, start_step, current_phase = load_spot_checkpoint_if_present()

    def save_spot_checkpoint_fixed_interval(epoch_idx, step_idx, phase):
        t0 = time.time()
        chkpt = build_checkpoint_payload(epoch_idx, step_idx, phase)
        atomic_save_checkpoint(chkpt)
        send_checkpoint_file(checkpoint_path, remote_name=REMOTE_CHECKPOINT_FILENAME)
        dt = time.time() - t0
        checkpoint_times.append(dt)
        record_timing(phase, epoch_idx, step_idx, "checkpoint_fixed_interval", dt)

    def save_spot_checkpoint_async(epoch_idx, step_idx, phase):
        t0 = time.time()
        chkpt = build_checkpoint_payload(epoch_idx, step_idx, phase)
        enqueue_dt = time.time() - t0
        checkpoint_times.append(enqueue_dt)
        record_timing(phase, epoch_idx, step_idx, "checkpoint_async_enqueue", enqueue_dt)
        async_checkpoint_writer.enqueue(chkpt, epoch_idx, step_idx, phase)

    def save_spot_checkpoint(epoch_idx, step_idx, phase):
        if CHECKPOINT_MODE == "fixed_interval":
            save_spot_checkpoint_fixed_interval(epoch_idx, step_idx, phase)
        elif CHECKPOINT_MODE == "async":
            save_spot_checkpoint_async(epoch_idx, step_idx, phase)
        else:
            raise ValueError(
                f"Unsupported CHECKPOINT_MODE={CHECKPOINT_MODE}. Use 'fixed_interval' or 'async'."
            )

    def train_one_epoch(
        model, dataloader, optimizer, scheduler, epoch_idx, phase_name, start_step=0
    ):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        last_save_time = time.time()
        total_batches = len(dataloader)

        with tqdm(
            total=total_batches,
            initial=start_step,
            desc=f"[{phase_name}] Epoch {epoch_idx}",
        ) as pbar:
            for step, batch in enumerate(dataloader, start=1):
                if step <= start_step:
                    continue

                step_t0 = time.time()

                batch = move_batch(batch, DEVICE)
                with torch.autocast(
                    device_type="cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")
                ):
                    out = model(**batch)
                    loss = out.loss / GRAD_ACCUM

                loss.backward()
                running += loss.item() * GRAD_ACCUM

                if step % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                step_dt = time.time() - step_t0
                record_timing(phase_name, epoch_idx, step, "train_step", step_dt)

                pbar.update(1)
                pbar.set_postfix({"loss": f"{running / (step - start_step):.4f}"})

                if run_type == "spot":
                    if (SAVE_EVERY_N_STEPS > 0 and step % SAVE_EVERY_N_STEPS == 0) or (
                        SAVE_EVERY_N_SECONDS > 0
                        and time.time() - last_save_time >= SAVE_EVERY_N_SECONDS
                    ):
                        save_spot_checkpoint(epoch_idx, step, phase_name)
                        last_save_time = time.time()

        return running / max(total_batches - start_step, 1)

    # -----------------------------
    # Phase 1: normal fine-tuning
    # -----------------------------
    global_start_time = time.time()

    if current_phase == "fp":
        print(f"Starting standard fine-tuning for {model_name}...")
        for epoch in range(start_epoch, NUM_EPOCHS_FP + 1):
            epoch_start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, "fp", start_step
            )
            start_step = 0

            eval_t0 = time.time()
            val_loss, val_ppl = evaluate(model, eval_loader)
            eval_dt = time.time() - eval_t0
            record_timing("fp", epoch, 0, "evaluate", eval_dt)

            if run_type == "spot":
                save_spot_checkpoint(epoch + 1, 0, "fp")

            total_epoch_time = time.time() - epoch_start_time
            epoch_times.append(total_epoch_time)
            print(
                f"[fp] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} time={total_epoch_time:.2f}s"
            )

        start_epoch = 1
        current_phase = "qat"

    # -----------------------------
    # Phase 2: QAT prepare -> train
    # -----------------------------
    if current_phase == "qat":
        if start_epoch == 1:
            print(f"Preparing model {model_name} for QAT...")
            base_config = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(32),
            )
            quantize_(model, QATConfig(base_config, step="prepare"))

        for epoch in range(start_epoch, NUM_EPOCHS_QAT + 1):
            epoch_start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, "qat", start_step
            )
            start_step = 0

            eval_t0 = time.time()
            val_loss, val_ppl = evaluate(model, eval_loader)
            eval_dt = time.time() - eval_t0
            record_timing("qat", epoch, 0, "evaluate", eval_dt)

            if run_type == "spot":
                save_spot_checkpoint(epoch + 1, 0, "qat")

            total_epoch_time = time.time() - epoch_start_time
            epoch_times.append(total_epoch_time)
            print(
                f"[qat] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} time={total_epoch_time:.2f}s"
            )

    if async_checkpoint_writer is not None:
        async_checkpoint_writer.flush()
    if upload_client is not None:
        upload_client.flush()

    # -----------------------------
    # Convert after QAT
    # -----------------------------
    print(f"Converting QAT model {model_name}...")
    convert_t0 = time.time()

    base_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
    )
    quantize_(model, QATConfig(base_config, step="convert"))

    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    convert_time = time.time() - convert_t0
    total_time = time.time() - global_start_time

    print(f"Saved quantized {model_name} model to: {output_dir}")
    print(f"Finished processing {model_name} ({run_type}) in {total_time:.2f}s!\n")

    if async_checkpoint_writer is not None:
        async_checkpoint_writer.close()
    if upload_client is not None:
        upload_client.close()

    shutil.rmtree(upload_staging_dir, ignore_errors=True)

    del model
    del optimizer
    del scheduler
    del train_loader
    del eval_loader
    del raw
    del lm_ds
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "run_type": run_type,
        "total_time": total_time,
        "epoch_times": epoch_times,
        "checkpoint_times": checkpoint_times,
        "convert_time": convert_time,
    }


def print_statistics(results):
    table = PrettyTable()
    table.field_names = [
        "Model",
        "Type",
        "Total (s)",
        "Epoch Avg (s)",
        "Epoch Std (s)",
        "Chkpt Avg (s)",
        "Chkpt Std (s)",
        "Convert (s)",
    ]

    for r in results:
        t_tot = f"{r['total_time']:.2f}"

        ep_times = r["epoch_times"]
        if len(ep_times) > 0:
            ep_avg = f"{statistics.mean(ep_times):.2f}"
            ep_std = (
                f"{statistics.stdev(ep_times):.2f}" if len(ep_times) > 1 else "0.00"
            )
        else:
            ep_avg = ep_std = "N/A"

        cp_times = r["checkpoint_times"]
        if len(cp_times) > 0:
            cp_avg = f"{statistics.mean(cp_times):.2f}"
            cp_std = (
                f"{statistics.stdev(cp_times):.2f}" if len(cp_times) > 1 else "0.00"
            )
        else:
            cp_avg = cp_std = "N/A"

        t_conv = f"{r['convert_time']:.2f}"

        table.add_row(
            [
                r["model"].split("/")[-1],
                r["run_type"],
                t_tot,
                ep_avg,
                ep_std,
                cp_avg,
                cp_std,
                t_conv,
            ]
        )

    print("\n" + str(table) + "\n")


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)

    results = []

    for model_name in MODELS:
        res_spot = run_training_pipeline(model_name, "spot")
        results.append(res_spot)

        res_base = run_training_pipeline(model_name, "baseline")
        results.append(res_base)

    print_statistics(results)


if __name__ == "__main__":
    main()
