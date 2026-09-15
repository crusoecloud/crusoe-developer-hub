"""Helpers for the multipart upload notebook: dataset export, part reading, parallel sends, and polling."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MiB = 1024 * 1024


def build_training_file(dataset, split, path, max_rows=None, token=None):
    """Write a Hugging Face chat dataset to chat-format JSONL and return the path."""
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split, token=token)
    if max_rows:
        rows = rows.select(range(max_rows))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
    return path


def part_count(path, part_size):
    """Number of parts needed to cover the file."""
    return -(-Path(path).stat().st_size // part_size)


def read_part(path, index, part_size):
    """Read one part of the file by offset."""
    with Path(path).open("rb") as f:
        f.seek(index * part_size)
        return f.read(part_size)


def md5_of(path):
    """MD5 hex digest of the whole file, streamed in 8 MiB chunks."""
    digest = hashlib.md5()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * MiB):
            digest.update(chunk)
    return digest.hexdigest()


def send_in_parallel(send_part, n_parts, workers, attempts=3):
    """Run send_part(index) for every part from a thread pool, retrying a failed part on its own.

    send_part must return (index, part_id). Returns the part ids in file order.
    """

    def with_retry(index):
        for attempt in range(1, attempts + 1):
            try:
                return send_part(index)
            except Exception as exc:
                if attempt == attempts:
                    raise
                print(f"part {index + 1} attempt {attempt} failed ({type(exc).__name__}), re-sending")

    part_ids = [None] * n_parts
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(with_retry, i) for i in range(n_parts)]
        for future in as_completed(futures):
            index, part_id = future.result()
            part_ids[index] = part_id
            print(f"[{time.time() - started:5.1f}s] part {index + 1:>2}/{n_parts} landed  {part_id}")
    return part_ids, time.time() - started


def wait_for_assembly(base_url, headers, upload_id, poll_seconds=5, deadline_seconds=1800):
    """Poll GET /uploads/{id} until the status leaves pending; return the upload JSON and seconds waited."""
    started = time.time()
    while time.time() - started < deadline_seconds:
        state = requests.get(f"{base_url}/uploads/{upload_id}", headers=headers, timeout=30).json()
        if state["status"] != "pending":
            return state, time.time() - started
        time.sleep(poll_seconds)
    raise TimeoutError(f"upload {upload_id} still pending after {deadline_seconds} s")
