import json
import os
import datetime as dt
import hashlib
import streamlit as st

DATASET_FILE = "dataset.jsonl"

def _ensure_file():
    """Ensure the dataset file exists."""
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            pass  # create empty file

def _hash_api_key(api_key: str) -> str:
    """Create a hash of the API key for privacy."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]

def append_record(transcript: str, summary_md: str, resources_md: str, api_key: str, filename: str = "unknown") -> None:
    """Append a new sample in JSON-Lines format with API key hash."""
    _ensure_file()
    record = {
        "id": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "api_key_hash": _hash_api_key(api_key),
        "filename": filename,
        "transcript": transcript,
        "summary_markdown": summary_md,
        "resources_markdown": resources_md,
        "preview": (summary_md[:150] + "…") if len(summary_md) > 150 else summary_md,
    }
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_user_records(api_key: str) -> list:
    """Get all records for a specific API key."""
    _ensure_file()
    user_hash = _hash_api_key(api_key)
    records = []

    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if record.get("api_key_hash") == user_hash:
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass

    return records

def read_dataset() -> bytes:
    """Return full file bytes for download."""
    _ensure_file()
    with open(DATASET_FILE, "rb") as f:
        return f.read()

def get_dataset_stats() -> dict:
    """Get statistics about the dataset."""
    _ensure_file()
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return {
                "total_records": len([l for l in lines if l.strip()]),
                "file_size_mb": os.path.getsize(DATASET_FILE) / (1024 * 1024)
            }
    except FileNotFoundError:
        return {"total_records": 0, "file_size_mb": 0}

def is_owner_api_key(api_key: str) -> bool:
    """Check if the API key is the owner's key. Replace with your actual API key hash."""
    # Replace this with your actual API key hash
    # You can generate it by running: _hash_api_key("your_actual_api_key")
    # import hashlib, os
    # print(hashlib.sha256("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".encode()).hexdigest()[:16])
    OWNER_API_KEY_HASH = st.secrets["OWNER_API_KEY_HASH"]  # Replace with your actual hash
    return _hash_api_key(api_key) == OWNER_API_KEY_HASH