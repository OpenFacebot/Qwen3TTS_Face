"""Atomic artifacts and explicit content identities."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_write(path, lambda f: f.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()))


def read_json(path):
    return json.loads(Path(path).read_text())


def read_contract(path):
    value = read_json(path)
    expected = digest({key: item for key, item in value.items() if key != "fingerprint"})
    if value.get("fingerprint") != expected:
        raise ValueError("Dataset contract fingerprint mismatch")
    return value


def model_identity(root):
    root = Path(root).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".txt", ".safetensors"))
    if not files or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"No complete local model snapshot: {root}")
    return {str(p.relative_to(root)): file_hash(p) for p in files}
