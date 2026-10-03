"""Portable atomic outputs, provenance and exclusive per-card/run locks."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def replace(tmp, target):
    deadline = time.monotonic() + 5
    while True:
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    replace(tmp, path)

def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    replace(tmp, path)

@contextlib.contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("a+b")
    except OSError as exc:
        raise RuntimeError(f"Exclusive lock occupied: {path}") from exc
    try:
        f.seek(0)
        if not f.read(1):
            f.write(b"\0")
            f.flush()
        f.seek(0)
    except OSError as exc:
        f.close()
        raise RuntimeError(f"Exclusive lock occupied: {path}") from exc
    if os.name == "nt":
        import msvcrt
        lock = lambda: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        unlock = lambda: msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        lock = lambda: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        unlock = lambda: fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    try:
        lock()
    except OSError:
        f.close()
        raise RuntimeError(f"Exclusive lock occupied: {path}")
    try:
        yield
    finally:
        f.seek(0)
        unlock()
        f.close()

def code_hashes():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for folder in (ROOT / "code", ROOT / "core")
            for p in sorted(folder.rglob("*")) if p.suffix in (".py", ".json", ".sh")}
