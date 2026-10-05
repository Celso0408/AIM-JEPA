"""Crash-safe persistence used by the staged pipeline."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """OneDrive can briefly hold the destination open on Windows."""
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.1 * (attempt + 1), 1.0))


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a torch payload without ever exposing a partially-written checkpoint."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    _replace_with_retry(temporary, path)


def load_torch_checkpoint(path: Path) -> dict[str, Any] | None:
    import torch

    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Ignoring unreadable checkpoint {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _replace_with_retry(temporary, path)
