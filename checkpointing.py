"""Crash-safe persistence used by the staged pipeline."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _replace_with_windows_retry(temporary: Path, path: Path) -> None:
    """Preserve atomic replacement across short OneDrive/antivirus file locks."""
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a torch payload without ever exposing a partially-written checkpoint."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    _replace_with_windows_retry(temporary, path)


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _replace_with_windows_retry(temporary, path)
