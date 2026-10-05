"""
Run the migrated True-JEPA notebook as an ordered, resumable pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from checkpointing import atomic_json_save, atomic_torch_save, load_torch_checkpoint

# ============================================================================
# CONTROLES EDITÁVEIS
# ============================================================================

# Modo usado quando --mode não é informado na linha de comando.
SELECTED_MODE = "STANDARD"

@dataclass(frozen=True)
class ModePreset:
    per_family: int
    reference_nodes: int
    model_nodes: int
    validation_nodes: int
    hidden: int
    ae_epochs: int
    jepa_epochs: int
    supervised_epochs: int
    physics_epochs: int
    finetune_epochs: int
    refinement_train_steps: int
    refinement_inference_steps: int
    seeds: tuple[int, ...]

PRESETS = {
    "QUICK": ModePreset(3, 513, 129, 257, 32, 8, 1, 3, 1, 1, 1, 1, (7,)),
    "DEVELOPMENT": ModePreset(32, 513, 129, 257, 72, 24, 6, 18, 12, 8, 6, 8, (7, 17, 29)),
    "STANDARD": ModePreset(128, 1025, 257, 513, 128, 32, 24, 80, 40, 30, 3, 5, (7, 17, 29)),
    "FULL": ModePreset(192, 2049, 513, 1025, 160, 60, 30, 80, 40, 30, 5, 10, (7, 17, 29)),
}

PROJECT_DIR = Path(__file__).resolve().parent
STAGE_DIR = PROJECT_DIR / "stages"
STAGES = ("s00_bootstrap.py", "s01_physics.py", "s02_data.py", "s03_models.py",
    "s04_validation.py", "s05_training.py", "s06_evaluation.py", "s07_reports.py",
)

class TeeStream:
    """Mirror a text stream to the terminal and to the execution log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal, self.log = terminal, log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def classify_failure(exc: BaseException) -> str:
    description = f"{type(exc).__name__}: {exc}".lower()
    if "outofmemory" in description or "out of memory" in description or "cannot allocate memory" in description:
        return "out_of_memory"
    if isinstance(exc, KeyboardInterrupt):
        return "keyboard_interrupt"
    if isinstance(exc, SystemExit):
        return "system_exit"
    return "execution_error"

def recover_stale_run(ledger_path: Path) -> None:
    """Record a prior run that ended before it could update its ledger."""
    if not ledger_path.exists():
        return
    try:
        import json
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if previous.get("status") != "running":
        return
    previous.update({
        "status": "aborted_external",
        "failure_type": "unclean_termination",
        "error": ("The process ended without recording an exception. Possible causes include "
                  "SIGKILL, operating-system/container OOM termination, host shutdown, or power loss."),
        "detected_at": utc_now(),
    })
    atomic_json_save(previous, ledger_path.with_name("pipeline_state_previous_aborted.json"))

def make_log_path(mode: str) -> Path:
    log_dir = PROJECT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return log_dir / f"run_{stamp}_{mode.lower()}_{os.getpid()}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("QUICK", "DEVELOPMENT", "STANDARD", "FULL"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--restart", action="store_true", help="Ignore resumable training state.")
    parser.add_argument("--stop-after", choices=STAGES, help="Useful for validating one pipeline boundary.")
    return parser.parse_args()

def execute_stage(path: Path, namespace: dict[str, object]) -> None:
    source = path.read_text(encoding="utf-8")
    namespace["__file__"] = str(path)
    namespace["__name__"] = "__main__"
    exec(compile(source, str(path), "exec"), namespace)

def main() -> None:
    args = parse_args()
    selected_mode = args.mode or SELECTED_MODE
    if selected_mode not in PRESETS:
        raise ValueError(f"SELECTED_MODE deve ser um destes valores: {tuple(PRESETS)}")
    os.environ["PSI_JEPA_RUN_MODE"] = selected_mode
    if args.seed is not None:
        os.environ["PSI_JEPA_SEED"] = str(args.seed)
    os.environ.setdefault("PSI_JEPA_REUSE_NUMERICAL_CACHE", "1")
    os.environ["PSI_JEPA_RESTART"] = "1" if args.restart else "0"
    os.environ.setdefault("PSI_JEPA_PROJECT_DIR", str(PROJECT_DIR))

    namespace: dict[str, object] = {
        "atomic_torch_save": atomic_torch_save,
        "load_torch_checkpoint": load_torch_checkpoint,
        "ModePreset": ModePreset,
        "PRESETS": PRESETS,
    }
    ledger_path = PROJECT_DIR / "pipeline_state.json"
    recover_stale_run(ledger_path)
    log_path = make_log_path(selected_mode)
    started = time.monotonic()
    ledger: dict[str, object] = {
        "status": "running", "mode": selected_mode, "pid": os.getpid(),
        "started_at": utc_now(), "log_file": str(log_path), "stages": {},
    }
    atomic_json_save(ledger, ledger_path)
    old_handlers: dict[int, object] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        ledger.update({"status": "interrupted", "failure_type": "external_signal",
            "error": f"Received external signal {name} ({signum})", "finished_at": utc_now(),
            "seconds": time.monotonic() - started})
        atomic_json_save(ledger, ledger_path)
        print(f"\nPIPELINE INTERRUPTED BY {name} ({signum})", file=sys.stderr, flush=True)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    stopped_early = False
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        with contextlib.redirect_stdout(TeeStream(sys.stdout, log_handle)), contextlib.redirect_stderr(TeeStream(sys.stderr, log_handle)):
            print(f"PIPELINE START | utc={ledger['started_at']} | mode={selected_mode} | pid={os.getpid()}")
            print(f"LOG FILE: {log_path}")
            try:
                for filename in STAGES:
                    stage_started = time.monotonic()
                    ledger["stages"][filename] = {"status": "running", "started_at": utc_now()}
                    atomic_json_save(ledger, ledger_path)
                    print(f"\n=== Running {filename} ===", flush=True)
                    try:
                        execute_stage(STAGE_DIR / filename, namespace)
                    except BaseException:
                        ledger["stages"][filename].update({"status": "failed",
                            "seconds": time.monotonic() - stage_started, "finished_at": utc_now()})
                        raise
                    ledger["stages"][filename].update({"status": "complete",
                        "seconds": time.monotonic() - stage_started, "finished_at": utc_now()})
                    print(f"=== Completed {filename} in {ledger['stages'][filename]['seconds']:.3f}s ===")
                    atomic_json_save(ledger, ledger_path)
                    if args.stop_after == filename:
                        stopped_early = True
                        break
            except BaseException as exc:
                signal_failure = ledger.get("failure_type") == "external_signal"
                ledger.update({"status": "interrupted" if signal_failure else "failed",
                    "failure_type": "external_signal" if signal_failure else classify_failure(exc),
                    "error": ledger.get("error") if signal_failure else f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(), "seconds": time.monotonic() - started})
                atomic_json_save(ledger, ledger_path)
                print(f"\nPIPELINE FAILED | type={ledger['failure_type']} | elapsed={ledger['seconds']:.3f}s", file=sys.stderr)
                traceback.print_exc()
                raise
            else:
                ledger.update({"status": "stopped" if stopped_early else "complete",
                    "finished_at": utc_now(), "seconds": time.monotonic() - started})
                atomic_json_save(ledger, ledger_path)
                print(f"\nPIPELINE {str(ledger['status']).upper()} | elapsed={ledger['seconds']:.3f}s")
            finally:
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)

if __name__ == "__main__":
    main()
