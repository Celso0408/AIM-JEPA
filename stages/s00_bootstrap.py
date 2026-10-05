"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 2]
from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import io
import json
import math
import os
import pickle
import platform
import random
import shutil
import socket
import sys
import time
import tracemalloc
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-psi-true-jepa")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from scipy import linalg, optimize, sparse, special
from scipy.spatial import cKDTree
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.stats import qmc
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from main import ModePreset as _ModePreset, PRESETS as _PRESETS
except Exception:  # pragma: no cover - fallback for direct imports
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
else:
    ModePreset = _ModePreset
    PRESETS = _PRESETS

# %% [notebook cell 5]
RUN_MODE = os.environ.get("PSI_JEPA_RUN_MODE", "STANDARD").upper()
RUN_MODE_OPTIONS = ("QUICK", "DEVELOPMENT", "STANDARD", "FULL")
BACKBONE_OPTIONS = ("direct_baseline", "gnn_mlp", "gnn_kan")
KAN_BASIS_OPTIONS = ("chebyshev", "legendre", "hermite")
DECODER_OPTIONS = ("spectral_sine_residual", "topology_phase")
PRIMARY_OUTPUT_OPTIONS = ("direct_topology", "hybrid", "physics_selected")
ARCHITECTURE_VERSION = "psi-true-jepa-1d-v8-expanded-domain-topology-gated"
SOLVER_VERSION = "mapped-p1-fem-gl4-v3-bound-state-validation"
SOURCE_REVISION = "2026-09-25-expanded-domain-direct-supervision"
PYTHON_HASH_SEED_AT_PROCESS_START=os.environ.get("PYTHONHASHSEED")
_source_root = Path(os.environ.get("PSI_JEPA_PROJECT_DIR", Path(__file__).resolve().parents[1]))
_source_files = [_source_root / "main.py", _source_root / "checkpointing.py"] + sorted((_source_root / "stages").glob("s*.py"))
_source_digest = hashlib.sha256()
for _source_path in _source_files:
    _source_digest.update(_source_path.relative_to(_source_root).as_posix().encode())
    _source_digest.update(_source_path.read_bytes())
SOURCE_SHA256 = _source_digest.hexdigest()
del _source_digest, _source_files, _source_path, _source_root


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


try:
    ACTIVE_SEED = int(os.environ.get("PSI_JEPA_SEED", "7"))
except ValueError as exc:
    raise ValueError("PSI_JEPA_SEED must be an integer") from exc
seed_everything(ACTIVE_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "deterministic_algorithms_enforced": torch.are_deterministic_algorithms_enabled(),
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "python_hash_seed_at_process_start": PYTHON_HASH_SEED_AT_PROCESS_START,
    "python_hash_seed_runtime_value": os.environ.get("PYTHONHASHSEED"),
}
print(json.dumps(ENVIRONMENT, indent=2))

# %% [notebook cell 7]
@dataclass(frozen=True)
class TinyOverfitPolicy:
    """Deterministic capacity-test policy, separate from production training budgets."""

    fidelity_min: float
    loss_ratio_max: float
    max_steps: int
    native_steps: int = 180
    native_fidelity_min: float = .75
    field_steps: int = 120
    eval_interval: int = 10
    stable_evaluations: int = 2
    learning_rate: float = 3e-3
    minimum_learning_rate: float = 3e-5


# Larger modes use wider models and finer grids.  A fixed 180-update budget made their
# capacity test strictly harder while its acceptance threshold also increased.  These
# are maximum diagnostic budgets: convergence stops early after consecutive successful
# evaluations, and the exact policy is recorded in the configuration/run card.
TINY_OVERFIT_POLICIES = {
    "QUICK": TinyOverfitPolicy(.80, .55, 360,field_steps=120),
    "DEVELOPMENT": TinyOverfitPolicy(.90, .40, 900,field_steps=240),
    "STANDARD": TinyOverfitPolicy(.90, .40, 1200,field_steps=300),
    "FULL": TinyOverfitPolicy(.90, .40, 1500,field_steps=360),
}
TINY_OVERFIT_SEED = 7


@dataclass
class Config:
    mode: str = RUN_MODE
    seed: int = ACTIVE_SEED
    k_states: int = 11
    extra_eigenpairs: int = 8
    kinetic_coefficient: float = 0.5
    backbone: str = "gnn_kan"
    kan_basis: str = "chebyshev"
    kan_degree: int = 4
    kan_rank: int = 4
    kan_placements: tuple[str, ...] = ("message", "update", "predictor", "decoder", "energy")
    graph_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    graph_layers: int = 3
    positional_frequencies: int = 12
    batch_size: int = 8
    gradient_accumulation: int = 2
    gradient_clip_norm: float = 5.0
    module_gradient_clip_rms: float = 4.0
    module_gradient_clip_hard_norm: float = 1000.0
    gradient_clip_target_fraction: float = 0.20
    bootstrap_repeats: int = 40
    latent_dim: int = 120
    spectral_modes: int = 120
    decoder_type: str = "topology_phase"
    primary_output: str = "physics_selected"
    phase_q_min: float = 1e-3
    phase_logit_clip: float = 8.0
    amplitude_log_clip: float = 6.0
    minimum_intervals_per_lobe: int = 3
    topology_min_lobe_mass: float = 1e-5
    topology_node_loss_weight: float = 0.20
    topology_lobe_mass_weight: float = 0.08
    # Target-only field supervision.  The stronger phase weight reflects that node
    # placement is controlled directly by C(t), whereas log amplitude is also anchored
    # by the waveform, density, and H1 objectives.
    topology_phase_cdf_loss_weight: float = 5.0
    topology_log_amplitude_loss_weight: float = 0.5
    topology_target_carrier_floor: float = 1e-3
    topology_amplitude_target_clip_margin: float = 5e-2
    topology_phase_smoothness_weight: float = 2e-4
    topology_phase_resolution_weight: float = 0.04
    topology_amplitude_smoothness_weight: float = 2e-5
    topology_safeguard: bool = True
    route_selection_gram_weight: float = 0.05
    route_selection_topology_weight: float = 0.0
    route_selection_tolerance: float = 1e-7
    dropout: float = 0.0
    boundary_mode: str = "explicit"  # explicit, envelope, soft
    orthonormalization: str = "rayleigh_ritz"  # loss_only, lowdin, lowdin_plus_loss, rayleigh_ritz
    orth_eigen_floor: float = 1e-6
    primary_use_continuum_threshold: bool = False
    ema_tau_start: float = 0.99
    ema_tau_end: float = 0.999
    learning_rate: float = 3e-4
    decoder_learning_rate_multiplier: float = 1.0
    energy_ground_residual_clip: float = 16.0
    energy_log_gap_clip: float = 12.0
    energy_scale_floor: float = 1e-3*math.pi**2
    energy_reference_gap_floor: float = 1e-6
    energy_ground_loss_weight: float = 0.15
    energy_log_gap_loss_weight: float = 0.15
    energy_exact_loss_weight: float = 0.15
    rayleigh_exact_loss_weight: float = 0.08
    energy_rayleigh_consistency_weight: float = 0.05
    physics_learning_rate: float = 1e-4
    ae_learning_rate: float = 1e-5
    finetune_learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    physics_start_fidelity: float = 0.70
    physics_stop_fidelity: float = 0.55
    physics_authorization_patience: int = 2
    physics_ramp_epochs: int = 8
    ritz_gradient_start_fidelity: float = 0.65
    ritz_gradient_start_projector_loss: float = 0.20
    ritz_gradient_authorization_patience: int = 2
    ritz_gradient_min_relative_gap: float = 1e-3
    ritz_gradient_min_safe_fraction: float = 0.75
    refinement_enabled: bool = True
    refinement_learned_correction: bool = True
    refinement_damping: float = 0.75
    refinement_preconditioner_floor: float = 0.05
    refinement_max_correction_norm: float = 0.75
    refinement_exploration_scale: float = 0.05
    refinement_learned_multiplier: float = 0.25
    refinement_acceptance_tolerance: float = 1e-5
    refinement_backtrack_steps: int = 2
    refinement_contraction_target: float = 0.98
    refinement_stop_residual: float = 1e-3
    refinement_stagnation_tolerance: float = 1e-4
    refinement_stagnation_patience: int = 2
    early_stopping_min_delta: float = 1e-4
    early_stopping_patience: int = 7
    reflection_probability: float = 0.50
    shift_probability: float = 0.25
    translation_probability: float = 0.25
    dilation_probability: float = 0.25
    resampling_probability: float = 0.15
    reflection_pair_probability: float = 0.50
    ood_unseen_per_family: int = 0  # mode-derived: QUICK 1, DEVELOPMENT 16, STANDARD 64, FULL 128
    max_generation_attempts: int = 3
    clean_output: bool = os.environ.get("PSI_JEPA_CLEAN_OUTPUT", "0") == "1"
    use_sigreg: bool = False
    architecture_version: str = ARCHITECTURE_VERSION
    solver_version: str = SOLVER_VERSION

    def __post_init__(self) -> None:
        self.mode = self.mode.upper()
        if self.mode not in RUN_MODE_OPTIONS:
            raise ValueError(f"mode must be one of {RUN_MODE_OPTIONS}")
        if self.seed not in PRESETS[self.mode].seeds:
            raise ValueError(f"seed {self.seed} is not supported for {self.mode}; choose one of {PRESETS[self.mode].seeds}")
        if self.backbone not in BACKBONE_OPTIONS:
            raise ValueError(f"backbone must be one of {BACKBONE_OPTIONS}")
        if self.kan_basis not in KAN_BASIS_OPTIONS:
            raise ValueError(f"KAN basis must be one of {KAN_BASIS_OPTIONS}")
        if self.decoder_type not in DECODER_OPTIONS:
            raise ValueError(f"decoder_type must be one of {DECODER_OPTIONS}")
        if self.primary_output not in PRIMARY_OUTPUT_OPTIONS:
            raise ValueError(f"primary_output must be one of {PRIMARY_OUTPUT_OPTIONS}")
        if self.primary_output=="direct_topology" and self.decoder_type!="topology_phase":
            raise ValueError("primary_output='direct_topology' requires decoder_type='topology_phase'")
        route_values=(self.route_selection_gram_weight,self.route_selection_topology_weight,self.route_selection_tolerance)
        if any(not math.isfinite(value) or value<0 for value in route_values):
            raise ValueError("route-selection weights and tolerance must be finite and non-negative")
        tiny_policy=TINY_OVERFIT_POLICIES[self.mode]
        if (not 0<tiny_policy.fidelity_min<1 or not 0<tiny_policy.loss_ratio_max<1 or
            tiny_policy.max_steps<=tiny_policy.native_steps+tiny_policy.field_steps or
            tiny_policy.native_steps<1 or tiny_policy.field_steps<1 or
            not 0<tiny_policy.native_fidelity_min<=tiny_policy.fidelity_min or
            tiny_policy.eval_interval<1 or tiny_policy.stable_evaluations<1 or
            not 0<tiny_policy.minimum_learning_rate<=tiny_policy.learning_rate):
            raise ValueError(f"invalid tiny-overfit policy for {self.mode}: {tiny_policy}")
        if (not math.isfinite(self.phase_q_min) or self.phase_q_min<=0 or
            not math.isfinite(self.phase_logit_clip) or self.phase_logit_clip<=0 or
            not math.isfinite(self.amplitude_log_clip) or self.amplitude_log_clip<=0):
            raise ValueError("phase_q_min, phase_logit_clip, and amplitude_log_clip must be finite and strictly positive")
        if (isinstance(self.minimum_intervals_per_lobe,bool) or
            not isinstance(self.minimum_intervals_per_lobe,(int,np.integer)) or
            self.minimum_intervals_per_lobe<2):
            raise ValueError("minimum_intervals_per_lobe must be an integer of at least two")
        if not math.isfinite(self.topology_min_lobe_mass) or not 0<=self.topology_min_lobe_mass<1:
            raise ValueError("topology_min_lobe_mass must lie in [0,1)")
        topology_weights=(self.topology_node_loss_weight,self.topology_lobe_mass_weight,
                          self.topology_phase_cdf_loss_weight,self.topology_log_amplitude_loss_weight,
                          self.topology_phase_smoothness_weight,self.topology_phase_resolution_weight,
                          self.topology_amplitude_smoothness_weight)
        if any(value<0 or not math.isfinite(value) for value in topology_weights):
            raise ValueError("topology loss weights must be finite and non-negative")
        if not math.isfinite(self.topology_target_carrier_floor) or not 0<self.topology_target_carrier_floor<1:
            raise ValueError("topology_target_carrier_floor must lie strictly between zero and one")
        if (not math.isfinite(self.topology_amplitude_target_clip_margin) or
            not 0<self.topology_amplitude_target_clip_margin<self.amplitude_log_clip):
            raise ValueError("topology_amplitude_target_clip_margin must lie in (0, amplitude_log_clip)")
        if not math.isfinite(self.decoder_learning_rate_multiplier) or self.decoder_learning_rate_multiplier<=0:
            raise ValueError("decoder_learning_rate_multiplier must be finite and strictly positive")
        energy_values=(self.energy_ground_residual_clip,self.energy_log_gap_clip,self.energy_scale_floor,
                       self.energy_reference_gap_floor,self.energy_ground_loss_weight,
                       self.energy_log_gap_loss_weight,self.energy_exact_loss_weight,
                       self.rayleigh_exact_loss_weight,self.energy_rayleigh_consistency_weight)
        if (any(not math.isfinite(value) for value in energy_values) or
            any(value<=0 for value in energy_values[:4])):
            raise ValueError("energy coordinate clips, scale floor, and reference-gap floor must be finite and strictly positive")
        if any(value<0 for value in energy_values[4:]):
            raise ValueError("energy loss weights must be finite and non-negative")
        if self.boundary_mode not in ("explicit", "envelope", "soft"):
            raise ValueError("unknown boundary mode")
        if self.spectral_modes < self.latent_dim:
            raise ValueError("spectral_modes must be at least latent_dim")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        if not (self.module_gradient_clip_rms>0 and self.module_gradient_clip_hard_norm>0):
            raise ValueError("module gradient clip thresholds must be positive")
        if not 0<self.gradient_clip_target_fraction<1:
            raise ValueError("gradient_clip_target_fraction must lie in (0,1)")
        if not (0<=self.physics_stop_fidelity<self.physics_start_fidelity<=1):
            raise ValueError("physics fidelity hysteresis is invalid")
        if self.physics_authorization_patience<1 or self.ritz_gradient_authorization_patience<1:
            raise ValueError("authorization patience values must be positive")
        if self.physics_ramp_epochs<1:
            raise ValueError("physics_ramp_epochs must be positive")
        if not (0<=self.ritz_gradient_start_fidelity<=1 and 0<=self.ritz_gradient_start_projector_loss<=1):
            raise ValueError("Ritz-gradient authorization thresholds must lie in [0,1]")
        if self.ritz_gradient_min_relative_gap<=0 or not 0<=self.ritz_gradient_min_safe_fraction<=1:
            raise ValueError("Ritz gradient gap/safe-fraction settings are invalid")
        if self.preset.refinement_train_steps<0 or self.preset.refinement_inference_steps<self.preset.refinement_train_steps:
            raise ValueError("refinement inference depth must be at least the non-negative training depth")
        if not (0<self.refinement_damping<=1 and self.refinement_preconditioner_floor>0 and self.refinement_max_correction_norm>0):
            raise ValueError("refinement damping, preconditioner floor, and trust norm must be positive")
        if not (0<=self.refinement_exploration_scale<1 and 0<=self.refinement_learned_multiplier<=1):
            raise ValueError("refinement exploration and learned multipliers must lie in [0,1]")
        if self.refinement_acceptance_tolerance<0 or self.refinement_backtrack_steps<1:
            raise ValueError("refinement acceptance tolerance/backtracking settings are invalid")
        if not 0<self.refinement_contraction_target<=1 or self.refinement_stop_residual<=0:
            raise ValueError("refinement contraction and stopping settings are invalid")
        if self.refinement_stagnation_tolerance<0 or self.refinement_stagnation_patience<1:
            raise ValueError("refinement stagnation settings are invalid")

    @property
    def preset(self) -> ModePreset:
        return PRESETS[self.mode]

    @property
    def tiny_overfit_policy(self) -> TinyOverfitPolicy:
        return TINY_OVERFIT_POLICIES[self.mode]

    @property
    def unseen_per_family(self) -> int:
        if self.ood_unseen_per_family > 0:
            return self.ood_unseen_per_family
        return {"QUICK":1,"DEVELOPMENT":16,"STANDARD":64,"FULL":128}[self.mode]


CFG = Config()
TRAINING_MICROBATCH_SIZE = int(os.environ.get("PSI_JEPA_TRAINING_MICROBATCH_SIZE", min(CFG.batch_size, 2)))
if not 1 <= TRAINING_MICROBATCH_SIZE <= CFG.batch_size:
    raise ValueError("PSI_JEPA_TRAINING_MICROBATCH_SIZE must lie in [1, CFG.batch_size]")
TRAINING_GRADIENT_ACCUMULATION = CFG.gradient_accumulation * math.ceil(CFG.batch_size / TRAINING_MICROBATCH_SIZE)
EVALUATION_BATCH_SIZE = int(os.environ.get("PSI_JEPA_EVALUATION_BATCH_SIZE", min(CFG.batch_size, 2)))
if not 1 <= EVALUATION_BATCH_SIZE <= CFG.batch_size:
    raise ValueError("PSI_JEPA_EVALUATION_BATCH_SIZE must lie in [1, CFG.batch_size]")
seed_everything(CFG.seed)
CONFIG_DICT = {**asdict(CFG), "preset": asdict(CFG.preset),
               "tiny_overfit_policy":asdict(CFG.tiny_overfit_policy),"tiny_overfit_seed":TINY_OVERFIT_SEED}
CONFIG_JSON = json.dumps(CONFIG_DICT, sort_keys=True, default=str)
CONFIG_HASH = hashlib.sha256(CONFIG_JSON.encode()).hexdigest()[:16]
CODE_HASH = hashlib.sha256(f"{ARCHITECTURE_VERSION}|{SOLVER_VERSION}|{SOURCE_REVISION}|{SOURCE_SHA256}".encode()).hexdigest()[:16]
EXPERIMENT_NAME = f"true_jepa_{CFG.mode.lower()}_{CONFIG_HASH[:8]}_{CODE_HASH[:8]}"
PROJECT_DIR = Path(os.environ.get("PSI_JEPA_PROJECT_DIR", Path.cwd())).resolve()
OUT = PROJECT_DIR / "outputs" / EXPERIMENT_NAME
if CFG.clean_output and OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "config.json").write_text(json.dumps(CONFIG_DICT, indent=2, default=str))
print(json.dumps({"run_mode": CFG.mode, "experiment": EXPERIMENT_NAME, "config_hash": CONFIG_HASH,
                  "code_hash": CODE_HASH, "source_sha256":SOURCE_SHA256,"output": str(OUT),
                  "training_microbatch_size":TRAINING_MICROBATCH_SIZE,
                  "evaluation_batch_size":EVALUATION_BATCH_SIZE,
                  "training_gradient_accumulation":TRAINING_GRADIENT_ACCUMULATION,
                  "deterministic_algorithms_enforced":torch.are_deterministic_algorithms_enabled()}, indent=2))
