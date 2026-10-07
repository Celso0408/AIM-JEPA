#!/usr/bin/env python3
"""Publication-style diagnostics for Periodic Field JEPA V31 training outputs.

Put this file beside the notebook's CSV/JSON/JSONL output files and run:

    python generate_periodic_jepa_figures_complete.py

Figures and an audit of what was/was not available are placed in ``figures/``.
The script never treats a rejected candidate or a validation metric as an
accepted checkpoint or held-out test result. Missing later stages are reported,
not extrapolated. Figure text (titles, axes, legends, annotations) is English.

Dependencies: numpy, pandas, matplotlib; optional: seaborn is NOT required.
    pip install numpy pandas matplotlib

Options: --run-dir PATH --out-dir PATH --formats png,pdf --dpi 220
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
import textwrap
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "periodic_jepa_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ORDER = [
    "autoencoder", "D_latent", "E_density", "E_density_potential", "E_full",
    "F_periodic_ramp", "P_cohesive_refinement", "G_cohesive_calibration",
    "G_adapter_warmup", "G_joint_finetune", "G_retention_repair",
]
SHORT = {
    "autoencoder": "AE", "D_latent": "D: JEPA", "E_density": "E1: density",
    "E_density_potential": "E2: potential", "E_full": "E3: full",
    "F_periodic_ramp": "F: periodic", "P_cohesive_refinement": "P: head candidate",
    "G_cohesive_calibration": "G: Ni calibration", "G_adapter_warmup": "G: adapter",
    "G_joint_finetune": "G: joint", "G_retention_repair": "G: repair",
}
C = {
    "ink": "#172b4d", "muted": "#52647c", "grid": "#e5ebf2",
    "blue": "#2876b8", "teal": "#008a88", "amber": "#d28b1f",
    "red": "#c44747", "purple": "#8162a8", "gray": "#8796aa",
    "green": "#328865", "Al": "#2876b8", "Fe": "#c77b23", "Ni": "#9155a3",
}
DOMAIN = {"Al": C["Al"], "Fe": C["Fe"], "Ni": C["Ni"]}
STAGE_SECTION = {
    "autoencoder": "01_autoencoder", "D_latent": "02_jepa_latent",
    "E_density": "03_curriculum", "E_density_potential": "03_curriculum",
    "E_full": "03_curriculum", "F_periodic_ramp": "03_curriculum",
    "P_cohesive_refinement": "04_energy_refinement",
    "G_cohesive_calibration": "05_ni_transfer", "G_adapter_warmup": "05_ni_transfer",
    "G_joint_finetune": "05_ni_transfer", "G_retention_repair": "05_ni_transfer",
}
CORE = {
    "autoencoder": ["rho", "potential", "reconstruction", "cohesive", "ae_anchor"],
    "D_latent": ["jepa", "latent_cosine", "electronic_reconstruction",
                 "electronic_anchor", "latent_variance_penalty", "latent_covariance_penalty"],
    "E_density": ["rho", "jepa", "latent_cosine", "kan"],
    "E_density_potential": ["rho", "potential", "jepa", "latent_cosine", "kan"],
    "E_full": ["rho", "potential", "cohesive", "jepa", "kan"],
    "F_periodic_ramp": ["rho", "potential", "cohesive", "jepa", "translation_consistency"],
    "P_cohesive_refinement": ["cohesive", "kan"],
    "G_cohesive_calibration": ["cohesive", "calibration_regularization", "retention_distillation"],
}
VALIDATION_NOTE = "Validation data; not held-out test performance."


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
        "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cad3df", "axes.grid": True,
        "grid.color": C["grid"], "grid.alpha": 0.85,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def name_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")


def n(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def col(df: pd.DataFrame, *options: str) -> str | None:
    for option in options:
        if option in df.columns:
            return option
    return None


def has_numeric(df: pd.DataFrame, name: str) -> bool:
    return name in df and n(df[name]).notna().any()


def truth(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def ordered(names) -> list[str]:
    return sorted(names, key=lambda key: (ORDER.index(key) if key in ORDER else 100, str(key)))


def axes_flat(axes) -> list[plt.Axes]:
    return list(np.asarray(axes).flatten())


def finish_axes(ax: plt.Axes, *, zero: bool = False) -> None:
    ax.grid(axis="y", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    if zero:
        ax.axhline(0, color=C["gray"], lw=0.9, zorder=0)


def heading(fig: plt.Figure, title: str, note: str = "") -> None:
    fig.suptitle(title, x=0.04, ha="left", fontsize=16, fontweight="bold")
    if note:
        # Put the scientific qualifier below the axes. bbox_inches='tight'
        # keeps it in exports without stealing label/tick space.
        fig.text(0.04, -0.07, textwrap.fill(note, width=105),
                 va="bottom", fontsize=8.3, color=C["muted"])


class Run:
    def __init__(self, path: Path):
        self.path = path
        self.history: dict[str, pd.DataFrame] = {}
        self.physical: dict[str, pd.DataFrame] = {}
        self.issues: list[str] = []

    def csv(self, filename: str, *, columns: list[str] | None = None) -> pd.DataFrame | None:
        path = self.path / filename
        if not path.is_file():
            return None
        try:
            return pd.read_csv(path, low_memory=False, usecols=columns)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            self.issues.append(f"Cannot read {filename}: {exc}")
            return None

    def json(self, filename: str) -> dict | None:
        path = self.path / filename
        if not path.is_file():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            self.issues.append(f"Cannot read {filename}: {exc}")
            return None

    def load(self) -> None:
        for path in self.path.glob("history_*.csv"):
            stage = path.stem.removeprefix("history_")
            frame = self.csv(path.name)
            if frame is not None and len(frame):
                self.history[stage] = frame.sort_values("epoch") if "epoch" in frame else frame
        for path in self.path.glob("physical_validation_*.csv"):
            stage = path.stem.removeprefix("physical_validation_")
            frame = self.csv(path.name)
            if frame is not None and len(frame):
                self.physical[stage] = frame
        # Older outputs have the narrower validation_physical_* schema.
        for path in self.path.glob("validation_physical_*.csv"):
            stage = path.stem.removeprefix("validation_physical_")
            if stage not in self.physical:
                frame = self.csv(path.name)
                if frame is not None and len(frame):
                    self.physical[stage] = frame


class Figures:
    def __init__(self, path: Path, formats: list[str], dpi: int):
        self.path, self.formats, self.dpi = path, formats, dpi
        self.records: list[dict[str, str]] = []
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, section: str, stem: str, source: str,
             interpretation: str, caveat: str = "") -> None:
        sub = self.path / section
        sub.mkdir(parents=True, exist_ok=True)
        for ext in self.formats:
            dest = sub / (stem + "." + ext)
            fig.savefig(dest, dpi=self.dpi, bbox_inches="tight", pad_inches=0.22)
            self.records.append({
                "file": str(dest.relative_to(self.path)), "source": source,
                "interpretation": interpretation, "caveat": caveat,
            })
        plt.close(fig)


def pair_panel(ax: plt.Axes, df: pd.DataFrame, x: pd.Series, metric: str,
               *, ylabel: str = "Loss", color: str = C["blue"]) -> bool:
    plotted = False
    for prefix, sty, alpha in [("train_", "-", 0.92), ("validation_", "--", 0.85)]:
        key = prefix + metric
        if has_numeric(df, key):
            ax.plot(x, n(df[key]), sty, lw=1.95, color=color, alpha=alpha,
                    label="Training" if prefix == "train_" else "Validation")
            plotted = True
    if plotted:
        ax.set_title(metric.replace("_", " ").capitalize(), loc="left")
        ax.set_ylabel(ylabel)
        finish_axes(ax)
        ax.legend(loc="best", fontsize=8)
    return plotted


def plot_stage_history(stage: str, df: pd.DataFrame, figs: Figures) -> None:
    section = STAGE_SECTION.get(stage, "03_curriculum")
    short = SHORT.get(stage, stage.replace("_", " "))
    source = "history_" + stage + ".csv"
    x = n(df["epoch"]) + 1 if "epoch" in df else np.arange(1, len(df) + 1)
    valid = VALIDATION_NOTE
    if has_numeric(df, "train_total") and has_numeric(df, "validation_total"):
        fig, ax = plt.subplots(figsize=(8.8, 4.5))
        ax.plot(x, n(df["train_total"]), lw=2.3, color=C["blue"], label="Training")
        ax.plot(x, n(df["validation_total"]), lw=2.3, color=C["amber"], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Total objective (as logged)")
        ax.legend(ncol=2)
        finish_axes(ax)
        heading(fig, f"{short}  |  total objective", valid + " Training and validation scales may reflect different stream weights.")
        figs.save(fig, section, name_slug(stage) + "_01_total_objective", source,
                  "Within-stage training and validation objectives", valid)

    parts = [part for part in CORE.get(stage, ["jepa", "rho", "potential", "cohesive"])
             if has_numeric(df, "train_" + part) or has_numeric(df, "validation_" + part)]
    if parts:
        fig, axs = plt.subplots((len(parts) + 1) // 2, 2, figsize=(12, 3.15 * ((len(parts) + 1) // 2)))
        for ax, part in zip(axes_flat(axs), parts):
            pair_panel(ax, df, x, part, color=C["teal"] if part in ("jepa", "latent_cosine") else C["blue"])
            ax.set_xlabel("Epoch")
        for ax in axes_flat(axs)[len(parts):]:
            ax.axis("off")
        heading(fig, f"{short}  |  objective components", valid + " Each panel has its own y-axis.")
        figs.save(fig, section, name_slug(stage) + "_02_components", source,
                  "Training and validation component losses", valid)

    terms = [key.removeprefix("train_effective_weight_") for key in df
             if key.startswith("train_effective_weight_") and has_numeric(df, key)]
    if terms:
        fig, ax = plt.subplots(figsize=(10.8, 5.0))
        for part in terms:
            ax.plot(x, n(df["train_effective_weight_" + part]), lw=1.75,
                    label=part.replace("_", " "))
        ax.set(xlabel="Epoch", ylabel="Effective objective weight")
        ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        finish_axes(ax)
        heading(fig, f"{short}  |  objective-weight schedule",
                "Weights are logged effective coefficients; curves show the objective ramp, not raw loss sizes.")
        figs.save(fig, section, name_slug(stage) + "_03_objective_weights", source,
                  "Effective objective-weight ramps")

    bands = [field for field in ("density", "potential")
             if any(has_numeric(df, f"validation_{field}_cosine_{band}") for band in ("low", "middle", "high"))]
    if bands:
        fig, axs = plt.subplots(1, len(bands), figsize=(6.2 * len(bands), 4.3), squeeze=False)
        for ax, field in zip(axes_flat(axs), bands):
            for band, color in (("low", C["blue"]), ("middle", C["teal"]), ("high", C["purple"])):
                key = f"validation_{field}_cosine_{band}"
                if has_numeric(df, key):
                    ax.plot(x, n(df[key]), marker="o", ms=2.8, lw=1.8, color=color, label=band.title())
            ax.set(title=field.title(), xlabel="Epoch", ylabel="Validation cosine similarity")
            ax.legend(title="Frequency band", ncol=3, fontsize=8)
            finish_axes(ax)
        heading(fig, f"{short}  |  spectral-band alignment", valid)
        figs.save(fig, section, name_slug(stage) + "_04_spectral_bands", source,
                  "Low/middle/high Fourier-band cosine similarities", valid)

    physical = [metric for metric in ("rho", "potential", "energy")
                if any(has_numeric(df, f"validation_physical_{el}_{metric}") for el in ("Al", "Fe", "Ni"))]
    if physical:
        fig, axs = plt.subplots(1, len(physical), figsize=(5.3 * len(physical), 4.5), squeeze=False)
        for ax, metric in zip(axes_flat(axs), physical):
            for el in ("Al", "Fe", "Ni"):
                key = f"validation_physical_{el}_{metric}"
                if has_numeric(df, key):
                    scale = 1000 if metric == "energy" else 1
                    ax.plot(x, n(df[key]) * scale, lw=2, color=DOMAIN[el], label=el)
            unit = "MAE (meV/atom)" if metric == "energy" else "Relative L2 error"
            ax.set(title=metric.title(), xlabel="Epoch", ylabel=unit)
            ax.legend()
            finish_axes(ax)
        heading(fig, f"{short}  |  physical validation", valid)
        figs.save(fig, section, name_slug(stage) + "_05_physical_validation", source,
                  "Element-wise validation errors for density, potential and energy", valid)

    competence = [(metric, element, f"competence_{element}_{metric}")
                  for metric in ("rho", "potential") for element in ("Al", "Fe", "Ni")
                  if has_numeric(df, f"competence_{element}_{metric}")]
    if competence:
        fig, axs = plt.subplots(1, 2, figsize=(11.3, 4.0))
        for ax, metric in zip(axs, ("rho", "potential")):
            for term, element, key in competence:
                if term == metric:
                    ax.plot(x, n(df[key]), marker="o", ms=2.4, lw=1.85,
                            color=DOMAIN[element], label=element)
            ax.set(xlabel="Epoch", ylabel="Competence relative L2", title=metric.title())
            ax.legend()
            finish_axes(ax)
        heading(fig, f"{short}  |  element-wise field competence", valid)
        figs.save(fig, section, name_slug(stage) + "_05b_field_competence", source,
                  "Element-wise density and potential gate diagnostics", valid)

    if has_numeric(df, "competence_periodic_consistency"):
        fig, ax = plt.subplots(figsize=(8.8, 4.0))
        ax.plot(x, n(df["competence_periodic_consistency"]), marker="o", lw=2,
                color=C["purple"], label="Measured consistency error")
        ax.set(xlabel="Epoch", ylabel="Periodic consistency error")
        ax.legend()
        finish_axes(ax)
        heading(fig, f"{short}  |  periodic consistency",
                "This metric is reported by the F-stage competence audit; lower indicates stronger consistency.")
        figs.save(fig, section, name_slug(stage) + "_05c_periodic_consistency", source,
                  "Periodic-consistency validation trajectory", valid)

    if "readiness_streak" in df and "readiness_required" in df:
        fig, axs = plt.subplots(2, 1, figsize=(9.4, 5.9), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
        axs[0].plot(x, n(df["readiness_streak"]), marker="o", lw=2, color=C["blue"], label="Consecutive passes")
        axs[0].plot(x, n(df["readiness_required"]), "--", lw=1.6, color=C["red"], label="Required")
        axs[0].set_ylabel("Consecutive validations")
        axs[0].legend()
        for key, label, height, color in (
            ("stage_readiness_passed", "Readiness passed", 0.84, C["green"]),
            ("checkpoint_accepted", "Checkpoint accepted", 0.37, C["blue"]),
        ):
            if key in df:
                xx = np.asarray(x)[truth(df[key]).to_numpy()]
                axs[1].scatter(xx, np.full(len(xx), height), marker="|", s=320,
                               linewidth=2.5, color=color, label=label)
        axs[1].set(ylim=(0.0, 1.1), xlabel="Epoch", yticks=[], ylabel="Events")
        axs[1].legend(loc="upper left", ncol=2)
        for ax in axs:
            finish_axes(ax)
        reasons = df["checkpoint_reason"].astype(str).value_counts().head(2) if "checkpoint_reason" in df else pd.Series(dtype=int)
        reason_note = "; ".join(f"{key} (n={count})" for key, count in reasons.items())
        heading(fig, f"{short}  |  readiness and checkpoint decisions",
                "Observed decision counts: " + reason_note)
        figs.save(fig, section, name_slug(stage) + "_06_readiness", source,
                  "Consecutive readiness streak and actual accepted-checkpoint events")

    lr_keys = [key for key in df if key.startswith("train_lr_") and key.endswith("_mean") and has_numeric(df, key)]
    if lr_keys:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        for key in lr_keys:
            ax.plot(x, n(df[key]), lw=1.8, label=key[9:-5].replace("_", " "))
        ax.set(xlabel="Epoch", ylabel="Mean learning rate")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.legend(ncol=2, fontsize=8)
        finish_axes(ax)
        heading(fig, f"{short}  |  module learning rates")
        figs.save(fig, section, name_slug(stage) + "_07_learning_rates", source,
                  "Stage-level mean learning rate for each trainable parameter group")

    gaps = [part for part in CORE.get(stage, []) if has_numeric(df, "train_" + part)
            and has_numeric(df, "validation_" + part) and "cosine" not in part]
    if gaps:
        fig, ax = plt.subplots(figsize=(9.6, 4.5))
        for part in gaps:
            ax.plot(x, n(df["validation_" + part]) - n(df["train_" + part]),
                    lw=1.8, label=part.replace("_", " "))
        ax.axhline(0, color=C["gray"], lw=1)
        ax.set(xlabel="Epoch", ylabel="Validation minus training loss")
        ax.legend(ncol=2, fontsize=8)
        finish_axes(ax)
        heading(fig, f"{short}  |  train–validation differences",
                "Descriptive gap only; differing sampling/weighting can affect direct comparison.")
        figs.save(fig, section, name_slug(stage) + "_08_generalization_gap", source,
                  "Validation minus training losses; not a held-out generalization estimate", valid)

    if has_numeric(df, "ema_tau"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
        axs[0].plot(x, n(df["ema_tau"]), color=C["purple"], lw=2)
        axs[0].set(xlabel="Epoch", ylabel="EMA decay τ", title="EMA coefficient")
        key = col(df, "field_target_ema_updates", "global_alfe_updates")
        if key and has_numeric(df, key):
            axs[1].plot(x, n(df[key]), color=C["teal"], lw=2)
            axs[1].set(xlabel="Epoch", ylabel="Cumulative updates", title="Target encoder updates")
        else:
            axs[1].axis("off")
        for ax in axs:
            finish_axes(ax)
        heading(fig, f"{short}  |  target-encoder EMA",
                "A flat update count indicates a frozen target; it is not evidence of learning during that stage.")
        figs.save(fig, section, name_slug(stage) + "_09_ema", source,
                  "EMA decay and target update count")

    if has_numeric(df, "epoch_seconds"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        axs[0].bar(x, n(df["epoch_seconds"]) / 60, color=C["blue"])
        axs[0].set(xlabel="Epoch", ylabel="Minutes", title="Wall time per epoch")
        if has_numeric(df, "training_records_per_second"):
            axs[1].plot(x, n(df["training_records_per_second"]), color=C["teal"], lw=2)
            axs[1].set(xlabel="Epoch", ylabel="Records/s", title="Training throughput")
        else:
            axs[1].axis("off")
        for ax in axs:
            finish_axes(ax)
        heading(fig, f"{short}  |  training throughput")
        figs.save(fig, section, name_slug(stage) + "_10_throughput", source,
                  "Elapsed time and throughput by epoch")


def plot_overview(run: Run, figs: Figures) -> None:
    frames = [(stage, run.history[stage]) for stage in ordered(run.history)]
    if not frames:
        return
    rows = []
    for stage, df in frames:
        updates = n(df["optimizer_updates_this_epoch"]).sum() if "optimizer_updates_this_epoch" in df else np.nan
        minutes = n(df["epoch_seconds"]).sum() / 60 if "epoch_seconds" in df else np.nan
        rows.append({"stage": stage, "label": SHORT.get(stage, stage), "epochs": len(df),
                     "updates": updates, "minutes": minutes,
                     "ready": truth(df["stage_readiness_passed"]).any() if "stage_readiness_passed" in df else False,
                     "accepted": truth(df["checkpoint_accepted"]).any() if "checkpoint_accepted" in df else False})
    summary = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(13.2, max(4.5, 0.55 * len(rows) + 1)))
    yy = np.arange(len(rows))
    for ax, key, xlabel, color in [(axs[0], "updates", "Optimizer updates", C["blue"]),
                                   (axs[1], "minutes", "Wall time (minutes)", C["teal"])]:
        ax.barh(yy, summary[key], color=color, height=0.7)
        ax.set(yticks=yy, yticklabels=summary["label"], xlabel=xlabel)
        ax.invert_yaxis()
        ax.grid(axis="x", color=C["grid"])
        ax.grid(axis="y", visible=False)
    heading(fig, "Training progress  |  computational cost",
            "Candidate stages are included even when their checkpoint was rejected.")
    figs.save(fig, "00_overview", "stage_costs", "history_*.csv",
              "Stage-local optimizer updates and wall time")

    fig, ax = plt.subplots(figsize=(10, max(4.4, len(rows) * 0.5 + 1)))
    for i, row in summary.iterrows():
        state = "Accepted" if row.accepted else "Ready, not accepted" if row.ready else "Not ready"
        color = C["green"] if row.accepted else C["amber"] if row.ready else C["red"]
        ax.barh(i, 1.0, color=color, height=0.65)
        ax.text(0.03, i, state, va="center", color="white", fontweight="bold")
    ax.set(yticks=yy, yticklabels=summary["label"], xlim=(0, 1), xticks=[])
    ax.invert_yaxis()
    ax.grid(False)
    heading(fig, "Training progress  |  observed checkpoint decisions",
            "Accepted = at least one accepted checkpoint in that stage; readiness and acceptance are distinct.")
    figs.save(fig, "00_overview", "stage_decisions", "history_*.csv",
              "Observed readiness and checkpoint acceptance, not inferred from stage name")

    watched = ["train_rho", "train_potential", "train_cohesive", "train_jepa",
               "train_latent_cosine", "train_kan", "train_translation_consistency",
               "train_retention_distillation"]
    active = [key for key in watched if any(key in df and n(df[key]).notna().any() for _, df in frames)]
    if active:
        values = np.full((len(frames), len(active)), np.nan)
        for i, (_, df) in enumerate(frames):
            for j, key in enumerate(active):
                if has_numeric(df, key):
                    values[i, j] = float(n(df[key]).iloc[-1])
        fig, ax = plt.subplots(figsize=(max(9, len(active) * 1.3), max(4.4, len(frames) * 0.55 + 1.8)))
        mask = np.isfinite(values)
        display = np.where(mask, 1, np.nan)
        ax.imshow(np.ma.masked_invalid(display), cmap=matplotlib.colors.ListedColormap([C["teal"]]),
                  vmin=0, vmax=1, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, "logged" if mask[i, j] else "—", ha="center", va="center",
                        color="white" if mask[i, j] else C["muted"], fontsize=8)
        ax.set(xticks=range(len(active)), xticklabels=[key[6:].replace("_", " ") for key in active],
               yticks=range(len(frames)), yticklabels=[SHORT.get(stage, stage) for stage, _ in frames])
        ax.tick_params(axis="x", rotation=32)
        ax.grid(False)
        heading(fig, "Curriculum  |  logged objective availability",
                "Presence of a metric does not imply a positive optimization weight; see stage weight plots.")
        figs.save(fig, "00_overview", "objective_availability", "history_*.csv",
                  "Availability of logged objective components by curriculum stage")


def plot_ae(run: Run, figs: Figures) -> None:
    gate = run.csv("autoencoder_field_gate.csv")
    initial = run.csv("autoencoder_field_gate_initial_spectrum.csv")
    config = (run.json("run_configuration_contract.json") or {}).get("config", {})
    if gate is not None and not gate.empty:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
        for ax, (metric, label, limit_key) in zip(axs, (
            ("rho_relative_l2", "Density relative L2", "ae_rho_relative_l2_max"),
            ("potential_relative_l2", "Potential relative L2", "ae_potential_relative_l2_max"),
        )):
            if metric not in gate:
                ax.axis("off"); continue
            elements = list(gate["element"])
            xx = np.arange(len(elements))
            width = 0.32
            if initial is not None and metric in initial:
                old = initial.set_index("element")[metric].reindex(elements)
                ax.bar(xx - width/2, old, width, color=C["gray"], label="Initial spectrum")
                ax.bar(xx + width/2, gate[metric], width, color=C["blue"], label="Final AE")
            else:
                ax.bar(xx, gate[metric], width=0.55, color=C["blue"], label="Final AE")
            if limit_key in config:
                ax.axhline(float(config[limit_key]), ls="--", color=C["red"], lw=1.4,
                           label="Configured gate")
            ax.set(xticks=xx, xticklabels=elements, ylabel=label)
            ax.legend(fontsize=8)
            finish_axes(ax)
        heading(fig, "Autoencoder  |  field reconstruction gates", VALIDATION_NOTE)
        figs.save(fig, "01_autoencoder", "ae_field_gates", "autoencoder_field_gate*.csv",
                  "AE density and potential field errors against configured limits", VALIDATION_NOTE)

    eigen = run.csv("autoencoder_latent_covariance_eigenvalues.csv")
    audit = run.json("autoencoder_latent_audit.json") or {}
    if eigen is not None and "eigenvalue" in eigen:
        ev = n(eigen["eigenvalue"]).fillna(0).clip(lower=0).sort_values(ascending=False).to_numpy()
        if ev.sum() > 0:
            rank = np.arange(1, len(ev) + 1)
            fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
            axs[0].plot(rank, ev, color=C["blue"], lw=1.5)
            axs[0].set(xlabel="Covariance eigenvalue rank", ylabel="Eigenvalue", yscale="symlog")
            axs[1].plot(rank, 100 * ev.cumsum()/ev.sum(), color=C["teal"], lw=2)
            axs[1].axhline(90, ls="--", color=C["amber"], lw=1, label="90% variance")
            axs[1].set(xlabel="Number of components", ylabel="Cumulative variance (%)", ylim=(0, 102))
            axs[1].legend()
            for ax in axs:
                finish_axes(ax)
            note = (f"Effective rank: {audit.get('effective_rank', float('nan')):.1f}; "
                    f"active covariance rank: {audit.get('active_covariance_rank', 'n/a')}. "
                    "Eigenvalues are from the AE audit sample, not the full dataset.")
            heading(fig, "Autoencoder  |  latent covariance spectrum", note)
            figs.save(fig, "01_autoencoder", "ae_latent_eigenspectrum",
                      "autoencoder_latent_covariance_eigenvalues.csv; autoencoder_latent_audit.json",
                      "AE covariance spectrum and cumulative explained variance",
                      "Audit sample only; covariance eigenvalues are not per-mode Fourier amplitudes")

    fit = run.csv("diagnostic_overfit_history.csv")
    if fit is not None and len(fit):
        fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.0))
        for val, label, ax, color in (("loss", "Diagnostic loss", axs[0], C["blue"]),
                                       ("cosine_full", "Full-latent cosine", axs[1], C["teal"])):
            if val in fit:
                for translation, grp in fit.groupby("translations"):
                    ax.plot(grp["step"], n(grp[val]), lw=1.9, label=f"{translation} translations")
                ax.set(xlabel="Diagnostic optimization step", ylabel=label)
                ax.legend(fontsize=8)
                finish_axes(ax)
        heading(fig, "Autoencoder / JEPA  |  small-set fit diagnostic",
                "Training-only fit check; not validation and not a generalization result.")
        figs.save(fig, "01_autoencoder", "small_set_overfit_diagnostic", "diagnostic_overfit_history.csv",
                  "Fit capacity under diagnostic conditions", "Training-only, cloned-model diagnostic")


def plot_alignment(run: Run, figs: Figures) -> None:
    gate = run.csv("D_latent_alignment_gate.csv")
    if gate is None or gate.empty:
        return
    gate = gate.copy()
    gate["label"] = gate["element"].astype(str) + " · " + gate["field"].astype(str)
    xx = np.arange(len(gate))
    bands = [("low", C["blue"]), ("middle", C["teal"]), ("high", C["purple"])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for j, (band, color) in enumerate(bands):
        key = "cosine_by_band." + band
        if key in gate:
            ax.scatter(xx + (j-1)*0.13, gate[key], s=70, color=color, label=band.title())
    ax.plot(xx, n(gate["cosine_similarity"]), "x", ms=7, color=C["ink"], label="Full spectrum")
    ax.set(xticks=xx, xticklabels=gate["label"], ylabel="Predicted–target cosine similarity")
    ax.legend(ncol=4, fontsize=8)
    finish_axes(ax)
    heading(fig, "JEPA  |  validation spectral alignment", VALIDATION_NOTE)
    figs.save(fig, "02_jepa_latent", "jepa_alignment_by_band", "D_latent_alignment_gate.csv",
              "Full-spectrum and band-resolved alignment by element and field", VALIDATION_NOTE)

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axs[0].bar(xx - 0.25, gate["predicted_effective_rank"], 0.25, label="Predicted", color=C["blue"])
    axs[0].bar(xx, gate["target_effective_rank"], 0.25, label="Target", color=C["teal"])
    axs[0].bar(xx + 0.25, gate["required_rank"], 0.25, label="Required minimum", color=C["gray"])
    axs[0].set(ylabel="Effective rank", xticks=xx, xticklabels=gate["label"])
    axs[0].legend(fontsize=8)
    axs[1].bar(xx - 0.16, gate["predicted_dominant_fraction"], 0.32,
               label="Predicted", color=C["blue"])
    axs[1].bar(xx + 0.16, gate["target_dominant_fraction"], 0.32,
               label="Target", color=C["teal"])
    axs[1].plot(xx, gate["allowed_dominant_fraction"], "--", color=C["red"],
                label="Maximum allowed")
    axs[1].set(ylabel="Largest component / total variance", xticks=xx, xticklabels=gate["label"])
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.tick_params(axis="x", rotation=20)
        finish_axes(ax)
    heading(fig, "JEPA  |  effective rank and collapse checks", VALIDATION_NOTE)
    figs.save(fig, "02_jepa_latent", "jepa_rank_and_dominance", "D_latent_alignment_gate.csv",
              "Predicted/target rank and dominant-fraction diagnostics", VALIDATION_NOTE)


def canonical_physical(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {"rho": "rho_relative_l2", "potential": "potential_relative_l2",
               "rho_cosine": "rho_cosine_similarity", "potential_cosine": "potential_cosine_similarity",
               "energy": "cohesive_abs_error_ev_per_atom", "energy_signed": "cohesive_signed_error_ev_per_atom",
               "energy_target": "cohesive_target_ev_per_atom", "energy_prediction": "cohesive_pred_ev_per_atom",
               "weak": "cohesive_weak_binding"}
    return df.rename(columns={key: value for key, value in aliases.items() if value not in df})


def plot_physical_per_stage(stage: str, raw: pd.DataFrame, figs: Figures) -> None:
    df = canonical_physical(raw)
    if "element" not in df or not len(df):
        return
    section = STAGE_SECTION.get(stage, "03_curriculum")
    short, source = SHORT.get(stage, stage), ("physical_validation_" + stage + ".csv")
    if stage == "P_cohesive_refinement":
        note = "Candidate evaluation only; the P checkpoint was not accepted in the supplied run."
    else:
        note = VALIDATION_NOTE
    metrics = [("rho_relative_l2", "Density relative L2", 1),
               ("potential_relative_l2", "Potential relative L2", 1),
               ("cohesive_abs_error_ev_per_atom", "Energy absolute error (meV/atom)", 1000)]
    present = [(key, label, factor) for key, label, factor in metrics if key in df]
    if present:
        fig, axs = plt.subplots(1, len(present), figsize=(5 * len(present), 4.3), squeeze=False)
        for ax, (key, label, factor) in zip(axes_flat(axs), present):
            elems = [el for el in ("Al", "Fe", "Ni") if el in df.element.unique()]
            values = [n(df.loc[df.element == el, key]).dropna() * factor for el in elems]
            parts = ax.violinplot(values, positions=np.arange(len(elems)), showmeans=False,
                                  showmedians=True, showextrema=False)
            for body, el in zip(parts["bodies"], elems):
                body.set_facecolor(DOMAIN[el]); body.set_alpha(0.65)
            parts["cmedians"].set_color(C["ink"])
            ax.set(xticks=np.arange(len(elems)), xticklabels=elems, ylabel=label)
            finish_axes(ax)
        heading(fig, f"{short}  |  per-configuration error distributions", note)
        figs.save(fig, section, name_slug(stage) + "_physical_distributions", source,
                  "Validation sample distributions by element", note)

    target, pred = "cohesive_target_ev_per_atom", "cohesive_pred_ev_per_atom"
    if target in df and pred in df:
        fig, axs = plt.subplots(1, 2, figsize=(12.1, 4.7))
        lo, hi = np.inf, -np.inf
        for el, grp in df.groupby("element"):
            color = DOMAIN.get(str(el), C["gray"])
            x, y = n(grp[target]), n(grp[pred])
            axs[0].scatter(x, y, s=24, alpha=0.55, label=str(el), color=color, rasterized=True)
            axs[1].scatter(x, 1000*(y-x), s=24, alpha=0.55, label=str(el), color=color, rasterized=True)
            lo = min(lo, x.min(), y.min()); hi = max(hi, x.max(), y.max())
        if np.isfinite(lo) and np.isfinite(hi):
            axs[0].plot([lo, hi], [lo, hi], "--", color=C["gray"], lw=1.2, label="Parity")
        axs[0].set(xlabel="Reference cohesive energy (eV/atom)", ylabel="Predicted cohesive energy (eV/atom)")
        axs[1].axhline(0, ls="--", color=C["gray"])
        axs[1].set(xlabel="Reference cohesive energy (eV/atom)", ylabel="Signed prediction error (meV/atom)")
        for ax in axs:
            ax.legend(fontsize=8)
            finish_axes(ax)
        heading(fig, f"{short}  |  energy parity and residuals", note)
        figs.save(fig, section, name_slug(stage) + "_energy_parity_and_residuals", source,
                  "Energy predictions and signed residuals by element", note)

    if "family" in df and present:
        fig, axs = plt.subplots(1, len(present), figsize=(5.2*len(present), 4.5), squeeze=False)
        for ax, (key, label, factor) in zip(axes_flat(axs), present):
            tab = df.assign(_value=n(df[key])*factor).pivot_table(
                index="family", columns="element", values="_value", aggfunc="mean")
            im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
            ax.set(xticks=range(len(tab.columns)), xticklabels=tab.columns,
                   yticks=range(len(tab.index)), yticklabels=tab.index, title=label)
            for i in range(len(tab.index)):
                for j in range(len(tab.columns)):
                    if np.isfinite(tab.iloc[i, j]):
                        ax.text(j, i, f"{tab.iloc[i, j]:.3g}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04)
            ax.grid(False)
        heading(fig, f"{short}  |  mean error by structural family", note)
        figs.save(fig, section, name_slug(stage) + "_family_error_matrix", source,
                  "Mean validation errors by structural family and element", note)

    if "cohesive_weak_binding" in df and "cohesive_abs_error_ev_per_atom" in df:
        grp = df.assign(_weak=truth(df["cohesive_weak_binding"]),
                        _mae=n(df["cohesive_abs_error_ev_per_atom"])*1000)
        tab = grp.groupby(["element", "_weak"], observed=True)["_mae"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(8, 4.0))
        elems = list(grp.element.unique())
        xx = np.arange(len(elems))
        for weak, shift, color, label in [(False, -0.18, C["blue"], "Other"),
                                           (True, 0.18, C["amber"], "Weak binding")]:
            means = [tab.loc[(el, weak), "mean"] if (el, weak) in tab.index else np.nan for el in elems]
            count = [int(tab.loc[(el, weak), "count"]) if (el, weak) in tab.index else 0 for el in elems]
            ax.bar(xx+shift, means, 0.35, color=color, label=label)
            for pos, value, num in zip(xx+shift, means, count):
                if np.isfinite(value):
                    ax.text(pos, value, f"n={num}", ha="center", va="bottom", fontsize=7)
        ax.set(xticks=xx, xticklabels=elems, ylabel="Mean absolute error (meV/atom)")
        ax.legend()
        finish_axes(ax)
        heading(fig, f"{short}  |  weak-binding error", note)
        figs.save(fig, section, name_slug(stage) + "_weak_binding", source,
                  "Mean energy error in weak-binding and other samples with counts", note)


def plot_cross_stage_physical(run: Run, figs: Figures) -> None:
    frames = [(stage, canonical_physical(run.physical[stage])) for stage in ordered(run.physical)]
    if not frames:
        return
    measurements = [("rho_relative_l2", "Density relative L2", 1),
                    ("potential_relative_l2", "Potential relative L2", 1),
                    ("cohesive_abs_error_ev_per_atom", "Energy MAE (meV/atom)", 1000)]
    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.5))
    xx = np.arange(len(frames))
    for ax, (key, label, factor) in zip(axs, measurements):
        for element in ("Al", "Fe", "Ni"):
            ys = [n(df.loc[df.element == element, key]).mean() * factor
                  if key in df and "element" in df and (df.element == element).any() else np.nan
                  for _, df in frames]
            if np.isfinite(ys).any():
                ax.plot(xx, ys, marker="o", lw=1.9, color=DOMAIN[element], label=element)
        ax.set(xticks=xx, xticklabels=[SHORT.get(stage, stage) for stage, _ in frames],
               ylabel=label, xlabel="Evaluation checkpoint / candidate")
        ax.tick_params(axis="x", rotation=50)
        ax.legend()
        finish_axes(ax)
    heading(fig, "Physical performance  |  evaluation sequence",
            "Stage P is a rejected candidate in the supplied run; dots are validation means, not test metrics.")
    figs.save(fig, "00_overview", "physical_metrics_across_stages", "physical_validation_*.csv",
              "Stage-by-stage element-wise field and energy validation metrics",
              "No P checkpoint adoption inferred from candidate evaluation")

    for a, b in zip(frames, frames[1:]):
        left, right = a[1], b[1]
        if not {"calc_id", "element"}.issubset(left) or not {"calc_id", "element"}.issubset(right):
            continue
        keys = ["calc_id", "element"]
        present = [(key, label, factor) for key, label, factor in measurements if key in left and key in right]
        if not present:
            continue
        merged = left[keys+[key for key, _, _ in present]].merge(
            right[keys+[key for key, _, _ in present]], on=keys, suffixes=("_before", "_after"),
            validate="one_to_one")
        if len(merged) < 8:
            continue
        fig, axs = plt.subplots(1, len(present), figsize=(5.1*len(present), 4.4), squeeze=False)
        for ax, (key, label, factor) in zip(axes_flat(axs), present):
            for element, grp in merged.groupby("element"):
                vals = (n(grp[key+"_after"]) - n(grp[key+"_before"])) * factor
                ax.boxplot([vals.dropna()], positions=[list(merged.element.unique()).index(element)],
                           widths=0.4, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": DOMAIN.get(str(element), C["gray"]), "alpha": 0.55},
                           medianprops={"color": C["ink"]})
            ax.axhline(0, color=C["gray"], lw=1)
            labels = list(merged.element.unique())
            ax.set(xticks=range(len(labels)), xticklabels=labels,
                   ylabel="After minus before: " + label)
            finish_axes(ax)
        heading(fig, f"Paired change  |  {SHORT.get(a[0], a[0])} → {SHORT.get(b[0], b[0])}",
                f"Matched calc_id + element only (n={len(merged)}). Negative error change means improvement; {VALIDATION_NOTE}")
        figs.save(fig, "06_comparisons", f"paired_{name_slug(a[0])}_to_{name_slug(b[0])}",
                  f"physical_validation_{a[0]}.csv; physical_validation_{b[0]}.csv",
                  "Paired per-configuration changes rather than unpaired mean differences",
                  "A rejected candidate is a comparison, not an accepted model")


def plot_curriculum(run: Run, figs: Figures) -> None:
    history = run.csv("data_curriculum_history.csv")
    if history is not None and len(history) and {"stage", "epoch"}.issubset(history):
        segments = []
        offset = 0
        for stage in ordered(history.stage.unique()):
            grp = history[history.stage == stage].sort_values("epoch").copy()
            grp["sequence_epoch"] = np.arange(offset + 1, offset + len(grp) + 1)
            segments.append((stage, grp))
            offset += len(grp)
        fig, axs = plt.subplots(2, 1, figsize=(13.5, 6.0), sharex=True)
        for idx, (stage, grp) in enumerate(segments):
            lo, hi = grp.sequence_epoch.min()-.5, grp.sequence_epoch.max()+.5
            for ax in axs:
                ax.axvspan(lo, hi, color=C["gray"], alpha=.07 if idx % 2 else .025)
                ax.axvline(lo, color=C["grid"], lw=1)
            if "difficulty_progress" in grp:
                axs[0].plot(grp.sequence_epoch, n(grp.difficulty_progress),
                            color=C["teal"], lw=2)
            if "active_unique_records" in grp:
                axs[1].plot(grp.sequence_epoch, n(grp.active_unique_records),
                            color=C["blue"], lw=2)
            axs[0].text((lo+hi)/2, 1.035, SHORT.get(stage,stage), ha="center", va="bottom",
                        fontsize=7.3, rotation=30, color=C["muted"], clip_on=False)
        axs[0].set(ylabel="Difficulty progress", ylim=(-.02, 1.07))
        axs[1].set(xlabel="Cumulative logged epoch", ylabel="Unique active records")
        for ax in axs:
            finish_axes(ax)
        heading(fig, "Data curriculum  |  chronological progression",
                "Shaded bands mark stages; a smaller Ni pool reflects the 10% selected adaptation data.")
        figs.save(fig, "03_curriculum", "curriculum_chronological_timeline",
                  "data_curriculum_history.csv", "Stage-continuous difficulty and active-record timeline")

        metrics = [("active_fraction", "Active fraction"), ("active_unique_records", "Unique active records"),
                   ("difficulty_progress", "Difficulty progress"),
                   ("full_data_epochs_completed", "Full-data epochs completed")]
        fig, axs = plt.subplots(2, 2, figsize=(12.4, 7.0))
        for ax, (key, label) in zip(axes_flat(axs), metrics):
            if key not in history:
                ax.axis("off"); continue
            for stage in ordered(history.stage.unique()):
                grp = history[history.stage == stage].sort_values("epoch")
                ax.plot(grp.epoch + 1, n(grp[key]), marker="o", ms=2.5, lw=1.5,
                        label=SHORT.get(stage, stage))
            ax.set(xlabel="Epoch within stage", ylabel=label)
            finish_axes(ax)
        axs[0, 0].legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(1.02, -0.18))
        heading(fig, "Data curriculum  |  coverage and difficulty",
                "Active fraction is logged batch-composition coverage, not necessarily excluded records.")
        figs.save(fig, "03_curriculum", "curriculum_coverage_and_difficulty", "data_curriculum_history.csv",
                  "Coverage, unique records, difficulty and full-data exposure by stage")

        fig, ax = plt.subplots(figsize=(10.7, 4.1))
        if "curriculum_tier" in history:
            tier = history.groupby(["stage", "curriculum_tier"], observed=True).size().unstack(fill_value=0)
            tier = tier.reindex(ordered(tier.index))
            bottom = np.zeros(len(tier))
            for label in tier.columns:
                ax.bar(range(len(tier)), tier[label].to_numpy(), bottom=bottom, label=str(label))
                bottom += tier[label].to_numpy()
            ax.set(xticks=range(len(tier)), xticklabels=[SHORT.get(k,k) for k in tier.index],
                   ylabel="Logged epochs by curriculum tier")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(ncol=3, fontsize=8)
            finish_axes(ax)
            heading(fig, "Data curriculum  |  tier exposure")
            figs.save(fig, "03_curriculum", "curriculum_tier_exposure", "data_curriculum_history.csv",
                      "Epoch counts under each logged curriculum tier")
        else:
            plt.close(fig)

    for source in ("data_curriculum_manifest_pretrain_AlFe.csv",
                   "data_curriculum_manifest_Ni_selected_only.csv"):
        df = run.csv(source)
        if df is None or not {"element", "family", "physical_difficulty"}.issubset(df):
            continue
        fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.4))
        for element, grp in df.groupby("element"):
            axs[0].hist(n(grp.physical_difficulty).dropna(), bins=30, density=True,
                        histtype="step", lw=2, color=DOMAIN.get(element,C["gray"]), label=element)
        axs[0].set(xlabel="Physical difficulty descriptor", ylabel="Probability density")
        axs[0].legend()
        fam = df.groupby(["element", "family"], observed=True).size().unstack(fill_value=0)
        bottom = np.zeros(len(fam))
        for j, family in enumerate(fam.columns):
            axs[1].bar(fam.index, fam[family], bottom=bottom, label=str(family),
                       color=[C["blue"], C["teal"], C["amber"], C["purple"]][j % 4])
            bottom += fam[family].to_numpy()
        axs[1].set(xlabel="Element", ylabel="Records by structural family")
        axs[1].legend(ncol=3, fontsize=8)
        for ax in axs:
            finish_axes(ax)
        label = "Ni selected training pool" if "Ni_" in source else "Al/Fe pretraining pool"
        heading(fig, f"Data curriculum  |  {label}", "Training selection only; no representativeness claim against unseen Ni test data.")
        figs.save(fig, "03_curriculum", name_slug(source.removesuffix(".csv")), source,
                  "Selected training-pool difficulty distribution and family composition")

    exposure = run.csv("sample_stage_exposure.csv")
    if exposure is not None and {"stage", "element", "draws"}.issubset(exposure):
        group = exposure.groupby(["stage", "element"], observed=True).agg(
            draws=("draws", "sum"), records=("calc_id", "nunique"))
        stages = ordered(exposure.stage.unique())
        fig, axs = plt.subplots(1, 2, figsize=(13.3, 4.9))
        xx = np.arange(len(stages)); width = 0.25
        for idx, el in enumerate(("Al", "Fe", "Ni")):
            vals = [group.loc[(s, el), "draws"] if (s, el) in group.index else 0 for s in stages]
            axs[0].bar(xx+(idx-1)*width, vals, width, label=el, color=DOMAIN[el])
            coverage = [group.loc[(s, el), "records"] if (s, el) in group.index else 0 for s in stages]
            axs[1].bar(xx+(idx-1)*width, coverage, width, label=el, color=DOMAIN[el])
        for ax, ylabel in zip(axs, ("Total sample draws", "Unique configurations exposed")):
            ax.set(xticks=xx, xticklabels=[SHORT.get(s,s) for s in stages], ylabel=ylabel)
            ax.tick_params(axis="x", rotation=35)
            ax.legend(ncol=3)
            finish_axes(ax)
        heading(fig, "Data curriculum  |  actual sample exposure",
                "Draws include repeat exposure; unique configurations count distinct calc_id values.")
        figs.save(fig, "03_curriculum", "curriculum_actual_exposure", "sample_stage_exposure.csv",
                  "Element-wise draws and distinct configurations by stage")

        if "physical_difficulty" in exposure:
            fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
            for el, grp in exposure.groupby("element"):
                axs[0].scatter(n(grp.physical_difficulty), n(grp.draws), s=5, alpha=0.12,
                               color=DOMAIN.get(el,C["gray"]), label=str(el), rasterized=True)
            axs[0].set(xlabel="Physical difficulty descriptor", ylabel="Sample draws")
            axs[0].legend(markerscale=3)
            temp = exposure.copy()
            temp["difficulty_bin"] = pd.qcut(n(temp.physical_difficulty), 10, duplicates="drop")
            binned = temp.groupby("difficulty_bin", observed=True).agg(
                difficulty=("physical_difficulty", "mean"), draws=("draws", "mean"))
            axs[1].plot(binned.difficulty, binned.draws, marker="o", color=C["teal"])
            axs[1].set(xlabel="Mean difficulty within decile", ylabel="Mean draws")
            for ax in axs:
                finish_axes(ax)
            heading(fig, "Data curriculum  |  difficulty versus exposure",
                    "Pooled records from observed stages; repeated stage records are intentionally distinct observations.")
            figs.save(fig, "03_curriculum", "curriculum_difficulty_vs_exposure",
                      "sample_stage_exposure.csv", "Sample draw allocation over difficulty")


def plot_trainability(run: Run, figs: Figures) -> None:
    df = run.csv("trainability_by_stage.csv")
    if df is None or not {"stage", "module", "trainable_parameters", "parameters"}.issubset(df):
        return
    stages = ordered(df.stage.unique())
    pct = df.assign(pct=np.where(n(df.parameters)>0,
                                 100*n(df.trainable_parameters)/n(df.parameters), np.nan))
    tab = pct.pivot_table(index="module", columns="stage", values="pct", aggfunc="first").reindex(columns=stages)
    fig, ax = plt.subplots(figsize=(max(10, len(stages)*1.25), max(5, len(tab)*0.38+1.8)))
    im = ax.imshow(tab.values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set(xticks=range(len(tab.columns)), xticklabels=[SHORT.get(s,s) for s in tab.columns],
           yticks=range(len(tab.index)), yticklabels=[m.replace("_", " ") for m in tab.index])
    ax.tick_params(axis="x", rotation=35)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Trainable parameters within module (%)", fraction=0.02)
    heading(fig, "Architecture  |  module freeze / unfreeze matrix",
            "Entries show the fraction of each module marked trainable for each observed stage.")
    figs.save(fig, "00_overview", "architecture_trainability_matrix", "trainability_by_stage.csv",
              "Which parameter groups can update in each curriculum stage")

    totals = df.groupby("stage", observed=True).agg(total=("parameters", "sum"),
                                                    trainable=("trainable_parameters", "sum")).reindex(stages)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    xx = np.arange(len(totals))
    ax.bar(xx, totals.trainable/1e6, color=C["teal"], label="Trainable")
    ax.bar(xx, (totals.total-totals.trainable)/1e6, bottom=totals.trainable/1e6,
           color="#d9e2eb", label="Frozen")
    ax.set(xticks=xx, xticklabels=[SHORT.get(s,s) for s in totals.index],
           ylabel="Parameters (million)")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(ncol=2)
    finish_axes(ax)
    heading(fig, "Architecture  |  trainable parameter budget",
            "Counts are the sum of logged disjoint module entries, not a FLOP or memory estimate.")
    figs.save(fig, "00_overview", "architecture_parameter_budget", "trainability_by_stage.csv",
              "Trainable and frozen parameter counts per observed stage")


def plot_optimizer(run: Run, figs: Figures) -> None:
    for stage in ordered(run.history):
        filename = "optimizer_history_" + stage + ".csv"
        path = run.path / filename
        if not path.is_file():
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns
            cols = [c for c in header if c in ("epoch", "stage_optimizer_update", "loss_total", "ema_tau")
                    or c.startswith(("lr_", "grad_norm_", "clip_"))]
            df = run.csv(filename, columns=cols)
        except (OSError, ValueError) as exc:
            run.issues.append(f"Cannot inspect {filename}: {exc}")
            continue
        if df is None or df.empty or "epoch" not in df:
            continue
        section = "07_optimization"
        short = SHORT.get(stage, stage)
        x = n(df.epoch) + 1
        groups = [("lr_", "Learning rate", C["blue"]),
                  ("grad_norm_", "Gradient norm", C["purple"])]
        for prefix, label, color in groups:
            keys = [c for c in df if c.startswith(prefix) and has_numeric(df, c)]
            if not keys:
                continue
            fig, ax = plt.subplots(figsize=(9.5, 4.3))
            for key in keys:
                by_epoch = df.groupby("epoch", observed=True)[key].median()
                ax.plot(by_epoch.index+1, by_epoch, marker="o", ms=2.4, lw=1.8,
                        label=key.removeprefix(prefix).replace("_", " "))
            ax.set(xlabel="Epoch", ylabel="Median update-level " + label.lower())
            ax.legend(ncol=2, fontsize=8)
            if prefix == "lr_":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            finish_axes(ax)
            heading(fig, f"{short}  |  optimizer {label.lower()}",
                    "Median across logged optimizer updates in each epoch; this is not an epoch maximum.")
            figs.save(fig, section, name_slug(stage)+"_optimizer_"+prefix[:-1], filename,
                      "Median per-epoch optimizer group trajectories")

        clip = [c for c in df if c.endswith("_clipped") and c.startswith("clip_")]
        if clip:
            fig, ax = plt.subplots(figsize=(9.5, 4.3))
            for key in clip:
                value = truth(df[key]).groupby(df.epoch).mean() * 100
                ax.plot(value.index+1, value.values, marker="o", lw=1.7,
                        label=key.removeprefix("clip_").removesuffix("_clipped").replace("_", " "))
            ax.set(xlabel="Epoch", ylabel="Updates clipped (%)", ylim=(0, 105))
            ax.legend(ncol=2, fontsize=8)
            finish_axes(ax)
            heading(fig, f"{short}  |  gradient clipping frequency")
            figs.save(fig, section, name_slug(stage)+"_optimizer_clipping", filename,
                      "Fraction of optimizer updates clipped in each parameter group")

        if has_numeric(df, "loss_total"):
            aggregate = df.groupby("epoch", observed=True)["loss_total"].agg(
                median="median", q10=lambda s: s.quantile(.1), q90=lambda s: s.quantile(.9))
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.fill_between(aggregate.index+1, aggregate.q10, aggregate.q90,
                            alpha=0.2, color=C["blue"], label="10–90% update range")
            ax.plot(aggregate.index+1, aggregate["median"], color=C["blue"], lw=2,
                    label="Update median")
            ax.set(xlabel="Epoch", ylabel="Update-level training loss")
            ax.legend()
            finish_axes(ax)
            heading(fig, f"{short}  |  optimizer-level loss spread",
                    "Update-level distribution, not an uncertainty interval for generalization.")
            figs.save(fig, section, name_slug(stage)+"_optimizer_loss_spread", filename,
                      "Median and 10–90th percentiles of update-level loss")

    conflicts = run.csv("gradient_conflicts.csv")
    if conflicts is not None and {"stage", "gradient_cosine", "objective_a", "objective_b", "conflict"}.issubset(conflicts):
        conflicts["pair"] = conflicts.objective_a.astype(str) + " × " + conflicts.objective_b.astype(str)
        stages = ordered(conflicts.stage.unique())
        if "norm_a" in conflicts and "norm_b" in conflicts:
            active = conflicts[(n(conflicts.norm_a)>0) & (n(conflicts.norm_b)>0)].copy()
        else:
            active = conflicts.copy()
        if len(active):
            tab = active.assign(_conflict=truth(active.conflict)).groupby(
                ["stage", "pair"], observed=True)["_conflict"].mean().unstack()
            tab = tab.reindex(stages)
            tab = tab.loc[:, tab.notna().any(axis=0)]
            fig, ax = plt.subplots(figsize=(max(10, .72*len(tab.columns)+4), max(4.8, .43*len(tab)+2)))
            cmap = plt.cm.Reds.copy(); cmap.set_bad("#edf2f7")
            im = ax.imshow(np.ma.masked_invalid(tab.values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set(xticks=range(len(tab.columns)), xticklabels=tab.columns,
                   yticks=range(len(tab.index)), yticklabels=[SHORT.get(s,s) for s in tab.index])
            ax.tick_params(axis="x", rotation=43)
            ax.grid(False)
            fig.colorbar(im, ax=ax, label="Observed conflict frequency", fraction=.025)
            heading(fig, "Multi-objective optimization  |  gradient conflicts",
                    "Only records with non-zero norms on both objectives; blank = pair not evaluated.")
            figs.save(fig, "07_optimization", "gradient_conflict_matrix", "gradient_conflicts.csv",
                      "Observed pairwise gradient conflict frequency under non-zero gradients")

            fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
            by_stage = [n(active.loc[active.stage == s, "gradient_cosine"]).dropna() for s in stages]
            axs[0].boxplot(by_stage, tick_labels=[SHORT.get(s,s) for s in stages], showfliers=False,
                           medianprops={"color": C["red"]})
            axs[0].axhline(0, color=C["gray"], lw=1)
            axs[0].set(ylabel="Pairwise gradient cosine")
            axs[0].tick_params(axis="x", rotation=35)
            rate = active.assign(conflict_bool=truth(active.conflict)).groupby("stage").conflict_bool.mean().reindex(stages)*100
            axs[1].bar(range(len(rate)), rate, color=C["amber"])
            axs[1].set(xticks=range(len(rate)), xticklabels=[SHORT.get(s,s) for s in stages],
                       ylabel="Conflicting pairs (%)", ylim=(0, 100))
            axs[1].tick_params(axis="x", rotation=35)
            for ax in axs:
                finish_axes(ax)
            heading(fig, "Multi-objective optimization  |  gradient geometry",
                    "Zero-norm pairs are excluded because a zero cosine does not establish conflict.")
            figs.save(fig, "07_optimization", "gradient_cosine_and_conflict_rate", "gradient_conflicts.csv",
                      "Non-zero objective pair cosine distributions and conflict rate")

    invalid = run.path / "invalid_optimizer_updates.jsonl"
    if invalid.is_file():
        try:
            rows = [json.loads(line) for line in invalid.read_text().splitlines() if line.strip()]
            if rows:
                df = pd.json_normalize(rows)
                group = col(df, "stage")
                if group:
                    counts = df[group].value_counts().reindex(ordered(df[group].unique()))
                    fig, ax = plt.subplots(figsize=(8, 3.7))
                    ax.bar(counts.index, counts.values, color=C["red"])
                    ax.set(ylabel="Logged invalid updates")
                    ax.tick_params(axis="x", rotation=30)
                    finish_axes(ax)
                    heading(fig, "Optimizer  |  invalid / non-finite updates",
                            "Event counts only; inspect the JSONL for root cause and recovery.")
                    figs.save(fig, "07_optimization", "invalid_optimizer_updates", invalid.name,
                              "Invalid optimizer events by stage")
        except (OSError, json.JSONDecodeError) as exc:
            run.issues.append(f"Cannot read {invalid.name}: {exc}")


def plot_ni_and_refinement(run: Run, figs: Figures) -> None:
    df = run.history.get("G_cohesive_calibration")
    config = (run.json("run_configuration_contract.json") or {}).get("config", {})
    if df is not None:
        x = n(df.epoch) + 1 if "epoch" in df else np.arange(1,len(df)+1)
        specs = [
            ("ni_physical_cohesive_mae_ev_per_atom", "MAE", .10),
            ("ni_physical_cohesive_absolute_bias_ev_per_atom", "Absolute bias", config.get("final_energy_bias_goal_ev", .02)),
            ("ni_physical_cohesive_error_p95_ev_per_atom", "P95 absolute error", .30),
            ("ni_physical_cohesive_max_absolute_error_ev_per_atom", "Maximum absolute error", .60),
        ]
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.9))
        for ax, (key, title, limit) in zip(axes_flat(axs), specs):
            if not has_numeric(df, key):
                ax.axis("off"); continue
            values = n(df[key])*1000
            ax.plot(x, values, marker="o", color=C["red"] if "bias" in key else C["blue"], lw=2)
            ax.axhline(1000*float(limit), ls="--", color=C["red"], lw=1.25,
                       label=f"V31 gate: {1000*float(limit):g} meV/atom")
            ax.set(xlabel="Calibration epoch", ylabel="meV/atom", title=title)
            ax.legend(fontsize=8)
            finish_axes(ax)
        heading(fig, "Ni transfer  |  energy gate trajectories",
                "V31 fixed limits: MAE 100, bias 20, P95 300, maximum 600 meV/atom; bias goal read from run config when present.")
        figs.save(fig, "05_ni_transfer", "ni_calibration_energy_gates", "history_G_cohesive_calibration.csv; run_configuration_contract.json",
                  "Measured Ni energy errors against V31 gate limits", VALIDATION_NOTE)

        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, key, label, color in zip(axs,
                ("ni_energy_scale", "ni_energy_offset_ev"),
                ("Ni affine scale", "Ni affine offset (meV)"),
                (C["blue"], C["purple"])):
            if has_numeric(df, key):
                factor = 1000 if key.endswith("_ev") else 1
                ax.plot(x, n(df[key])*factor, marker="o", color=color, lw=2)
                ax.set(xlabel="Calibration epoch", ylabel=label)
                finish_axes(ax)
            else:
                ax.axis("off")
        heading(fig, "Ni transfer  |  learned calibration scalars",
                "These parameters do not show that the shared encoder or cohesive head changed.")
        figs.save(fig, "05_ni_transfer", "ni_affine_parameters", "history_G_cohesive_calibration.csv",
                  "Ni energy-only affine scale and offset")

        for group, keys, unit in [
            ("fields", ["ni_physical_rho_relative_l2_mean", "ni_physical_potential_relative_l2_mean",
                        "ni_physical_latent_cosine_mean"], "Logged metric"),
            ("energy", ["ni_physical_cohesive_mae_mev_per_atom",
                        "ni_physical_cohesive_rmse_mev_per_atom",
                        "ni_physical_cohesive_absolute_bias_mev_per_atom"], "meV/atom"),
        ]:
            present = [key for key in keys if has_numeric(df, key)]
            if not present:
                continue
            fig, axs = plt.subplots(len(present), 1, figsize=(9.4, 2.7*len(present)), sharex=True)
            for ax, key in zip(axes_flat(axs), present):
                ax.plot(x, n(df[key]), marker="o", lw=1.9, color=C["teal"])
                ax.set(ylabel=key.removeprefix("ni_physical_").replace("_", " "))
                finish_axes(ax)
            axes_flat(axs)[-1].set_xlabel("Calibration epoch")
            heading(fig, f"Ni transfer  |  validation {group}",
                    f"Each panel has its own scale; energy panel units: {unit}. {VALIDATION_NOTE}")
            figs.save(fig, "05_ni_transfer", "ni_"+group+"_diagnostics", "history_G_cohesive_calibration.csv",
                      "Detailed Ni physical validation diagnostics", VALIDATION_NOTE)

        keys = [k for k in df if k.startswith("retention_") and k.endswith("_excess") and has_numeric(df,k)]
        if keys:
            max_excess = df[keys].apply(n).max(axis=1)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, max_excess, marker="o", color=C["teal"], lw=2)
            ax.axhline(0, ls="--", lw=1, color=C["red"], label="Retention bound")
            ax.set(xlabel="Calibration epoch", ylabel="Largest signed retention excess")
            ax.legend()
            finish_axes(ax)
            heading(fig, "Ni transfer  |  worst Al/Fe retention excess",
                    "Above zero means a logged retention limit was exceeded; no positive excess is not a Ni energy pass.")
            figs.save(fig, "05_ni_transfer", "ni_retention_max_excess", "history_G_cohesive_calibration.csv",
                      "Worst Al/Fe retention excess during Ni calibration")

    retention = run.csv("retention_metrics_by_component.csv")
    if retention is not None and {"call", "element", "metric", "excess"}.issubset(retention):
        subset = retention.copy()
        subset["label"] = subset.element.astype(str) + " · " + subset.metric.astype(str)
        pivot = subset.pivot_table(index="label", columns="call", values="excess", aggfunc="max")
        if len(pivot) and len(pivot.columns):
            arr = pivot.to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            extent = max(abs(finite.min()), abs(finite.max()), 1e-9) if len(finite) else 1
            fig, ax = plt.subplots(figsize=(max(8, .75*len(pivot.columns)+4), max(5, .32*len(pivot)+2)))
            im = ax.imshow(np.ma.masked_invalid(arr), cmap="RdBu_r",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-extent, vmax=extent), aspect="auto")
            ax.set(yticks=range(len(pivot)), yticklabels=pivot.index,
                   xticks=range(len(pivot.columns)), xticklabels=pivot.columns)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(False)
            fig.colorbar(im, ax=ax, label="Excess above tolerance", fraction=.02)
            heading(fig, "Source retention  |  component audit",
                    "Positive values indicate exceeded tolerance; negative values retain safety margin.")
            figs.save(fig, "05_ni_transfer", "al_fe_retention_heatmap", "retention_metrics_by_component.csv",
                      "Metric-specific retention excess by element and evaluation call")

    if df is not None:
        fig, axs = plt.subplots(1, 2, figsize=(11.3, 4.2))
        for ax, key, title in zip(axs, ("mae", "bias"),
                                  ("Source MAE", "Source signed bias")):
            for element in ("Al", "Fe"):
                metric = f"{element}_{key}_ev_per_atom"
                if has_numeric(df, metric):
                    ax.plot(x, n(df[metric])*1000, marker="o", lw=2,
                            color=DOMAIN[element], label=element)
            ax.set(xlabel="Ni calibration epoch", ylabel="meV/atom", title=title)
            if key == "bias":
                ax.axhline(0, lw=1, color=C["gray"])
            ax.legend()
            finish_axes(ax)
        heading(fig, "Ni transfer  |  Al/Fe energy retention",
                "Validation metrics for the source elements during Ni-only calibration; not held-out test performance.")
        figs.save(fig, "05_ni_transfer", "ni_source_energy_retention", "history_G_cohesive_calibration.csv",
                  "Al and Fe energy MAE/bias during Ni calibration", VALIDATION_NOTE)


def plot_protocol_and_splits(run: Run, figs: Figures) -> None:
    split = run.json("split_manifest.json") or {}
    records = split.get("records", {})
    counts = {key: value.get("count") for key, value in records.items()
              if isinstance(value, dict) and isinstance(value.get("count"), (int, float))}
    if counts:
        labels = list(counts)
        colors = [C["purple"] if "ood" in key else C["gray"] if "test" in key else
                  C["amber"] if "validation" in key else C["blue"] for key in labels]
        fig, ax = plt.subplots(figsize=(10.2, max(4.2, 0.47*len(labels)+1)))
        y = np.arange(len(labels))
        ax.barh(y, [counts[key] for key in labels], color=colors)
        ax.set(yticks=y, yticklabels=[key.replace("_", " ") for key in labels],
               xlabel="Number of configurations")
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        heading(fig, "Data protocol  |  split sizes",
                "These are split counts only. Test/OOD labels do not imply the model was evaluated on them.")
        figs.save(fig, "08_audits", "split_counts", "split_manifest.json",
                  "Sizes of configured train, validation, ID test and family-OOD splits",
                  "Test/OOD performance is unavailable")

    ni = run.json("ni_finetune_data_audit.json") or {}
    eligible, selected = ni.get("eligible_ni_id"), ni.get("unique_training_ni")
    if isinstance(eligible, (int, float)) and isinstance(selected, (int, float)) and eligible > 0:
        fig, ax = plt.subplots(figsize=(7.3, 3.6))
        ax.barh(["Eligible Ni ID pool"], [eligible], color="#dce4ec", label="Eligible")
        ax.barh(["Eligible Ni ID pool"], [selected], color=C["purple"], label="Selected for adaptation")
        ax.text(selected + eligible*.02, 0, f"{selected:,} / {eligible:,} ({100*selected/eligible:.1f}%)",
                va="center", color=C["ink"])
        ax.set(xlim=(0,eligible*1.26), xlabel="Distinct configurations")
        ax.legend()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        heading(fig, "Ni adaptation  |  available training fraction",
                "Selected fraction is a data constraint, not an accuracy or generalization measurement.")
        figs.save(fig, "05_ni_transfer", "ni_adaptation_fraction", "ni_finetune_data_audit.json",
                  "Ni training configurations versus eligible ID pool")

    source = run.json("source_domain_readiness.json") or {}
    metrics = source.get("diagnostics", {})
    energy = [(element, part, metrics.get(f"{element}_{part}_ev_per_atom"))
              for element in ("Al", "Fe") for part in ("mae", "bias", "p95", "maximum")]
    if any(isinstance(value, (int,float)) for _, _, value in energy):
        fig, ax = plt.subplots(figsize=(9.5, 4.1))
        xx = np.arange(4)
        for el, shift in (("Al", -0.18), ("Fe", 0.18)):
            values = [metrics.get(f"{el}_{part}_ev_per_atom", np.nan)*1000
                      for part in ("mae", "bias", "p95", "maximum")]
            ax.bar(xx+shift, values, width=0.36, label=el, color=DOMAIN[el])
        ax.set(xticks=xx, xticklabels=["MAE", "Signed bias", "P95", "Maximum"],
               ylabel="Energy error (meV/atom)")
        ax.legend()
        finish_axes(ax)
        heading(fig, "Source readiness  |  Al/Fe energy diagnostics",
                "Values come from the source-domain readiness audit; biases retain their sign.")
        figs.save(fig, "00_overview", "source_domain_energy_readiness", "source_domain_readiness.json",
                  "Al/Fe energy diagnostics at source readiness", VALIDATION_NOTE)


def plot_transition_protocol(run: Run, figs: Figures) -> None:
    path = run.path / "stage_transition_audit.jsonl"
    if not path.is_file():
        return
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        run.issues.append(f"Cannot read {path.name}: {exc}")
        return
    rows = [row for row in rows if isinstance(row, dict) and row.get("to_stage")]
    if not rows:
        return
    targets = [str(row["to_stage"]) for row in rows]
    terms = ["jepa", "latent_cosine", "rho", "potential", "cohesive",
             "translation_consistency", "retention_distillation"]
    values = np.array([[row.get("objective_weights_end", {}).get(term, np.nan)
                        for term in terms] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, max(4.6, .5*len(rows)+1.5)))
    im = ax.imshow(np.ma.masked_invalid(values), cmap="YlGnBu", vmin=0,
                   vmax=max(1, np.nanmax(values)), aspect="auto")
    ax.set(xticks=range(len(terms)), xticklabels=[term.replace("_", " ") for term in terms],
           yticks=range(len(targets)), yticklabels=[SHORT.get(t,t) for t in targets])
    ax.tick_params(axis="x", rotation=35)
    for i in range(len(targets)):
        for j in range(len(terms)):
            if np.isfinite(values[i,j]):
                ax.text(j, i, f"{values[i,j]:.2g}", ha="center", va="center", fontsize=7,
                        color="white" if values[i,j] > .5 else C["ink"])
    fig.colorbar(im, ax=ax, label="Target end-of-stage objective weight", fraction=.025)
    ax.grid(False)
    heading(fig, "Curriculum  |  objective targets at transitions",
            "Recorded target weights, not measured losses; transition audit is available only for entered stages.")
    figs.save(fig, "03_curriculum", "transition_objective_targets", path.name,
              "End-of-stage objective targets from the transition protocol")

    band_rows = []
    for row in rows:
        spectral = row.get("spectral_competence_state", {})
        for field in ("density", "potential"):
            state = spectral.get(field, {})
            for band in ("middle", "high"):
                value = state.get(f"{band}_unlocked_at")
                if isinstance(value, (int,float)):
                    band_rows.append((row["to_stage"], field, band, value))
    if band_rows:
        fig, ax = plt.subplots(figsize=(9, 4.0))
        for field, marker in (("density", "o"), ("potential", "s")):
            for band, color in (("middle", C["teal"]), ("high", C["purple"])):
                subset = [item for item in band_rows if item[1] == field and item[2] == band]
                if subset:
                    ax.scatter([targets.index(item[0]) for item in subset],
                               [item[3] for item in subset], marker=marker, s=60,
                               color=color, label=f"{field.title()} · {band}")
        ax.set(xticks=range(len(targets)), xticklabels=[SHORT.get(t,t) for t in targets],
               ylabel="Recorded unlock progress (fraction)", xlabel="Entered stage")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(ncol=2, fontsize=8)
        finish_axes(ax)
        heading(fig, "JEPA  |  spectral-band unlock audit",
                "Unlock position comes from recorded spectral competence state; repeated points reflect retained state.")
        figs.save(fig, "02_jepa_latent", "spectral_band_unlock_events", path.name,
                  "Recorded middle/high spectral unlock progress")


def plot_additional_audits(run: Run, figs: Figures) -> None:
    res = run.csv("spectral_resolution_validation.csv")
    if res is not None and {"element", "modes"}.issubset(res):
        metrics = [("rho_relative_l2", "Density relative L2"),
                   ("potential_relative_l2", "Potential relative L2")]
        fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.0))
        for ax, (key, ylabel) in zip(axs, metrics):
            if key not in res:
                ax.axis("off"); continue
            data = res.groupby(["element", "modes"], observed=True)[key].agg(
                median="median", q25=lambda s:s.quantile(.25), q75=lambda s:s.quantile(.75))
            for el in res.element.unique():
                if el not in data.index.get_level_values(0):
                    continue
                part = data.loc[el]
                ax.plot(part.index, part["median"], marker="o", lw=2,
                        label=el, color=DOMAIN.get(el,C["gray"]))
                ax.fill_between(part.index.astype(float), part.q25, part.q75,
                                color=DOMAIN.get(el,C["gray"]), alpha=.15)
            ax.set(xlabel="Number of spectral modes", ylabel=ylabel)
            ax.legend()
            finish_axes(ax)
        heading(fig, "Spectral resolution  |  field error",
                "Median with interquartile range over audited structures, not a retraining ablation.")
        figs.save(fig, "08_audits", "spectral_resolution_sensitivity", "spectral_resolution_validation.csv",
                  "Error across available Fourier-mode counts")

    audit = run.csv("data_audit.csv")
    if audit is not None and {"element", "native_charge_error", "canonical_charge_error"}.issubset(audit):
        fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.2))
        for el, grp in audit.groupby("element"):
            axs[0].scatter(np.abs(n(grp.native_charge_error)),
                           np.abs(n(grp.canonical_charge_error)), s=20,
                           color=DOMAIN.get(el,C["gray"]), alpha=.7, label=el)
        axs[0].set(xlabel="Native charge absolute error (e)", ylabel="Canonical charge absolute error (e)")
        axs[0].legend()
        if "potential_mean_ry" in audit:
            for el, grp in audit.groupby("element"):
                axs[1].hist(n(grp.potential_mean_ry).dropna(), bins=20, histtype="step",
                            lw=1.8, color=DOMAIN.get(el,C["gray"]), label=el)
            axs[1].set(xlabel="Potential mean (Ry)", ylabel="Audited records")
            axs[1].legend()
        else:
            axs[1].axis("off")
        for ax in axs:
            finish_axes(ax)
        heading(fig, "Input-data audit  |  charge and potential gauge",
                "Audit subset only; sample count and selection are defined in the data audit files.")
        figs.save(fig, "08_audits", "data_physical_integrity", "data_audit.csv",
                  "Input-record charge and potential-gauge checks")

    audit_path = run.path / "stage_transition_audit.jsonl"
    if audit_path.is_file():
        try:
            rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
            event = pd.json_normalize(rows)
            if "stage" in event and len(event):
                counts = event.stage.value_counts().reindex(ordered(event.stage.unique()))
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(counts.index, counts.values, color=C["blue"])
                ax.set(ylabel="Logged transition events")
                ax.tick_params(axis="x", rotation=32)
                finish_axes(ax)
                heading(fig, "Curriculum  |  stage-transition audit",
                        "Counts represent logged audit events, not optimizer updates or epochs.")
                figs.save(fig, "08_audits", "stage_transition_events", audit_path.name,
                          "Audit event counts by stage")
        except (OSError, json.JSONDecodeError) as exc:
            run.issues.append(f"Cannot read {audit_path.name}: {exc}")


def write_coverage(run: Run, figs: Figures) -> None:
    pd.DataFrame(figs.records).to_csv(figs.path / "figures_manifest.csv", index=False)
    missing = []
    for stage in ORDER:
        if stage not in run.history:
            missing.append("history_" + stage + ".csv")
    for artifact in ["physical_validation_G_cohesive_calibration.csv",
                     "ni_id_holdout_test.csv", "ni_family_ood_test.csv"]:
        if not (run.path/artifact).exists():
            missing.append(artifact)
    extension = ", ".join(figs.formats)
    report = [
        "# Periodic Field JEPA – figure coverage",
        "",
        f"Input directory: `{run.path}`",
        f"Figures: {len(figs.records)} files ({len(figs.records)//len(figs.formats)} distinct plots); formats: {extension}.",
        f"Observed stage histories: {', '.join(ordered(run.history)) or 'none'}.",
        f"Observed per-configuration physical validations: {', '.join(ordered(run.physical)) or 'none'}.",
        "",
        "## Interpretation boundaries",
        "",
        "- Every physical-validation curve in these files is validation, not held-out test or OOD performance.",
        "- `P_cohesive_refinement` is a candidate evaluation. Its checkpoint must not be treated as adopted unless the stage audit reports acceptance.",
        "- `G_cohesive_calibration` in this run has a history but no Ni per-configuration physical-validation CSV. Ni gate graphs use its epoch summaries.",
        "- Raw density/potential voxel maps, atomic/graph activations, individual Fourier coefficients, per-sample latent embeddings, causal interventions, baseline-model runs, and uncertainty calibration cannot be reconstructed from these summary files alone.",
        "- No downstream Ni joint-training, held-out test, or family-OOD performance is inferred from a missing output.",
        "",
        "## Missing stage / test inputs (conditional plots omitted)",
        "",
        *["- `"+item+"`" for item in missing],
        "",
        "## Read / plotting warnings",
        "",
        *(["- "+item for item in run.issues] if run.issues else ["- None."]),
    ]
    (figs.path / "analysis_coverage.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    entries = {}
    for item in figs.records:
        stem = Path(item["file"]).with_suffix("")
        key = str(stem)
        if key not in entries or item["file"].endswith(".png"):
            entries[key] = item
    groups = {}
    for item in entries.values():
        section = item["file"].split("/", 1)[0]
        groups.setdefault(section, []).append(item)
    toc = "".join(
        f'<a href="#{html.escape(section)}">{html.escape(section.replace("_", " "))} '
        f'<span>{len(items)}</span></a>' for section, items in groups.items()
    )
    blocks = []
    for section, items in groups.items():
        cards = []
        for item in items:
            filepath = html.escape(item["file"], quote=True)
            title = html.escape(Path(item["file"]).stem.replace("_", " ").title())
            source = html.escape(item["source"])
            description = html.escape(item["interpretation"])
            caveat = html.escape(item["caveat"])
            preview = (f'<img loading="lazy" src="{filepath}" alt="{title}">' if
                       item["file"].endswith((".png", ".svg")) else
                       '<div class="no-preview">Open figure</div>')
            caveat_html = '<small class="caveat">' + caveat + '</small>' if caveat else ''
            cards.append(f'<article class="card" data-search="{title} {description} {source}">'
                         f'<a href="{filepath}" target="_blank">{preview}</a>'
                         f'<div class="body"><h3>{title}</h3><p>{description}</p>'
                         f'<small>Source: {source}</small>'
                         f'{caveat_html}'
                         '</div></article>')
        blocks.append(f'<section id="{html.escape(section)}"><h2>{html.escape(section.replace("_", " ").title())}'
                      f' <em>{len(items)} figures</em></h2><div class="grid">{"".join(cards)}</div></section>')
    sections_html = "".join(blocks)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Periodic Field JEPA | Figure gallery</title>
<style>
:root{{--ink:#172b4d;--muted:#52647c;--blue:#2876b8;--line:#e2e9f0}}
*{{box-sizing:border-box}}body{{margin:0;background:#f5f8fb;color:var(--ink);font:15px/1.5 system-ui,Arial,sans-serif}}
header{{background:linear-gradient(120deg,#102748,#265784);color:white;padding:42px max(5vw,28px)}}
header h1{{margin:0 0 6px;font-size:clamp(25px,3vw,40px)}}header p{{max-width:950px;color:#dae7f3}}
main{{max-width:1650px;margin:auto;padding:24px max(2vw,18px)}}
.toolbar{{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:22px}}
input{{min-width:min(95vw,400px);padding:11px 13px;border:1px solid #c4d1df;border-radius:8px;font:inherit}}
.links{{display:flex;gap:8px;flex-wrap:wrap}}.links a{{color:var(--blue);text-decoration:none;background:white;border:1px solid var(--line);padding:7px 10px;border-radius:6px}}
.links span{{color:var(--muted);font-size:12px}}h2{{margin:38px 0 14px;font-size:23px;border-bottom:2px solid var(--line);padding-bottom:7px}}
h2 em{{font-size:13px;color:var(--muted);font-style:normal;font-weight:normal}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:20px}}
.card{{background:white;border:1px solid var(--line);border-radius:11px;overflow:hidden;box-shadow:0 3px 11px #1731510a}}
.card img{{width:100%;height:245px;object-fit:contain;padding:8px;background:#fff;border-bottom:1px solid var(--line)}}
.body{{padding:14px 18px 18px}}.body h3{{font-size:17px;margin:0 0 5px}}.body p{{margin:0 0 8px;color:var(--muted)}}
small{{display:block;color:var(--muted);overflow-wrap:anywhere}}.caveat{{color:#b25135;margin-top:5px}}
.no-preview{{height:245px;display:grid;place-items:center}}footer{{padding:35px;color:var(--muted);text-align:center}}
</style></head><body><header><h1>Periodic Field JEPA</h1><p>Training figure gallery · {len(entries)} distinct figures ·
English labels throughout. Validation, candidate checkpoints and held-out tests are explicitly distinguished.</p></header>
<main><div class="toolbar"><input id="filter" placeholder="Search figure, metric or source…" aria-label="Search figures">
<nav class="links">{toc}</nav></div>{sections_html}</main>
<footer>Read analysis_coverage.md for missing data and interpretation limits. Generated from files in the same run folder.</footer>
<script>document.getElementById('filter').addEventListener('input',e=>{{let q=e.target.value.toLowerCase();
document.querySelectorAll('.card').forEach(c=>c.style.display=c.dataset.search.toLowerCase().includes(q)?'':'none');}});</script>
</body></html>'''
    (figs.path / "gallery.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Input directory (default: directory containing this script)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <run-dir>/figures)")
    parser.add_argument("--formats", default="png", help="Comma-separated png,pdf,svg (default: png)")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution (default: 220)")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Input directory does not exist: {run_dir}")
    formats = list(dict.fromkeys(x.strip().lower() for x in args.formats.split(",") if x.strip()))
    if not formats or set(formats) - {"png", "pdf", "svg"}:
        parser.error("--formats accepts only png,pdf,svg")
    if args.dpi < 50 or args.dpi > 1200:
        parser.error("--dpi must be between 50 and 1200")
    style()
    run = Run(run_dir)
    run.load()
    if not run.history and not run.physical:
        parser.error("No history_*.csv or physical_validation_*.csv files found beside the script")
    figs = Figures((args.out_dir or run_dir/"figures").expanduser().resolve(), formats, args.dpi)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        plot_overview(run, figs)
        plot_trainability(run, figs)
        plot_protocol_and_splits(run, figs)
        plot_transition_protocol(run, figs)
        for stage in ordered(run.history):
            plot_stage_history(stage, run.history[stage], figs)
        plot_ae(run, figs)
        plot_alignment(run, figs)
        for stage in ordered(run.physical):
            plot_physical_per_stage(stage, run.physical[stage], figs)
        plot_cross_stage_physical(run, figs)
        plot_curriculum(run, figs)
        plot_optimizer(run, figs)
        plot_ni_and_refinement(run, figs)
        plot_additional_audits(run, figs)
    write_coverage(run, figs)
    print(f"Created {len(figs.records)//len(formats)} distinct plots in {figs.path}")
    print("Open gallery.html for visual browsing; analysis_coverage.md lists limitations.")


if __name__ == "__main__":
    main()
