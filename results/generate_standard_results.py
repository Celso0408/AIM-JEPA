"""Generate stored-data result tables and figures for the completed V12 STANDARD run.

No training or model inference is performed.  Every result is derived from files
already written by the pipeline, and outputs are namespaced by experiment name so
older result bundles are never silently overwritten.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "true_jepa_standard_fbc1677c_aee2f53f"
TRAINING_MICROBATCH_SIZE = 1
RUN = PROJECT / "outputs" / RUN_NAME
RESULTS = PROJECT / "results" / RUN_NAME
METRICS = RESULTS / "01_metrics"
FIGURES = RESULTS / "02_figures"
SPECTRAL = RESULTS / "03_spectral_analysis"

COPIED_METRICS = (
    "family_summary.csv",
    "family_state_summary.csv",
    "generalization_summary.csv",
    "geometry_summary.csv",
    "output_route_summary_by_category.csv",
    "output_route_summary_by_family.csv",
    "output_route_summary_by_geometry.csv",
    "per_state_metrics.csv",
    "per_state_output_metrics.csv",
    "refinement_depth_ablation.csv",
    "refinement_iteration_metrics.csv",
    "state_summary.csv",
    "test_results.csv",
    "topology_metrics.csv",
    "topology_summary_by_category.csv",
    "topology_summary_by_family.csv",
    "topology_summary_by_geometry.csv",
    "topology_summary_by_state.csv",
    "training_history.csv",
)


def require_completed_standard() -> tuple[dict, dict]:
    if not RUN.is_dir():
        raise FileNotFoundError(f"Missing STANDARD run: {RUN}")
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    card = json.loads((RUN / "run_card.json").read_text(encoding="utf-8"))
    if config.get("mode") != "STANDARD" or card.get("run_mode") != "STANDARD":
        raise RuntimeError("Selected run is not a STANDARD run")
    if card.get("experiment_name") != RUN_NAME:
        raise RuntimeError("Run-card experiment name does not match RUN_NAME")
    return config, card


def copy_metrics() -> list[str]:
    copied = []
    for name in COPIED_METRICS:
        source = RUN / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, METRICS / name)
        copied.append(name)
    return copied


def recover_autoencoder_history() -> None:
    history = json.loads((RUN / "solution_autoencoder_history.json").read_text(encoding="utf-8"))
    split = json.loads((RUN / "split_manifest.json").read_text(encoding="utf-8"))
    updates_per_epoch = (
        len(split["train"]) + TRAINING_MICROBATCH_SIZE - 1
    ) // TRAINING_MICROBATCH_SIZE
    frame = pd.DataFrame(
        {
            "stage": "B_solution_autoencoder",
            "epoch": [row["epoch"] for row in history],
            "train_loss": [row["training_loss_mean"] for row in history],
            "optimizer_update": [(row["epoch"] + 1) * updates_per_epoch for row in history],
        }
    )
    frame.to_csv(METRICS / "solution_autoencoder_history_recovered.csv", index=False)


def selected_route(data: pd.DataFrame) -> str:
    routes = set(data["output_route"].dropna())
    return "physics_selected" if "physics_selected" in routes else "direct_topology"


def save_heatmap(data: pd.DataFrame, category: str, label: str) -> dict:
    table = (
        data.query("category == @category")
        .groupby(["family", "state"])["fidelity"]
        .mean()
        .unstack("family")
    )
    if table.empty:
        raise RuntimeError(f"No rows for heatmap category {category!r}")
    table = table[table.mean().sort_values(ascending=False).index]
    table.to_csv(SPECTRAL / f"fidelity_heatmap_{label}.csv")
    fig, ax = plt.subplots(figsize=(max(8.0, 0.31 * table.shape[1]), 4.0))
    image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set(
        title=f"V12 STANDARD — {category}",
        xlabel="Hamiltonian family",
        ylabel=r"Spectral index $n$",
        xticks=range(table.shape[1]),
        xticklabels=[value.replace("_", " ") for value in table.columns],
        yticks=range(table.shape[0]),
        yticklabels=table.index,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    fig.colorbar(image, ax=ax, label="Mean fidelity")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fidelity_heatmap_{label}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"fidelity_heatmap_{label}.pdf", bbox_inches="tight")
    plt.close(fig)
    values = table.to_numpy(dtype=float)
    return {
        "category": category,
        "families": int(table.shape[1]),
        "states": int(table.shape[0]),
        "mean_fidelity": float(np.nanmean(values)),
        "minimum_fidelity": float(np.nanmin(values)),
        "cells_below_0_8": int(np.count_nonzero(values < 0.8)),
    }


def save_curriculum() -> None:
    history = pd.read_csv(RUN / "training_history.csv").sort_values("global_epoch")
    x = history["global_epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    panels = (
        ("train_loss", "Training loss"),
        ("validation_score", "Validation score"),
        ("physics_ramp", "Physics curriculum"),
        ("learning_rate", "Learning rate and gradient norm"),
    )
    for ax, (column, title) in zip(axes.flat, panels):
        ax.plot(x, history[column], "-o", ms=2.5, label=column)
        for _, part in history.groupby("stage", sort=False):
            ax.axvline(part.global_epoch.iloc[0], color="0.75", lw=0.7)
        ax.set(xlabel="Epoch", title=title)
        ax.legend(frameon=False, fontsize=8)
    if "physics_authorized" in history:
        axes[1, 0].plot(x, history.physics_authorized.astype(int), "--", label="physics_authorized")
        axes[1, 0].legend(frameon=False, fontsize=8)
    if "gradient_norm_post_clip" in history:
        twin = axes[1, 1].twinx()
        twin.plot(x, history.gradient_norm_post_clip, color="tab:red", label="gradient norm")
        twin.set_ylabel("Gradient norm")
    fig.suptitle(RUN_NAME)
    fig.savefig(FIGURES / f"curriculum_dashboard_{RUN_NAME}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"curriculum_dashboard_{RUN_NAME}.pdf", bbox_inches="tight")
    plt.close(fig)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def save_spectral_tension() -> int:
    keys = ["group_id", "family", "state"]
    columns = ["category", *keys, "output_route", "fidelity", "independent_residual_head"]
    data = pd.read_csv(RUN / "per_state_output_metrics.csv", usecols=columns)
    data = data[data.category.eq("ID_test") & data.output_route.isin(["direct_topology", "initial_ritz"])]
    direct = data[data.output_route.eq("direct_topology")].drop(columns=["category", "output_route"])
    ritz = data[data.output_route.eq("initial_ritz")].drop(columns=["category", "output_route"])
    pairs = direct.merge(ritz, on=keys, suffixes=("_direct", "_rr"), validate="one_to_one")

    topology_columns = [
        "category", *keys, "phase_crossing_count", "persistent_node_count_compliance",
        "minimum_grid_intervals_per_lobe", "minimum_predicted_lobe_probability_mass",
        "initial_ritz_topology_valid",
    ]
    topology = pd.read_csv(RUN / "topology_metrics.csv", usecols=topology_columns)
    topology = topology[topology.category.eq("ID_test")].copy()
    topology["topology_direct"] = (
        topology.phase_crossing_count.eq(topology.state)
        & topology.persistent_node_count_compliance.eq(1)
        & topology.minimum_grid_intervals_per_lobe.ge(3)
        & topology.minimum_predicted_lobe_probability_mass.ge(1e-5)
    ).astype(int)
    topology["topology_rr"] = topology.initial_ritz_topology_valid.astype(int)
    pairs = pairs.merge(topology[keys + ["topology_direct", "topology_rr"]], on=keys, validate="one_to_one")
    if not pairs.groupby("group_id").state.nunique().eq(11).all():
        raise RuntimeError("Incomplete state coverage in spectral comparison")

    rows = []
    z = 1.96
    for state, group in pairs.groupby("state", sort=True):
        for route in ("direct", "rr"):
            residual = group[f"independent_residual_head_{route}"]
            correct = group[f"topology_{route}"]
            low, high = wilson(int(correct.sum()), len(correct))
            rows.append(
                {
                    "n": int(state), "route": route, "count": len(group),
                    "mean_residual": residual.mean(),
                    "residual_ci95": z * residual.std(ddof=1) / np.sqrt(len(group)),
                    "topology_fraction": correct.mean(),
                    "topology_ci95_low": low, "topology_ci95_high": high,
                    "mean_fidelity": group[f"fidelity_{route}"].mean(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(SPECTRAL / "spectral_tension_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for route, color, marker, style, label in (
        ("direct", "#222222", "o", "-", "Direct"),
        ("rr", "#8b3a3a", "s", "--", "Rayleigh–Ritz"),
    ):
        part = summary[summary.route.eq(route)]
        axes[0].errorbar(part.n, part.mean_residual, yerr=part.residual_ci95,
                         color=color, marker=marker, ms=3.5, lw=1.15, capsize=2,
                         ls=style, label=label)
        error = np.maximum(0.0, np.vstack([
            part.topology_fraction - part.topology_ci95_low,
            part.topology_ci95_high - part.topology_fraction,
        ]))
        axes[1].errorbar(part.n, part.topology_fraction, yerr=error,
                         color=color, marker=marker, ms=3.5, lw=1.15, capsize=2,
                         ls=style, label=label)
    axes[0].set(xlabel=r"Spectral index $n$", ylabel="Mean normalized residual",
                xticks=range(11), ylim=(0, None))
    axes[1].set(xlabel=r"Spectral index $n$", ylabel="Resolved topology fraction",
                xticks=range(11), ylim=(-0.03, 1.03))
    for tag, ax in zip(("(a)", "(b)"), axes):
        ax.text(-0.13, 1.04, tag, transform=ax.transAxes, fontweight="bold")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(FIGURES / f"spectral_tension_{RUN_NAME}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"spectral_tension_{RUN_NAME}.pdf", bbox_inches="tight")
    plt.close(fig)
    return int(pairs.group_id.nunique())


def main() -> None:
    config, card = require_completed_standard()
    for directory in (METRICS, FIGURES, SPECTRAL):
        directory.mkdir(parents=True, exist_ok=True)
    copied = copy_metrics()
    recover_autoencoder_history()
    routes = pd.read_csv(RUN / "per_state_output_metrics.csv")
    route = selected_route(routes)
    selected = routes.query("output_route == @route")
    heatmaps = [
        save_heatmap(selected, "ID_test", "regular_id"),
        save_heatmap(selected, "unseen_analytical_functional_forms", "regular_ood_unseen"),
    ]
    save_curriculum()
    paired_hamiltonians = save_spectral_tension()
    manifest = {
        "experiment_name": RUN_NAME,
        "mode": config["mode"],
        "run_status": card.get("status"),
        "source_run": str(RUN),
        "selected_heatmap_route": route,
        "copied_metrics": copied,
        "heatmaps": heatmaps,
        "spectral_tension_paired_hamiltonians": paired_hamiltonians,
        "training_executed": False,
        "model_inference_executed": False,
        "physicless_standard_available": False,
        "physicless_note": "A matched V12 physicless STANDARD run does not exist; no cross-model comparison was generated.",
    }
    (RESULTS / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"RESULT GENERATION PASSED: {RESULTS}")


if __name__ == "__main__":
    main()
