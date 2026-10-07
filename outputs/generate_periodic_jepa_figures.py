#!/usr/bin/env python3
"""Generate diagnostic figures from a Periodic Field JEPA notebook run.

Usage
-----
Coloque este arquivo na mesma pasta dos CSV/JSON produzidos pelo notebook e execute:

    python generate_periodic_jepa_figures.py

Por padrão, os gráficos serão criados em ``./figures``. Opcionalmente, você
pode apontar outra pasta com ``--run-dir`` ou mudar a saída com ``--out-dir``.

The script looks for the CSV/JSON/JSONL artefacts emitted by the notebook and
only creates a figure when its input is available.  It therefore works for
partial, blocked, and completed runs alike.

Dependencies: pandas, numpy, matplotlib
    pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "periodic_jepa_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#475569",
    "light": "#e2e8f0",
}


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "axes.edgecolor": "#cbd5e1",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
        }
    )


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = {str(column).lower(): str(column) for column in frame.columns}
    for name in candidates:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


class FigureWriter:
    def __init__(self, output_dir: Path, formats: list[str], dpi: int) -> None:
        self.output_dir = output_dir
        self.formats = formats
        self.dpi = dpi
        self.records: list[dict[str, str]] = []

    def save(self, figure: plt.Figure, name: str, source: str, description: str) -> None:
        figure.tight_layout()
        for fmt in self.formats:
            destination = self.output_dir / f"{name}.{fmt}"
            figure.savefig(destination, dpi=self.dpi, bbox_inches="tight")
            self.records.append(
                {
                    "file": destination.name,
                    "source": source,
                    "description": description,
                }
            )
        plt.close(figure)


def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as error:  # malformed experimental artefacts should not stop plotting
        warnings.append(f"Não foi possível ler {path.name}: {error}")
        return None


def nice_stage(name: str) -> str:
    return name.replace("history_", "").replace("_", " ")


def numeric_columns(frame: pd.DataFrame, include: Iterable[str], limit: int = 8) -> list[str]:
    names: list[str] = []
    for column in frame.columns:
        label = str(column).lower()
        if any(term in label for term in include) and pd.api.types.is_numeric_dtype(frame[column]):
            names.append(str(column))
    return names[:limit]


def plot_history_file(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> dict[str, float | str]:
    stage = path.stem.replace("history_", "")
    stage_label = nice_stage(stage)
    epoch = first_column(frame, ["epoch", "global_epoch", "step"])
    x = frame[epoch] if epoch else pd.Series(np.arange(1, len(frame) + 1), name="epoch")
    x_label = str(x.name).replace("_", " ").title()
    source = path.name

    loss_columns = numeric_columns(
        frame,
        ["total", "loss", "objective", "score", "validation", "val_"],
        limit=10,
    )
    excluded = {column for column in loss_columns if any(term in column.lower() for term in ["grad", "learning", "lr", "step"])}
    loss_columns = [column for column in loss_columns if column not in excluded]
    if loss_columns:
        figure, axis = plt.subplots(figsize=(10.5, 5.2))
        for index, column in enumerate(loss_columns):
            axis.plot(x, frame[column], label=column.replace("_", " "), linewidth=2, alpha=0.9)
        axis.set_title(f"{stage_label}: objetivos e métricas agregadas")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Valor")
        axis.legend(loc="best", fontsize=8, ncol=2)
        writer.save(figure, f"history_{slug(stage)}_objectives", source, "Curvas agregadas por época")

    term_columns = numeric_columns(
        frame,
        ["rho", "density", "potential", "cohesive", "energy", "jepa", "latent", "reconstruction", "translation", "retention"],
        limit=12,
    )
    if term_columns:
        figure, axis = plt.subplots(figsize=(10.5, 5.2))
        for column in term_columns:
            axis.plot(x, frame[column], label=column.replace("_", " "), linewidth=1.8)
        axis.set_title(f"{stage_label}: termos físicos e de representação")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Valor")
        axis.legend(loc="best", fontsize=7.5, ncol=2)
        writer.save(figure, f"history_{slug(stage)}_terms", source, "Termos de perda e validação disponíveis")

    lr_columns = numeric_columns(frame, ["learning_rate", "learning rate", "lr"], limit=3)
    grad_columns = numeric_columns(frame, ["grad_norm", "gradient_norm", "max_grad"], limit=4)
    if lr_columns or grad_columns:
        figure, left = plt.subplots(figsize=(10.5, 5.2))
        handles, labels = [], []
        for column in lr_columns:
            line = left.plot(x, frame[column], color=PALETTE["blue"], linewidth=2, label=column.replace("_", " "))[0]
            handles.append(line)
            labels.append(line.get_label())
        left.set_xlabel(x_label)
        left.set_ylabel("Learning rate", color=PALETTE["blue"])
        right = left.twinx() if grad_columns else None
        if right is not None:
            for index, column in enumerate(grad_columns):
                color = [PALETTE["orange"], PALETTE["red"], PALETTE["purple"], PALETTE["green"]][index]
                line = right.plot(x, frame[column], color=color, linewidth=1.7, label=column.replace("_", " "))[0]
                handles.append(line)
                labels.append(line.get_label())
            right.set_ylabel("Norma do gradiente")
        left.set_title(f"{stage_label}: learning rate e gradientes")
        left.legend(handles, labels, loc="best", fontsize=8)
        writer.save(figure, f"history_{slug(stage)}_optimizer", source, "Learning rate e normas de gradiente")

    def final_value(names: Iterable[str]) -> float | str:
        column = first_column(frame, names)
        if column and pd.api.types.is_numeric_dtype(frame[column]):
            return float(frame[column].dropna().iloc[-1]) if frame[column].notna().any() else np.nan
        return np.nan

    return {
        "stage": stage,
        "epochs": float(len(frame)),
        "updates": final_value(["optimizer_updates", "global_step", "updates"]),
        "minutes": final_value(["elapsed_minutes", "elapsed_min", "duration_minutes"]),
        "ready": str(frame["ready"].iloc[-1]) if "ready" in frame.columns and len(frame) else "n/a",
    }


def plot_stage_overview(stage_rows: list[dict[str, float | str]], writer: FigureWriter) -> None:
    if not stage_rows:
        return
    summary = pd.DataFrame(stage_rows)
    summary = summary.drop_duplicates(subset="stage", keep="last")
    labels = [nice_stage(str(value)) for value in summary["stage"]]
    updates = pd.to_numeric(summary["updates"], errors="coerce")
    minutes = pd.to_numeric(summary["minutes"], errors="coerce")
    if updates.notna().any() or minutes.notna().any():
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].barh(labels, updates.fillna(0), color=PALETTE["blue"])
        axes[0].set_title("Atualizações do otimizador por estágio")
        axes[0].set_xlabel("Atualizações")
        axes[1].barh(labels, minutes.fillna(0), color=PALETTE["purple"])
        axes[1].set_title("Tempo decorrido por estágio")
        axes[1].set_xlabel("Minutos")
        writer.save(figure, "stages_runtime_and_updates", "history_*.csv", "Esforço computacional por estágio")

    ready = summary["ready"].astype(str).str.lower().isin(["true", "1", "yes"])
    colors = np.where(ready, PALETTE["green"], PALETTE["red"])
    figure, axis = plt.subplots(figsize=(10.5, 4.7))
    axis.barh(labels, np.ones(len(labels)), color=colors)
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.set_title("Readiness final por estágio")
    for index, value in enumerate(summary["ready"]):
        axis.text(0.02, index, str(value), va="center", color="white", weight="bold")
    writer.save(figure, "stages_readiness", "history_*.csv", "Estado final de readiness por estágio")


def plot_g_calibration(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> None:
    if "cohesive" not in path.stem.lower() and "calibration" not in path.stem.lower():
        return
    epoch = first_column(frame, ["epoch", "global_epoch"]) or frame.columns[0]
    x = frame[epoch]
    series = [
        ("MAE (eV/átomo)", ["ni_energy_mae", "energy_mae", "mae"], 0.10, PALETTE["blue"]),
        ("|Bias| (eV/átomo)", ["ni_energy_bias_abs", "energy_bias_abs", "bias_abs"], 0.02, PALETTE["red"]),
        ("P95 (eV/átomo)", ["ni_energy_p95", "energy_p95", "p95"], 0.30, PALETTE["orange"]),
        ("Máximo (eV/átomo)", ["ni_energy_max", "energy_max", "maximum"], 0.60, PALETTE["purple"]),
    ]
    present = [(title, first_column(frame, candidates), limit, color) for title, candidates, limit, color in series]
    present = [item for item in present if item[1] is not None]
    if not present:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    for axis, (title, column, limit, color) in zip(axes.flat, present):
        axis.plot(x, frame[column], marker="o", color=color, linewidth=2)
        axis.axhline(limit, color=PALETTE["red"], linestyle="--", linewidth=1.5, label=f"limite {limit:.3f}")
        axis.set_title(title)
        axis.set_xlabel("Época")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(present) :]:
        axis.axis("off")
    figure.suptitle("Diagnóstico da calibração energética / coesiva", y=1.02, weight="bold")
    writer.save(figure, "g_calibration_energy_gates", path.name, "Métricas de energia e limites de gate")

    parameter_columns = numeric_columns(frame, ["energy_scale", "energy_offset", "affine"], limit=5)
    if parameter_columns:
        figure, axis = plt.subplots(figsize=(10.5, 5.0))
        for column in parameter_columns:
            axis.plot(x, frame[column], marker="o", linewidth=2, label=column.replace("_", " "))
        axis.set_title("Parâmetros aprendidos na calibração energética")
        axis.set_xlabel("Época")
        axis.legend(loc="best")
        writer.save(figure, "g_calibration_affine_parameters", path.name, "Escala e offset de calibração")


def flatten_numeric(data: dict, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            output[name] = float(value)
    return output


def plot_json_metrics(run_dir: Path, writer: FigureWriter, warnings: list[str]) -> None:
    sources = [
        ("source_readiness.json", "source_readiness", "Métricas de readiness das fontes"),
        ("autoencoder_field_gate.json", "autoencoder_field_gates", "Gates de campo do autoencoder"),
        ("d_latent_spectral_alignment.json", "d_latent_spectral_alignment", "Alinhamento espectral do espaço latente"),
    ]
    for filename, output, title in sources:
        path = run_dir / filename
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        values = flatten_numeric(data)
        if not values:
            warnings.append(f"{filename} não contém métricas numéricas simples para plotar.")
            continue
        labels = list(values)
        metric_values = list(values.values())
        figure_height = max(4.5, 0.33 * len(labels) + 1.5)
        figure, axis = plt.subplots(figsize=(11.5, figure_height))
        colors = [PALETTE["blue"] if value >= 0 else PALETTE["red"] for value in metric_values]
        axis.barh(labels, metric_values, color=colors)
        axis.set_title(title)
        axis.set_xlabel("Valor")
        writer.save(figure, output, filename, title)


def plot_physical_validation(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> None:
    source = path.name
    label = slug(path.stem.replace("physical_validation_", "").replace("validation_physical_", ""))
    element = first_column(frame, ["element", "species"])
    target = first_column(frame, ["cohesive_target_ev_per_atom", "energy_target", "target_energy", "energy_reference"])
    prediction = first_column(frame, ["cohesive_pred_ev_per_atom", "energy_prediction", "pred_energy", "energy_pred"])
    if target and prediction:
        subset = frame[[target, prediction] + ([element] if element else [])].dropna()
        if not subset.empty:
            figure, axis = plt.subplots(figsize=(6.8, 6.0))
            if element:
                for group, values in subset.groupby(element):
                    axis.scatter(values[target], values[prediction], s=26, alpha=0.75, label=str(group))
                axis.legend(title="Elemento", fontsize=8)
            else:
                axis.scatter(subset[target], subset[prediction], s=26, alpha=0.75, color=PALETTE["blue"])
            lower = min(subset[target].min(), subset[prediction].min())
            upper = max(subset[target].max(), subset[prediction].max())
            margin = max((upper - lower) * 0.06, 1e-6)
            axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color=PALETTE["red"], label="ideal")
            axis.set_xlabel("Energia alvo (eV/átomo)")
            axis.set_ylabel("Energia prevista (eV/átomo)")
            axis.set_title(f"Validação energética: {label}")
            axis.set_aspect("equal", adjustable="box")
            writer.save(figure, f"physical_{label}_energy_parity", source, "Paridade entre energia prevista e alvo")

    rho = first_column(frame, ["rho_relative_l2", "rho", "density_relative_l2"])
    potential = first_column(frame, ["potential_relative_l2", "potential", "potential_error"])
    available = [("Densidade ρ", rho), ("Potencial", potential)]
    available = [(title, column) for title, column in available if column]
    if available:
        figure, axes = plt.subplots(1, len(available), figsize=(6.2 * len(available), 5.0), squeeze=False)
        for axis, (title, column) in zip(axes.flat, available):
            if element:
                groups = [group[column].dropna().values for _, group in frame.groupby(element)]
                labels = [str(key) for key, _ in frame.groupby(element)]
                axis.boxplot(groups, tick_labels=labels, showfliers=False)
            else:
                axis.hist(frame[column].dropna(), bins=25, color=PALETTE["blue"], alpha=0.8)
            axis.set_title(title)
            axis.set_ylabel("Erro relativo")
        figure.suptitle(f"Validação de campos: {label}", y=1.02, weight="bold")
        writer.save(figure, f"physical_{label}_field_errors", source, "Distribuição dos erros de campo")


def plot_retention(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> None:
    call = first_column(frame, ["call", "stage"])
    metric = first_column(frame, ["metric"])
    excess = first_column(frame, ["excess", "degradation"])
    if not call or not excess:
        return
    plot_data = frame.copy()
    plot_data[excess] = pd.to_numeric(plot_data[excess], errors="coerce")
    grouped = plot_data.groupby(call, dropna=False)[excess].max().sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(10.5, max(4.5, 0.42 * len(grouped) + 1.5)))
    colors = [PALETTE["red"] if value > 0 else PALETTE["green"] for value in grouped]
    axis.barh(grouped.index.astype(str), grouped.values, color=colors)
    axis.axvline(0, color="#334155", linewidth=1)
    axis.set_title("Pior excesso de degradação de retenção por chamada")
    axis.set_xlabel("Excesso acima da tolerância")
    writer.save(figure, "retention_max_excess", path.name, "Pior excesso de retenção por chamada")

    if metric and len(plot_data[metric].dropna().unique()) <= 20:
        pivot = plot_data.pivot_table(index=call, columns=metric, values=excess, aggfunc="max").fillna(0)
        figure, axis = plt.subplots(figsize=(max(9, 0.8 * len(pivot.columns) + 4), max(4.5, 0.42 * len(pivot) + 2)))
        image = axis.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        axis.set_title("Mapa de calor: excesso de retenção")
        figure.colorbar(image, ax=axis, label="Excesso")
        writer.save(figure, "retention_excess_heatmap", path.name, "Excesso máximo por métrica e chamada")


def plot_curriculum(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> None:
    stage = first_column(frame, ["stage"])
    epoch = first_column(frame, ["epoch"])
    if not stage or not epoch:
        return
    metrics = [
        ("active_fraction", "Fração ativa"),
        ("active_records", "Registros ativos"),
        ("difficulty_progress", "Progresso de dificuldade"),
        ("mean_sampling_weight", "Peso médio de amostragem"),
    ]
    metrics = [(column, label) for column, label in metrics if column in frame.columns]
    if not metrics:
        return
    figure, axes = plt.subplots(len(metrics), 1, figsize=(10.5, 3.2 * len(metrics)), sharex=False)
    if len(metrics) == 1:
        axes = [axes]
    for axis, (column, label) in zip(axes, metrics):
        for stage_name, group in frame.groupby(stage):
            group = group.sort_values(epoch)
            axis.plot(group[epoch], group[column], marker="o", markersize=3, linewidth=1.8, label=str(stage_name))
        axis.set_ylabel(label)
        axis.legend(fontsize=7, ncol=2, loc="best")
    axes[-1].set_xlabel("Época")
    figure.suptitle("Evolução do currículo de dados", y=1.01, weight="bold")
    writer.save(figure, "data_curriculum_progress", path.name, "Fração ativa, dificuldade e pesos de amostragem")


def plot_exposure(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> None:
    element = first_column(frame, ["element"])
    draws = first_column(frame, ["draws", "batches_seen"])
    difficulty = first_column(frame, ["physical_difficulty", "difficulty"])
    if element and draws:
        grouped = frame.groupby(element)[draws].sum().sort_values(ascending=False)
        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(grouped.index.astype(str), grouped.values, color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"]][: len(grouped)])
        axis.set_title("Exposição de amostras por elemento")
        axis.set_ylabel("Draws / lotes observados")
        writer.save(figure, "sample_exposure_by_element", path.name, "Amostragem acumulada por elemento")
    if difficulty and draws:
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        color_values = frame[element].astype("category").cat.codes if element else PALETTE["blue"]
        scatter = axis.scatter(frame[difficulty], frame[draws], c=color_values, cmap="viridis", alpha=0.65, s=30)
        axis.set_title("Exposição versus dificuldade física")
        axis.set_xlabel("Dificuldade física")
        axis.set_ylabel("Draws / lotes observados")
        if element:
            figure.colorbar(scatter, ax=axis, label="Código do elemento")
        writer.save(figure, "sample_exposure_vs_difficulty", path.name, "Relação entre dificuldade e exposição")


def plot_gradient_conflicts(path: Path, frame: pd.DataFrame, writer: FigureWriter) -> None:
    cosine = first_column(frame, ["gradient_cosine", "cosine"])
    conflict = first_column(frame, ["conflict"])
    stage = first_column(frame, ["stage"])
    update = first_column(frame, ["optimizer_update", "global_step"])
    if cosine and update:
        figure, axis = plt.subplots(figsize=(10.5, 5.0))
        for stage_name, group in frame.groupby(stage) if stage else [("run", frame)]:
            axis.scatter(group[update], group[cosine], s=12, alpha=0.45, label=str(stage_name))
        axis.axhline(0, color=PALETTE["red"], linestyle="--", linewidth=1.2)
        axis.set_title("Cosseno entre gradientes das tarefas")
        axis.set_xlabel("Atualização do otimizador")
        axis.set_ylabel("Cosseno do gradiente")
        if stage:
            axis.legend(fontsize=7, ncol=2)
        writer.save(figure, "gradient_conflicts_cosine_over_time", path.name, "Alinhamento e conflito entre gradientes")
    if conflict and stage:
        grouped = frame.groupby(stage)[conflict].mean().sort_values(ascending=False)
        figure, axis = plt.subplots(figsize=(9.5, 4.7))
        axis.bar(grouped.index.astype(str), grouped.values, color=PALETTE["orange"])
        axis.set_ylim(0, min(1.0, max(0.05, grouped.max() * 1.15)))
        axis.set_title("Frequência de conflitos de gradiente por estágio")
        axis.set_ylabel("Fração de conflitos")
        axis.tick_params(axis="x", rotation=25)
        writer.save(figure, "gradient_conflicts_frequency", path.name, "Frequência média de conflitos por estágio")


def plot_transition_audit(path: Path, writer: FigureWriter, warnings: list[str]) -> None:
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append(data)
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"Não foi possível ler {path.name}: {error}")
        return
    if not rows:
        return
    frame = pd.json_normalize(rows)
    stage = first_column(frame, ["stage", "from_stage", "event"])
    status = first_column(frame, ["status", "decision", "outcome"])
    if not stage or not status:
        return
    counts = frame.groupby([stage, status]).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(max(9, 1.1 * len(counts)), 5.2))
    bottom = np.zeros(len(counts))
    for index, column in enumerate(counts.columns):
        axis.bar(counts.index.astype(str), counts[column], bottom=bottom, label=str(column))
        bottom += counts[column].to_numpy()
    axis.set_title("Auditoria das transições entre estágios")
    axis.set_ylabel("Eventos")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(fontsize=8)
    writer.save(figure, "stage_transition_audit", path.name, "Eventos e decisões de transição")


def generate(run_dir: Path, output_dir: Path, formats: list[str], dpi: int) -> tuple[int, list[str]]:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    writer = FigureWriter(output_dir, formats, dpi)

    histories: list[dict[str, float | str]] = []
    for path in sorted(run_dir.glob("history_*.csv")):
        frame = read_csv(path, warnings)
        if frame is not None and not frame.empty:
            histories.append(plot_history_file(path, frame, writer))
            plot_g_calibration(path, frame, writer)
    plot_stage_overview(histories, writer)

    plot_json_metrics(run_dir, writer, warnings)
    seen_physical_stages: set[str] = set()
    physical_paths = list(run_dir.glob("physical_validation_*.csv")) + list(run_dir.glob("validation_physical_*.csv"))
    # Some notebook versions write both name variants for the same validation.
    # Prefer the current ``physical_validation_*`` convention and avoid overwriting a plot.
    for path in sorted(physical_paths, key=lambda item: (not item.name.startswith("physical_validation_"), item.name)):
        validation_name = slug(path.stem.replace("physical_validation_", "").replace("validation_physical_", ""))
        if validation_name in seen_physical_stages:
            continue
        seen_physical_stages.add(validation_name)
        frame = read_csv(path, warnings)
        if frame is not None and not frame.empty:
            plot_physical_validation(path, frame, writer)

    optional_csvs = [
        ("retention_metrics_by_component.csv", plot_retention),
        ("data_curriculum_history.csv", plot_curriculum),
        ("sample_stage_exposure.csv", plot_exposure),
        ("gradient_conflicts.csv", plot_gradient_conflicts),
    ]
    for filename, function in optional_csvs:
        path = run_dir / filename
        if path.exists():
            frame = read_csv(path, warnings)
            if frame is not None and not frame.empty:
                function(path, frame, writer)
    audit_path = run_dir / "stage_transition_audit.jsonl"
    if audit_path.exists():
        plot_transition_audit(audit_path, writer, warnings)

    manifest = pd.DataFrame(writer.records)
    if not manifest.empty:
        manifest.to_csv(output_dir / "figures_manifest.csv", index=False)
    summary = [
        "Gerador de gráficos — Periodic Field JEPA",
        f"Run analisado: {run_dir.resolve()}",
        f"Pasta de saída: {output_dir.resolve()}",
        f"Figuras criadas: {len(writer.records)} arquivo(s)",
        f"Formatos: {', '.join(formats)}",
    ]
    if warnings:
        summary.extend(["", "Avisos:", *[f"- {warning}" for warning in warnings]])
    else:
        summary.append("Sem avisos de leitura.")
    (output_dir / "figure_generation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return len(writer.records), warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos de diagnóstico a partir da saída de um run do notebook Periodic Field JEPA."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Diretório de saída do run (padrão: pasta onde este script está salvo)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta para os gráficos (padrão: <run-dir>/figures)",
    )
    parser.add_argument("--formats", default="png", help="Formatos separados por vírgula; exemplo: png,pdf")
    parser.add_argument("--dpi", type=int, default=170, help="Resolução das figuras rasterizadas")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"--run-dir não existe ou não é diretório: {run_dir}")
    formats = [item.strip().lower().lstrip(".") for item in args.formats.split(",") if item.strip()]
    supported = {"png", "pdf", "svg"}
    invalid = sorted(set(formats) - supported)
    if not formats or invalid:
        parser.error(f"Formatos suportados: {', '.join(sorted(supported))}. Inválidos: {invalid}")
    output_dir = (args.out_dir or run_dir / "figures").expanduser()
    count, warnings = generate(run_dir, output_dir, formats, args.dpi)
    print(f"Concluído: {count} arquivo(s) de figura em {output_dir.resolve()}")
    if warnings:
        print(f"Avisos: {len(warnings)} (veja figure_generation_summary.txt)")


if __name__ == "__main__":
    main()
