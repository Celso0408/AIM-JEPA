"""Evaluate the accepted V31 F_periodic_ramp checkpoint on the original OOD split.

Run from the already initialized V31 notebook with:
    %run -i predict_v31_ood_holdout.py

The -i flag is required: this script reuses the notebook's model, split,
normalization, loader, and validation functions. It performs inference only.
"""
from pathlib import Path
from zipfile import ZipFile
import math
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Edit this path if the checkpoint archive/file is elsewhere.
CHECKPOINT_SOURCE = Path("G_joint_finetune.zip")
OUTPUT_DIR = Path("G_joint_finetune_ood_predictions")


def _need(name):
    if name not in globals():
        raise RuntimeError(
            f"Notebook symbol {name!r} is missing. Run the V31 setup/data/model cells first, "
            "then execute this file with %run -i."
        )
    return globals()[name]


def _checkpoint_path(source: Path):
    """Return best.pt from a direct checkpoint path or the supplied ZIP."""
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() != ".zip":
        return source, None
    if not source.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {source}. Set CHECKPOINT_SOURCE near the top of this file."
        )
    temp_dir = tempfile.TemporaryDirectory(prefix="v31_g_joint_finetune_")
    with ZipFile(source) as archive:
        matches = [
            name for name in archive.namelist()
            if name.replace("\\", "/").endswith("G_joint_finetune/best.pt")
        ]
        if not matches:
            matches = [name for name in archive.namelist() if name.endswith("best.pt")]
        if not matches:
            temp_dir.cleanup()
            raise FileNotFoundError("The ZIP does not contain a best.pt checkpoint.")
        member = matches[0]
        extracted = Path(temp_dir.name) / "best.pt"
        with archive.open(member) as src, extracted.open("wb") as dst:
            dst.write(src.read())
    return extracted, temp_dir


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"element", "family", "energy_target", "energy_prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Validation output is missing expected columns: {sorted(missing)}")

    work = frame.copy()
    work["family"] = work["family"].fillna("unknown").astype(str)
    work["energy_error_norm"] = work["energy_prediction"] - work["energy_target"]
    # V31 normalization stores cohesive-energy values as (E - center) / scale.
    norm = _need("pretrain_normalization")
    scale = float(norm.cohesive_scale_ev_per_atom)
    work["energy_target_eV_atom"] = (
        float(norm.cohesive_center_ev_per_atom) + scale * work["energy_target"]
    )
    work["energy_prediction_eV_atom"] = (
        float(norm.cohesive_center_ev_per_atom) + scale * work["energy_prediction"]
    )
    work["energy_error_meV_atom"] = 1000.0 * scale * work["energy_error_norm"]

    groups = ["evaluation_split", "element", "family"]
    rows = []
    for keys, group in work.groupby(groups, dropna=False):
        err = group["energy_error_meV_atom"].to_numpy(dtype=float)
        row = dict(zip(groups, keys))
        row.update({
            "n_configurations": int(len(group)),
            "energy_MAE_meV_atom": float(np.mean(np.abs(err))),
            "energy_RMSE_meV_atom": float(np.sqrt(np.mean(err ** 2))),
            "energy_bias_meV_atom": float(np.mean(err)),
            "energy_abs_error_P95_meV_atom": float(np.quantile(np.abs(err), 0.95)),
            "energy_abs_error_max_meV_atom": float(np.max(np.abs(err))),
        })
        for col in ("rho_cosine", "potential_cosine", "potential_rms_ratio"):
            if col in group:
                row[f"mean_{col}"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _ensure_physical_energy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Converte as energias normalizadas do snapshot para unidades físicas."""
    required = {"energy_target", "energy_prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Validation output is missing normalized energy columns: {sorted(missing)}"
        )

    work = frame.copy()
    normalization = _need("pretrain_normalization")
    scale = float(normalization.cohesive_scale_ev_per_atom)
    center = float(normalization.cohesive_center_ev_per_atom)
    work["energy_target_eV_atom"] = center + scale * work["energy_target"]
    work["energy_prediction_eV_atom"] = center + scale * work["energy_prediction"]
    work["energy_error_meV_atom"] = 1000.0 * scale * (
        work["energy_prediction"] - work["energy_target"]
    )
    return work


def _plot_energy(frame: pd.DataFrame, out_dir: Path):
    # Garante as colunas usadas pelos dois gráficos, inclusive se o snapshot
    # vier somente com as energias normalizadas do V31.
    frame = _ensure_physical_energy_columns(frame)
    splits = list(frame["evaluation_split"].drop_duplicates())
    fig, axes = plt.subplots(1, len(splits), figsize=(7.0 * len(splits), 6.0), squeeze=False)
    colors = {"Al": "#2878B5", "Fe": "#D9534F", "Ni": "#4C9F70"}
    for ax, split_name in zip(axes[0], splits):
        subset = frame[frame["evaluation_split"] == split_name]
        for element, group in subset.groupby("element"):
            color = colors.get(str(element), "#777777")
            ax.scatter(
                group["energy_target_eV_atom"], group["energy_prediction_eV_atom"],
                s=24, alpha=0.62, color=color, edgecolor="white", linewidth=0.25,
                label=f"{element} (n={len(group)})",
            )
        lo = float(min(subset["energy_target_eV_atom"].min(), subset["energy_prediction_eV_atom"].min()))
        hi = float(max(subset["energy_target_eV_atom"].max(), subset["energy_prediction_eV_atom"].max()))
        pad = max((hi - lo) * 0.04, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#333333", lw=1.4, label="Ideal prediction")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(split_name)
        ax.set_xlabel("DFT target cohesive energy (eV/atom)")
        ax.set_ylabel("Model prediction (eV/atom)")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("V31 G_joint_finetune: cohesive-energy predictions", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_parity_by_split.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(splits), figsize=(7.0 * len(splits), 5.2), squeeze=False)
    for ax, split_name in zip(axes[0], splits):
        subset = frame[frame["evaluation_split"] == split_name]
        for element, group in subset.groupby("element"):
            ax.scatter(
                group["energy_target_eV_atom"], group["energy_error_meV_atom"],
                s=24, alpha=0.62, color=colors.get(str(element), "#777777"),
                edgecolor="white", linewidth=0.25, label=str(element),
            )
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_title(split_name)
        ax.set_xlabel("DFT target cohesive energy (eV/atom)")
        ax.set_ylabel("Prediction error (meV/atom)")
        ax.grid(True, alpha=0.22)
        ax.legend(title="Element", frameon=False)
    fig.suptitle("V31 G_joint_finetune: signed energy residuals", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_residuals_by_split.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # These objects are defined by the V31 notebook setup cells.
    splits = _need("splits")
    model_cls = _need("PeriodicFieldJEPA")
    make_loader_fn = _need("make_loader")
    move_batch_fn = _need("move_batch")
    snapshot_fn = _need("physical_validation_snapshot")
    normalization = _need("pretrain_normalization")
    device = _need("DEVICE")

    # Preserve every row from the original manifest splits. Do not use the
    # development-mode/downsampled *_frame objects from the notebook.
    eval_frames = {
        "OOD family holdout (shear/combo)": splits["pretrain_family_ood"].copy(),
        "Al ID holdout": splits["al_holdout_test"].copy(),
        "Fe ID holdout": splits["fe_holdout_test"].copy(),
    }
    for name, data in eval_frames.items():
        if data.empty:
            raise RuntimeError(f"Split {name!r} is empty.")
        if "calc_id" in data and data["calc_id"].duplicated().any():
            raise RuntimeError(f"Split {name!r} contains duplicate calc_id values.")

    ckpt_path, temp_dir = _checkpoint_path(CHECKPOINT_SOURCE)
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with older PyTorch
            payload = torch.load(ckpt_path, map_location="cpu")
        stage = payload.get("stage")
        if stage != "G_joint_finetune":
            raise RuntimeError(f"Expected G_joint_finetune checkpoint; found stage={stage!r}.")
        if "model_state" not in payload:
            raise RuntimeError("Checkpoint has no model_state entry.")

        all_frames = []
        model = None
        for split_name, split_frame in eval_frames.items():
            loader = make_loader_fn(
                split_frame,
                normalization,
                shuffle=False,
                include_targets=True,
                balance_elements=False,
            )
            if model is None:
                first_batch = next(iter(loader))
                sample = move_batch_fn(first_batch)
                model = model_cls(sample).to(device)
                model.load_state_dict(payload["model_state"], strict=True)
                model.eval()
            # Re-create the loader because its first batch was used only to
            # infer model dimensions.
            loader = make_loader_fn(
                split_frame,
                normalization,
                shuffle=False,
                include_targets=True,
                balance_elements=False,
            )
            _, predictions = snapshot_fn(model, loader, autoencoder=False)
            predictions = predictions.copy()
            predictions["evaluation_split"] = split_name
            all_frames.append(predictions)
            print(f"Evaluated {split_name}: {len(predictions):,} configurations")

        predictions = pd.concat(all_frames, ignore_index=True)
        predictions = _ensure_physical_energy_columns(predictions)
        predictions.to_csv(output_dir / "predictions_per_configuration.csv", index=False)
        summary = _metric_summary(predictions)
        summary.to_csv(output_dir / "metrics_by_split_element_family.csv", index=False)
        _plot_energy(predictions, output_dir)

        metadata = {
            "checkpoint": str(ckpt_path),
            "stage": stage,
            "checkpoint_role": payload.get("checkpoint_role"),
            "global_step": payload.get("global_step"),
            "best_validation": payload.get("best_validation"),
            "n_predictions": int(len(predictions)),
            "splits": {name: int(len(frame)) for name, frame in eval_frames.items()},
            "energy_error_sign": "prediction minus DFT target; positive means overprediction",
        }
        pd.Series(metadata, dtype=object).to_json(output_dir / "run_metadata.json", indent=2)
        print("\nMetrics by split, element, and family:")
        print(summary.to_string(index=False))
        print(f"\nFiles saved under: {output_dir}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


main()
