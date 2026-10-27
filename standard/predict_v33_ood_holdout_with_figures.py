"""Evaluate V33 G_joint_finetune on all Al/Fe/Ni ID and family-OOD test records.

In a fresh V33 notebook kernel, execute its setup/data/model cells through the
cell defining PeriodicFieldJEPA (cell 24 in the supplied notebook). Training
cells are not needed. Then run:

    %run -i inference/predict_v33_ood_holdout.py

Change the path above if the script is placed elsewhere. The checkpoint ZIP
and the output directory below are resolved relative to the notebook's current
working directory, not the location of this script.
"""
from pathlib import Path
from zipfile import ZipFile
from dataclasses import asdict, replace
import json
import os
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CHECKPOINT_SOURCE = Path("G_joint_finetune.zip")
OUTPUT_DIR = Path("G_joint_finetune_predictions")
EXPECTED_STAGE = "G_joint_finetune"
EXPORT_DENSITY_MATRICES = True  # Predicted and reference 3D density per configuration.
ALLOW_EXPLORATORY_CHECKPOINT = True
FIGURE_DPI = 600
# Set known calc_ids to reproduce the exact configurations in an existing paper figure.
# None selects the first ID-holdout configuration encountered for that element.
FIGURE_EXAMPLE_IDS = {"Al": None, "Fe": None, "Ni": None}
FIGURE_SLICE_Z = None  # None selects the central plane of the canonical (z, y, x) grid.


def _need(name):
    if name not in globals():
        raise RuntimeError(
            f"Notebook symbol {name!r} is missing. Start a fresh kernel and "
            "execute the V33 setup/data/model cells through PeriodicFieldJEPA "
            "(cell 24), then run this script with %run -i."
        )
    return globals()[name]


def _checkpoint_path(source):
    """Find this stage's best.pt without extracting the archive as a tree."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    if source.suffix.lower() != ".zip":
        return source, None, None
    temp_dir = tempfile.TemporaryDirectory(prefix="v33_g_joint_finetune_")
    try:
        with ZipFile(source) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            matches = [name for name in names if name.replace("\\", "/").endswith(
                EXPECTED_STAGE + "/last.pt")]
            if not matches:
                matches = [name for name in names if name.replace("\\", "/") == "last.pt"]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {EXPECTED_STAGE}/last.pt in {source}; "
                    f"found {len(matches)}. Choose a direct .pt file if the archive is ambiguous."
                )
            member = matches[0]
            extracted = Path(temp_dir.name) / "last.pt"
            with archive.open(member) as src, extracted.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
        return extracted, temp_dir, member
    except BaseException:
        temp_dir.cleanup()
        raise


def _validate_checkpoint(payload, normalization):
    stage = payload.get("stage")
    if stage != EXPECTED_STAGE:
        raise RuntimeError(f"Expected {EXPECTED_STAGE} checkpoint; found stage={stage!r}")
    if not isinstance(payload.get("model_state"), dict):
        raise RuntimeError("Checkpoint does not contain model_state")
    
    # Permitir inferência exploratória com o checkpoint original.
    # Nenhum peso ou metadado do checkpoint é modificado.

    ALLOW_EXPLORATORY_CHECKPOINT = True

    epoch = int(payload.get("epoch", -1))
    role = payload.get("checkpoint_role")

    is_validation_best = (
        epoch >= 0
        and role == "validation_best"
    )

    if not is_validation_best:

        if not ALLOW_EXPLORATORY_CHECKPOINT:
            raise RuntimeError(
                "Checkpoint is not a trained validation-best candidate. "
                f"epoch={epoch}, role={role!r}"
            )

        print("\n" + "=" * 65)
        print("WARNING: EXPLORATORY CHECKPOINT EVALUATION")
        print("=" * 65)
        print(f"Checkpoint epoch: {epoch}")
        print(f"Checkpoint role: {role}")
        print(f"Checkpoint stage: {payload.get('stage')}")
        print(f"Global step: {payload.get('global_step')}")
        print("Original model weights will be used without modification.")
        print("This checkpoint is NOT verified as validation-best.")
        print("=" * 65 + "\n")

    
    version = _need("ARCHITECTURE_VERSION")
    if payload.get("architecture_version") != version:
        raise RuntimeError(
            "Checkpoint architecture does not match the V33 definitions in this kernel: "
            f"checkpoint={payload.get('architecture_version')!r}, notebook={version!r}. "
            "Restart the kernel and execute the V33 setup/model cells."
        )
    actual_hash = _need("stable_contract_hash")(_need("asdict")(normalization))
    if payload.get("normalization_hash") != actual_hash:
        raise RuntimeError("Checkpoint normalization differs from the V33 notebook's normalization")
    split_hash = _need("stable_contract_hash")(_need("split_manifest"))
    if payload.get("split_manifest_hash") != split_hash:
        raise RuntimeError("Checkpoint and notebook have different data splits; holdout evaluation blocked")
    _validate_ni_selection(payload)
    state = payload["model_state"]
    rms = state.get("field_online_encoder.spectral_rms")
    if rms is None or not torch.isfinite(rms).all() or not (rms > 0).all():
        raise RuntimeError("Checkpoint lacks a valid V33 spectral RMS buffer")
    # V33 constructs all spectral branches from this global. The checkpoint
    # is authoritative; all registered buffers are subsequently loaded strictly.
    globals()["SPECTRAL_RMS_STATISTICS"] = {"rms": rms.detach().cpu().tolist()}


def _validate_ni_selection(payload):
    """Accept metadata drift only if the checkpoint's actual Ni pools are identical."""
    stable_hash = _need("stable_contract_hash")
    current = _need("NI_SELECTION_MANIFEST")
    training_contract = payload.get("training_contract")
    saved = training_contract.get("ni_selection") if isinstance(training_contract, dict) else None
    checkpoint_hash = payload.get("ni_manifest_hash")
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise RuntimeError("Ni selection manifesto completo ausente no checkpoint/notebook")
    for label, manifest in (("checkpoint", saved), ("notebook", current)):
        expected = stable_hash({key: value for key, value in manifest.items() if key != "hash"})
        if manifest.get("hash") != expected:
            raise RuntimeError(f"Ni selection manifesto {label} tem hash interno inválido")
    if saved["hash"] != checkpoint_hash:
        raise RuntimeError("Ni selection no contrato do checkpoint não corresponde ao hash salvo")

    # The split-manifest hash was verified immediately above. In V33 its
    # records include ni_adapt_train, but the eligible Ni pool is stored only
    # in this full checkpoint training contract. Check both explicitly.
    for field in ("eligible_ids", "selected_ids"):
        old = list(map(str, saved.get(field, [])))
        new = list(map(str, current.get(field, [])))
        if not old or len(old) != len(set(old)) or len(new) != len(set(new)):
            raise RuntimeError(f"Ni {field}: IDs ausentes ou duplicados")
        if set(old) != set(new):
            raise RuntimeError(
                f"Ni {field} differs between checkpoint and notebook; "
                "the original training selection must be restored before holdout evaluation"
            )
    selected = set(map(str, saved["selected_ids"]))
    actual_training_ids = set(map(str, _need("splits")["ni_adapt_train"].calc_id))
    if selected != actual_training_ids:
        raise RuntimeError("Checkpoint Ni selected IDs do not match notebook ni_adapt_train")
    for field in ("version", "seed", "fraction", "frozen_split_hash"):
        if saved.get(field) is None or saved[field] != current.get(field):
            raise RuntimeError(f"Ni selection contract differs in {field!r}")
    if checkpoint_hash != current["hash"]:
        print("WARNING: Ni manifesto hash differs in metadata; verified identical "
              "eligible/training IDs, frozen split, selection settings, and split-manifest hash. "
              "Checkpoint weights and metadata remain unchanged.")


def _operator_input(batch):
    operator = batch.clone()
    for key in _need("TARGET_KEYS"):
        if key in operator:
            del operator[key]
    return operator


def _fit_classification_thresholds(splits, norm):
    """Recreate V33 density and energy cutoffs from training configurations only."""
    cfg = _need("CFG")
    sample_fn = _need("deterministic_sample")
    raw_record = _need("cached_raw_record")
    train = globals().get("pretrain_train_frame")
    if train is None:
        base = splits["pretrain_train"]
        if cfg.run_mode in ("SMOKE", "QUICK"):
            mode_frame = _need("frame_for_mode")
            train = pd.concat([mode_frame(group, 16) for _, group in
                               base.groupby("element")], ignore_index=True)
        elif cfg.run_mode == "QUICK_TRAIN":
            train = pd.concat([
                sample_fn(group, min(cfg.quick_train_records_per_element, len(group)),
                          cfg.seed + 450 + i)
                for i, (_, group) in enumerate(base.groupby("element"))
            ], ignore_index=True)
        else:
            train = base
    density, weak, upper, sampled_ids = {}, {}, {}, {}
    for offset, element in enumerate(("Al", "Fe", "Ni")):
        if element == "Ni":
            # V33 v29_fit_thresholds uses this exact training-only Ni pool.
            group = _need("frame_for_mode")(splits["ni_adapt_train"], 32, 1200)
            energy_group = splits["ni_adapt_train"]
        else:
            group = train.loc[train.element.eq(element)]
            energy_group = group
        if group.empty:
            raise RuntimeError(f"No {element} training configurations for classification thresholds")
        sample = sample_fn(group, min(cfg.classification_threshold_records_per_element,
                                      len(group)), cfg.seed + 710 + offset)
        fields = [np.asarray(raw_record(row.to_dict())["rho_scaled"],
                             dtype=np.float32).reshape(-1) for _, row in sample.iterrows()]
        density[element] = float(np.quantile(
            np.concatenate(fields), cfg.density_classification_quantile))
        coords = ((energy_group.cohesive_energy_ev_per_atom.to_numpy(dtype=float) -
                   float(norm.cohesive_center_ev_per_atom)) /
                  float(norm.cohesive_scale_ev_per_atom))
        weak[element] = float(np.quantile(coords, cfg.cohesive_weak_quantile))
        upper[element] = float(np.quantile(coords, .90))
        sampled_ids[element] = sample.calc_id.astype(str).tolist()
    return {"density_high": density, "weak_cohesion": weak,
            "energy_high_tail": upper, "density_sampled_training_ids": sampled_ids,
            "source": "V33 training configurations only", "potential_positive": 0.0}


def _binary_counts(truth, predicted):
    truth = truth.bool().reshape(-1)
    predicted = predicted.bool().reshape(-1)
    return {"tn": int((~truth & ~predicted).sum()),
            "fp": int((~truth & predicted).sum()),
            "fn": int((truth & ~predicted).sum()),
            "tp": int((truth & predicted).sum())}


@torch.inference_mode()
def _evaluate_split(model, loader, split_name, split_slug, test_type, device, norm,
                    thresholds, out_dir, field_examples=None):
    """Match V33 physical_validation_snapshot's field/energy calculations."""
    rows = []
    density_index = []
    classification_rows = []
    field_dir = out_dir / "density_matrices" / split_slug
    if EXPORT_DENSITY_MATRICES:
        field_dir.mkdir(parents=True, exist_ok=True)
    seen_filenames = set()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model.forward_operator(_operator_input(batch))
        rho_pred = out["rho_scaled"].float()
        rho_true = batch.rho_target.float()
        potential_pred = out["potential_scaled"].float()
        potential_true = batch.potential_target.float()
        potential_pred = potential_pred - potential_pred.mean(dim=(-3, -2, -1), keepdim=True)
        potential_true = potential_true - potential_true.mean(dim=(-3, -2, -1), keepdim=True)

        def rel_l2(pred, true):
            error = (pred - true).square().flatten(1).mean(-1)
            reference = true.square().flatten(1).mean(-1).clamp_min(1e-8)
            return torch.sqrt(error / reference + 1e-12)

        rho_l2 = rel_l2(rho_pred, rho_true)
        potential_l2 = rel_l2(potential_pred, potential_true)
        rho_cos = F.cosine_similarity(rho_pred.flatten(1), rho_true.flatten(1), dim=-1, eps=1e-6)
        potential_cos = F.cosine_similarity(
            potential_pred.flatten(1), potential_true.flatten(1), dim=-1, eps=1e-6)
        pred_rms = potential_pred.square().flatten(1).mean(-1).clamp_min(1e-12).sqrt()
        true_rms = potential_true.square().flatten(1).mean(-1).clamp_min(1e-12).sqrt()
        ratio = pred_rms / true_rms
        prediction = out["cohesive_energy_normalized"].flatten().float()
        target = batch.cohesive_energy_target.flatten().float()
        for j in range(batch.num_graphs):
            calc_id = str(batch.calc_id[j])
            element = str(batch.element[j])
            family = str(batch.family[j])
            # Store only one ID example per element, not every 3D field in RAM.
            # Arrays have canonical (z, y, x) order; show the central z plane.
            chosen_id = FIGURE_EXAMPLE_IDS.get(element)
            if (field_examples is not None and test_type == "ID"
                    and element not in field_examples
                    and (chosen_id is None or str(chosen_id) == calc_id)):
                z = int(rho_true.shape[-3] // 2 if FIGURE_SLICE_Z is None else FIGURE_SLICE_Z)
                if z < 0 or z >= rho_true.shape[-3]:
                    raise ValueError(f"FIGURE_SLICE_Z={z} is outside the field grid")
                field_examples[element] = {
                    "calc_id": calc_id, "evaluation_split": split_name, "z_index": z,
                    "rho_reference": rho_true[j, 0, z].cpu().numpy().copy(),
                    "rho_prediction": rho_pred[j, 0, z].cpu().numpy().copy(),
                    "potential_reference": potential_true[j, 0, z].cpu().numpy().copy(),
                    "potential_prediction": potential_pred[j, 0, z].cpu().numpy().copy(),
                }
            rows.append({
                "evaluation_split": split_name, "test_type": test_type, "calc_id": calc_id,
                "element": element, "family": family,
                "rho_relative_l2": float(rho_l2[j]),
                "potential_relative_l2": float(potential_l2[j]),
                "rho_cosine": float(rho_cos[j]), "potential_cosine": float(potential_cos[j]),
                "potential_rms_ratio": float(ratio[j]),
                "energy_target": float(target[j]), "energy_prediction": float(prediction[j]),
            })
            tasks = {
                "density_high": (rho_true[j] > thresholds["density_high"][element],
                                 rho_pred[j] > thresholds["density_high"][element],
                                 thresholds["density_high"][element], "voxels"),
                "potential_positive": (potential_true[j] > 0, potential_pred[j] > 0,
                                       0.0, "voxels"),
                "weak_cohesion": (target[j:j+1] <= thresholds["weak_cohesion"][element],
                                  prediction[j:j+1] <= thresholds["weak_cohesion"][element],
                                  thresholds["weak_cohesion"][element], "configurations"),
                "energy_high_tail": (
                    target[j:j+1] >= thresholds["energy_high_tail"][element],
                    prediction[j:j+1] >= thresholds["energy_high_tail"][element],
                    thresholds["energy_high_tail"][element], "configurations"),
            }
            for task, (truth, predicted, threshold, unit) in tasks.items():
                classification_rows.append({
                    "evaluation_split": split_name, "test_type": test_type, "calc_id": calc_id,
                    "element": element, "family": family, "task": task,
                    "count_unit": unit, "threshold_model_units": threshold,
                    **_binary_counts(truth, predicted),
                })
            if EXPORT_DENSITY_MATRICES:
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", calc_id).strip("._")
                if not safe_id or safe_id in seen_filenames:
                    raise RuntimeError(f"Invalid or colliding calc_id in {split_name}: {calc_id!r}")
                seen_filenames.add(safe_id)
                relative_path = Path("density_matrices") / split_slug / f"{safe_id}.npz"
                final_path = out_dir / relative_path
                temporary_path = final_path.with_suffix(".npz.partial")

                # V33 works on the canonical [48, 32, 32] grid. Density is
                # mean-one inside the model; recover e/Bohr^3 from charge/volume.
                density_scale = float(batch.electron_count[j] / batch.volume_bohr3[j])
                arrays = {
                    "density_prediction_e_bohr3":
                        (rho_pred[j, 0] * density_scale).cpu().numpy().astype(np.float32),
                    "density_reference_e_bohr3":
                        (rho_true[j, 0] * density_scale).cpu().numpy().astype(np.float32),
                    "cell_angstrom": batch.cell[j].cpu().numpy().astype(np.float32),
                    "fractional_atomic_positions": batch.pos_fractional[
                        batch.batch == j].cpu().numpy().astype(np.float32),
                    "native_grid_shape": batch.native_grid_shape[j].cpu().numpy().astype(np.int32),
                    "electron_count": np.asarray(float(batch.electron_count[j])),
                    "volume_bohr3": np.asarray(float(batch.volume_bohr3[j])),
                }
                with temporary_path.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                os.replace(temporary_path, final_path)
                density_index.append({
                    "evaluation_split": split_name, "test_type": test_type, "calc_id": calc_id,
                    "element": element, "family": family,
                    "file": relative_path.as_posix(),
                    "canonical_grid_zyx": "x".join(map(str, rho_pred.shape[-3:])),
                    "density_unit": "e/Bohr^3",
                })
    result = pd.DataFrame(rows)
    if result.empty or result.calc_id.duplicated().any():
        raise RuntimeError(f"Empty or duplicate configuration results for {split_name}")
    if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError(f"Nonfinite predictions on {split_name}")
    return result, density_index, classification_rows


def _classification_tables(rows):
    per_configuration = pd.DataFrame(rows)
    columns = ["tn", "fp", "fn", "tp"]
    tables = []
    for level, group_columns in (
        ("split", ["evaluation_split", "task", "count_unit"]),
        ("split_element", ["evaluation_split", "element", "task", "count_unit"]),
        ("split_element_family", ["evaluation_split", "element", "family", "task", "count_unit"]),
    ):
        table = per_configuration.groupby(group_columns, dropna=False).agg(
            configurations=("calc_id", "size"), **{col: (col, "sum") for col in columns}
        ).reset_index()
        table.insert(0, "aggregation_level", level)
        tables.append(table)
    counts = pd.concat(tables, ignore_index=True)
    tn, fp, fn, tp = (counts[col].to_numpy(dtype=float) for col in columns)
    safe = lambda numer, denom: np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
    metrics = counts.copy()
    metrics["precision"] = safe(tp, tp + fp)
    metrics["recall"] = safe(tp, tp + fn)
    metrics["specificity"] = safe(tn, tn + fp)
    metrics["f1"] = safe(2 * tp, 2 * tp + fp + fn)
    metrics["balanced_accuracy"] = .5 * (metrics.recall + metrics.specificity)
    metrics["accuracy"] = safe(tn + tp, tn + fp + fn + tp)
    metrics["support_positive"] = (tp + fn).astype(np.int64)
    metrics["support_negative"] = (tn + fp).astype(np.int64)
    return per_configuration, counts, metrics


def _plot_confusion_matrices(counts, out_dir):
    plot_dir = out_dir / "confusion_matrices"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for row in counts.loc[counts.aggregation_level.eq("split")].itertuples():
        values = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=np.int64)
        totals = values.sum(axis=1, keepdims=True)
        ratios = np.divide(values, totals, out=np.zeros_like(values, dtype=float), where=totals != 0)
        fig, ax = plt.subplots(figsize=(5.6, 4.7))
        ax.imshow(ratios, cmap="Blues", vmin=0, vmax=1)
        for (i, j), n in np.ndenumerate(values):
            ax.text(j, i, f"{n:,}\n{ratios[i, j]:.1%}", ha="center", va="center",
                    color="white" if ratios[i, j] > .55 else "#15232d", fontsize=11)
        ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=("Negative", "Positive"),
               yticklabels=("Negative", "Positive"), xlabel="Predicted class",
               ylabel="Reference class", title=f"{row.evaluation_split}\n{row.task} ({row.count_unit})")
        fig.tight_layout()
        slug = re.sub(r"[^a-z0-9]+", "_", row.evaluation_split.lower()).strip("_")
        fig.savefig(plot_dir / f"{slug}_{row.task}.png", dpi=200)
        plt.close(fig)


def _ensure_physical_energy_columns(frame, norm):
    work = frame.copy()
    scale = float(norm.cohesive_scale_ev_per_atom)
    center = float(norm.cohesive_center_ev_per_atom)
    work["energy_target_eV_atom"] = center + scale * work["energy_target"]
    work["energy_prediction_eV_atom"] = center + scale * work["energy_prediction"]
    work["energy_error_meV_atom"] = 1000.0 * scale * (work["energy_prediction"] - work["energy_target"])
    return work


def _metric_summary(frame):
    groups = ["evaluation_split", "test_type", "element", "family"]
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        err = group["energy_error_meV_atom"].to_numpy(dtype=float)
        row = dict(zip(groups, keys))
        row.update({
            "n_configurations": int(len(group)),
            "energy_MAE_meV_atom": float(np.mean(np.abs(err))),
            "energy_RMSE_meV_atom": float(np.sqrt(np.mean(err**2))),
            "energy_bias_meV_atom": float(np.mean(err)),
            "energy_abs_error_P95_meV_atom": float(np.quantile(np.abs(err), 0.95)),
            "energy_abs_error_max_meV_atom": float(np.max(np.abs(err))),
            "mean_rho_relative_l2": float(group.rho_relative_l2.mean()),
            "mean_potential_relative_l2": float(group.potential_relative_l2.mean()),
            "mean_rho_cosine": float(group.rho_cosine.mean()),
            "mean_potential_cosine": float(group.potential_cosine.mean()),
            "mean_potential_rms_ratio": float(group.potential_rms_ratio.mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _plot_energy(frame, out_dir):
    # One row for ID, one for OOD, and one column per chemical element. Colour
    # carries the deformation family within each independent test split.
    elements = ("Al", "Fe", "Ni")
    split_kinds = ("ID", "OOD")
    families = sorted(frame.family.astype(str).unique())
    cmap = plt.get_cmap("tab10")
    colors = {family: cmap(i % 10) for i, family in enumerate(families)}
    for kind, filename in (("parity", "energy_parity_by_split.png"),
                           ("residual", "energy_residuals_by_split.png")):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
        for row, split_kind in enumerate(split_kinds):
            for col, element in enumerate(elements):
                ax = axes[row, col]
                subset = frame.loc[(frame.element == element) &
                                   (frame.test_type == split_kind)]
                if subset.empty:
                    raise RuntimeError(f"No {element} {split_kind} predictions for plotting")
                for family, group in subset.groupby("family", sort=True):
                    ax.scatter(group.energy_target_eV_atom,
                               group.energy_prediction_eV_atom if kind == "parity"
                               else group.energy_error_meV_atom,
                               s=19, alpha=.65, color=colors[str(family)],
                               edgecolor="white", linewidth=.25,
                               label=f"{family} (n={len(group)})")
                if kind == "parity":
                    lo = float(min(subset.energy_target_eV_atom.min(),
                                   subset.energy_prediction_eV_atom.min()))
                    hi = float(max(subset.energy_target_eV_atom.max(),
                                   subset.energy_prediction_eV_atom.max()))
                    pad = max((hi - lo) * .04, 1e-6)
                    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#333333", lw=1.2)
                    ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
                           ylabel="Predicted cohesive energy (eV/atom)")
                else:
                    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
                    ax.set(ylabel="Prediction error (meV/atom)")
                mae = subset.energy_error_meV_atom.abs().mean()
                ax.set(title=f"{element} {split_kind} | n={len(subset)}, MAE={mae:.1f} meV/atom",
                       xlabel="DFT cohesive energy (eV/atom)")
                ax.grid(True, alpha=.22)
                ax.legend(title="Family", frameon=False, fontsize=8)
        fig.suptitle(f"V33 {EXPECTED_STAGE}: cohesive-energy {kind} on all test families",
                     fontsize=15)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def _plot_field_examples(field_examples, norm, out_dir):
    """Paper-style three-column maps: reference, prediction and signed error."""
    elements = ("Al", "Fe", "Ni")
    missing = set(elements) - set(field_examples)
    if missing:
        raise RuntimeError(f"Missing ID field examples for: {sorted(missing)}; "
                           "check FIGURE_EXAMPLE_IDS against ID holdout calc_ids")
    potential_scale = float(norm.potential_scale_ry)
    for field, filename, labels, factor in (
        ("rho", "density_maps.jpg", ("Normalized density", "Predicted", "Error"), 1.0),
        ("potential", "potential_maps.jpg",
         ("Potential (Ry)", "Potential (Ry) — Predicted", "Potential (Ry) — Error"),
         potential_scale),
    ):
        with plt.rc_context({"font.family": "DejaVu Sans", "font.weight": "bold",
                             "axes.titleweight": "bold", "axes.titlesize": 10}):
            fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.7), squeeze=False)
            fig.subplots_adjust(left=.06, right=.91, top=.925, bottom=.05,
                                wspace=.58, hspace=.45)
            for row, element in enumerate(elements):
                sample = field_examples[element]
                reference = sample[f"{field}_reference"] * factor
                predicted = sample[f"{field}_prediction"] * factor
                error = predicted - reference
                lo = float(min(reference.min(), predicted.min()))
                hi = float(max(reference.max(), predicted.max()))
                if hi <= lo:
                    hi = lo + 1e-12
                error_limit = max(float(np.max(np.abs(error))), 1e-12)
                for col, (values, color_map, limits) in enumerate((
                    (reference, "viridis", (lo, hi)),
                    (predicted, "viridis", (lo, hi)),
                    (error, "coolwarm", (-error_limit, error_limit)),
                )):
                    ax = axes[row, col]
                    image = ax.imshow(values, origin="lower", cmap=color_map,
                                      vmin=limits[0], vmax=limits[1],
                                      interpolation="nearest")
                    ax.set_title(labels[col], pad=10, fontsize=10)
                    ax.set_xticks(np.arange(0, values.shape[1], 5))
                    ax.set_yticks(np.arange(0, values.shape[0], 5))
                    ax.tick_params(labelsize=10, length=3, pad=1)
                    cbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.05, shrink=.72)
                    cbar.ax.tick_params(labelsize=9)
                    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels(),
                                 *cbar.ax.get_yticklabels()):
                        tick.set_fontweight("bold")
                fig.text(.045, axes[row, 0].get_position().y1 + .034,
                         f"({chr(ord('a') + row)}) {element}",
                         ha="left", va="bottom", fontsize=17, fontweight="bold")
            fig.savefig(out_dir / filename, dpi=FIGURE_DPI,
                        facecolor="white", pil_kwargs={"quality": 95})
            plt.close(fig)


def _plot_energy_reference_layout(frame, out_dir):
    """Additional parity layout matching the supplied 3-element × 2-split example."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.weight": "bold",
                         "axes.titleweight": "bold", "axes.labelweight": "bold"}):
        fig, axes = plt.subplots(3, 2, figsize=(8.0, 11.0), squeeze=False)
        fig.subplots_adjust(left=.10, right=.975, top=.965, bottom=.06,
                            wspace=.52, hspace=.54)
        for row, element in enumerate(("Al", "Fe", "Ni")):
            for col, split_kind in enumerate(("ID", "OOD")):
                subset = frame.loc[(frame.element == element) & (frame.test_type == split_kind)]
                if subset.empty:
                    raise RuntimeError(f"No {element} {split_kind} predictions for plotting")
                x = subset.energy_target_eV_atom.to_numpy(dtype=float)
                y = subset.energy_prediction_eV_atom.to_numpy(dtype=float)
                ax = axes[row, col]
                ax.scatter(x, y, s=19, alpha=.95, color="#0965b6",
                           edgecolors="#17304f", linewidths=.3)
                lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
                pad = max((hi - lo) * .04, 1e-6)
                ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1.2, dashes=(5, 3))
                ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
                       title=f"{element}-{split_kind}",
                       xlabel=r"$E_{\mathrm{coh}}$ (eV/atom)",
                       ylabel=r"Predicted $E_{\mathrm{coh}}$ (eV/atom)")
                ax.title.set_fontsize(14)
                ax.xaxis.label.set_size(11)
                ax.yaxis.label.set_size(11)
                ax.tick_params(axis="both", labelsize=10, width=1.2, length=4)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("bottom", "left"):
                    ax.spines[side].set_linewidth(1.2)
                for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                    tick.set_fontweight("bold")
                mae = float(np.mean(np.abs(y - x)))
                ss_total = float(np.sum((x - x.mean()) ** 2))
                r2 = 1 - float(np.sum((y - x) ** 2)) / ss_total if ss_total > 0 else float("nan")
                r2_label = f"{r2:.4f}" if np.isfinite(r2) else "undefined"
                ax.text(.04, .96, f"MAE = {mae:.4f} eV/atom\n$R^2$ = {r2_label}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        fontweight="bold",
                        bbox={"facecolor": "white", "edgecolor": "black",
                              "boxstyle": "round,pad=0.35", "linewidth": .7})
        fig.savefig(out_dir / "cohesive_energy.jpg", dpi=FIGURE_DPI,
                    facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)


def main():
    splits = _need("splits")
    model_cls = _need("PeriodicFieldJEPA")
    make_loader = _need("make_loader")
    norm = _need("pretrain_normalization")
    device = _need("DEVICE")

    alfe_ood = splits["pretrain_family_ood"]
    eval_frames = {
        "Al ID holdout": ("al_id_holdout", "ID", splits["al_holdout_test"].copy()),
        "Fe ID holdout": ("fe_id_holdout", "ID", splits["fe_holdout_test"].copy()),
        "Ni ID holdout": ("ni_id_holdout", "ID", splits["ni_id_holdout_test"].copy()),
        "Al OOD family holdout": ("al_ood_family_holdout", "OOD",
                                   alfe_ood.loc[alfe_ood.element.eq("Al")].copy()),
        "Fe OOD family holdout": ("fe_ood_family_holdout", "OOD",
                                   alfe_ood.loc[alfe_ood.element.eq("Fe")].copy()),
        "Ni OOD family holdout": ("ni_ood_family_holdout", "OOD",
                                   splits["ni_family_ood_test"].copy()),
    }
    ids_seen = set()
    for name, (_, split_kind, frame) in eval_frames.items():
        if frame.empty or frame.calc_id.astype(str).duplicated().any():
            raise RuntimeError(f"Empty or duplicate configuration IDs in {name}")
        ids = set(frame.calc_id.astype(str))
        if ids_seen.intersection(ids):
            raise RuntimeError(f"Test configurations occur in more than one split: {name}")
        ids_seen.update(ids)
        # Ni OOD is group-disjoint; a structural group may contain other
        # family labels, all of which must remain in that frozen test split.
        if (split_kind == "OOD" and name.startswith(("Al ", "Fe ")) and
                not set(frame.family).issubset(set(_need("CFG").ood_families))):
            raise RuntimeError(f"Non-OOD family in {name}")

    ckpt_path, temp_dir, member = _checkpoint_path(CHECKPOINT_SOURCE)
    try:
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch without weights_only
            payload = torch.load(ckpt_path, map_location="cpu")
        saved = payload.get("normalization") or {}
        current = asdict(norm)
        print("Modo atual:", _need("CFG").run_mode)
        print("Hash do checkpoint:", payload.get("normalization_hash"))
        print("Hash calculado agora:", _need("stable_contract_hash")(current))
        for key in sorted(set(saved) | set(current)):
            if saved.get(key) != current.get(key):
                old, new = saved.get(key), current.get(key)
                if isinstance(old, list) and isinstance(new, list):
                    print(f"{key}: checkpoint={len(old)} IDs; atual={len(new)} IDs")
                else:
                    print(f"{key}: checkpoint={old!r}; atual={new!r}")
        if (_need("stable_contract_hash")(current) != payload.get("normalization_hash")
                and set(saved) == set(current)
                and all(saved[key] == current[key] for key in current
                        if key != "potential_scale_ry")
                and _need("stable_contract_hash")(saved) == payload.get("normalization_hash")
                and np.isclose(saved["potential_scale_ry"], current["potential_scale_ry"],
                               rtol=2e-9, atol=0)):
            norm = replace(norm, potential_scale_ry=float(saved["potential_scale_ry"]))
        _validate_checkpoint(payload, norm)
        _, _, first_frame = next(iter(eval_frames.values()))
        first_loader = make_loader(first_frame, norm, shuffle=False,
                                   include_targets=True, balance_elements=False)
        sample = next(iter(first_loader)).to(device, non_blocking=True)
        model = model_cls(sample).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        thresholds = _fit_classification_thresholds(splits, norm)

        all_frames = []
        density_index = []
        classification_rows = []
        field_examples = {}
        output_dir = OUTPUT_DIR.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, (split_slug, split_kind, split_frame) in eval_frames.items():
            loader = make_loader(split_frame, norm, shuffle=False,
                                 include_targets=True, balance_elements=False)
            result, split_density_index, split_classification = _evaluate_split(
                model, loader, split_name, split_slug, split_kind,
                device, norm, thresholds, output_dir, field_examples=field_examples)
            if len(result) != len(split_frame):
                raise RuntimeError(f"Incomplete split {split_name}: {len(result)} of {len(split_frame)}")
            all_frames.append(result)
            density_index.extend(split_density_index)
            classification_rows.extend(split_classification)
            print(f"Evaluated {split_name}: {len(result):,} configurations")
        predictions = _ensure_physical_energy_columns(pd.concat(all_frames, ignore_index=True), norm)
        if EXPORT_DENSITY_MATRICES and len(density_index) != len(predictions):
            raise RuntimeError("Density matrix exports do not cover all evaluated configurations")
        predictions.to_csv(output_dir / "predictions_per_configuration.csv", index=False)
        if EXPORT_DENSITY_MATRICES:
            pd.DataFrame(density_index).to_csv(output_dir / "density_matrices_manifest.csv", index=False)
        summary = _metric_summary(predictions)
        summary.to_csv(output_dir / "metrics_by_split_element_family.csv", index=False)
        _plot_energy(predictions, output_dir)
        _plot_energy_reference_layout(predictions, output_dir)
        _plot_field_examples(field_examples, norm, output_dir)
        per_configuration, counts, classification_metrics = _classification_tables(classification_rows)
        if len(per_configuration) != 4 * len(predictions):
            raise RuntimeError("Classification count does not match the four V33 tasks")
        per_configuration.to_csv(output_dir / "classification_per_configuration.csv", index=False)
        counts.to_csv(output_dir / "confusion_matrices_counts.csv", index=False)
        classification_metrics.to_csv(output_dir / "classification_metrics.csv", index=False)
        _plot_confusion_matrices(counts, output_dir)
        with (output_dir / "classification_thresholds_train_only.json").open("w", encoding="utf-8") as handle:
            json.dump(thresholds, handle, indent=2)
        metadata = {
            "checkpoint_source": str(CHECKPOINT_SOURCE.expanduser().resolve()),
            "checkpoint_member": member, "stage": payload["stage"],
            "checkpoint_role": payload.get("checkpoint_role"),
            "epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
            "best_validation": payload.get("best_validation"),
            "architecture_version": payload.get("architecture_version"),
            "n_predictions": int(len(predictions)),
            "splits": {name: {"records": int(len(frame)),
                              "test_type": kind,
                              "families": frame.family.value_counts().to_dict()}
                       for name, (_, kind, frame) in eval_frames.items()},
            "density_matrices_exported": len(density_index),
            "density_matrices_grid": "canonical z,y,x; density e/Bohr^3",
            "field_figure_examples": {
                element: {key: sample[key] for key in ("calc_id", "evaluation_split", "z_index")}
                for element, sample in field_examples.items()
            },
            "field_figures_potential_gauge": "zero mean per 3D configuration; Ry after scaling",
            "classification_tasks": ["density_high", "potential_positive",
                                     "weak_cohesion", "energy_high_tail"],
            "confusion_layout": "rows reference [negative, positive]; columns prediction [negative, positive]",
            "classification_count_units": "voxels for density/potential; configurations for energy",
            "energy_error_sign": "prediction minus DFT target; positive means overprediction",
        }
        with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)
        print("\nMetrics by split, element, and family:")
        print(summary.to_string(index=False))
        print(f"\nFiles saved under: {output_dir}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
