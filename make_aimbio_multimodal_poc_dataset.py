#!/usr/bin/env python3
"""Generate a synthetic AIM-Bio multimodal JEPA proof-of-concept dataset.

This script creates independent synthetic samples rather than augmenting the
existing 512-example file. The schema is compatible with
`AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def blur_roll(x: np.ndarray, steps: int = 2) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    for _ in range(int(steps)):
        y = (
            y
            + np.roll(y, 1, axis=0)
            + np.roll(y, -1, axis=0)
            + np.roll(y, 1, axis=1)
            + np.roll(y, -1, axis=1)
        ) / 5.0
    return y


def upsample_smooth_noise(rng: np.random.Generator, low: int = 16, high: int = 64, blur: int = 3) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=(low, low)).astype(np.float32)
    scale = high // low
    field = np.repeat(np.repeat(field, scale, axis=0), scale, axis=1)
    return normalize01(blur_roll(field, blur))


def make_sld_profile(
    z: np.ndarray,
    n_layers: int,
    layer_d: np.ndarray,
    layer_rho: np.ndarray,
    interface_sigma: np.ndarray,
    substrate_rho: float,
) -> np.ndarray:
    values = [0.0] + [float(layer_rho[i]) for i in range(n_layers)] + [float(substrate_rho)]
    positions = [0.0]
    total = 0.0
    for i in range(n_layers):
        total += float(layer_d[i])
        positions.append(total)

    profile = np.full_like(z, values[0], dtype=np.float32)
    for i, pos in enumerate(positions):
        sigma = max(float(interface_sigma[i]), 1.0)
        step = 0.5 * (1.0 + np.tanh((z - pos) / (1.5 * sigma)))
        profile += (values[i + 1] - values[i]) * step.astype(np.float32)
    return profile.astype(np.float32)


def parratt_log_reflectivity(
    q: np.ndarray,
    n_layers: int,
    layer_d: np.ndarray,
    layer_rho: np.ndarray,
    interface_sigma: np.ndarray,
    substrate_rho: float,
    scale: float,
    background: float,
) -> np.ndarray:
    """Simple Parratt recursion with Nevot-Croce roughness."""
    rho = np.zeros(n_layers + 2, dtype=np.float64)
    rho[1 : n_layers + 1] = layer_rho[:n_layers].astype(np.float64)
    rho[n_layers + 1] = float(substrate_rho)

    q64 = q.astype(np.float64)
    kz = np.sqrt((q64[None, :] / 2.0) ** 2 - 4.0 * math.pi * rho[:, None] + 0j)

    def interface_r(j: int) -> np.ndarray:
        denom = kz[j] + kz[j + 1]
        r = (kz[j] - kz[j + 1]) / np.where(np.abs(denom) < 1e-14, 1e-14 + 0j, denom)
        sig = float(interface_sigma[j])
        if sig > 0:
            r = r * np.exp(-2.0 * kz[j] * kz[j + 1] * sig * sig)
        return r

    r_eff = interface_r(n_layers)
    for layer_idx in range(n_layers - 1, -1, -1):
        rj = interface_r(layer_idx)
        phase = np.exp(2j * kz[layer_idx + 1] * float(layer_d[layer_idx]))
        r_eff = (rj + r_eff * phase) / (1.0 + rj * r_eff * phase + 1e-14)

    refl = np.abs(r_eff) ** 2
    refl = float(scale) * refl + float(background)
    refl = np.clip(refl, 1e-9, 1.0)
    return np.log10(refl).astype(np.float32)


def make_qmap(
    q: np.ndarray,
    log_r: np.ndarray,
    morph: np.ndarray,
    rng: np.random.Generator,
    size: int = 64,
) -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)].astype(np.float32)
    angle = rng.uniform(0.0, math.pi)
    ca, sa = math.cos(angle), math.sin(angle)
    xr = ca * xx + sa * yy
    yr = -sa * xx + ca * yy
    rad = np.sqrt(xr * xr + yr * yr)
    q_rad = q.min() + np.clip(rad, 0.0, 1.0) * (q.max() - q.min())

    curve = normalize01(log_r)
    radial = np.interp(q_rad.ravel(), q, curve).reshape(size, size).astype(np.float32)
    radial = normalize01(radial)

    contrast = float(morph[3])
    matrix = float(morph[4])
    ring_center = 0.20 + 0.55 * float(morph[0])
    ring_width = 0.035 + 0.055 * float(morph[1])
    ring = np.exp(-0.5 * ((rad - ring_center) / ring_width) ** 2).astype(np.float32)
    streak = np.exp(-0.5 * (yr / (0.05 + 0.05 * matrix)) ** 2) * np.exp(-1.5 * rad)
    speckle = blur_roll(rng.random((size, size), dtype=np.float32), 1)

    qmap = 0.62 * radial + 0.22 * contrast * ring + 0.18 * matrix * streak + 0.08 * speckle
    # Keep the reciprocal-space map sparse, but not so dark that patch statistics
    # become nearly uninformative. The exponent/background were calibrated to the
    # original 512-sample POC qmap percentiles.
    qmap = normalize01(qmap) ** 0.85
    qmap = np.clip(qmap + 0.02 * speckle, 0.0, 1.0)
    return qmap.astype(np.float32)


def make_images(
    morph: np.ndarray,
    growth_state: int,
    prep_protocol: int,
    strain_id: int,
    rng: np.random.Generator,
    size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = upsample_smooth_noise(rng, high=size, blur=4)
    base2 = upsample_smooth_noise(rng, high=size, blur=3)
    texture = upsample_smooth_noise(rng, high=size, blur=1)

    matrix_fraction = float(morph[4])
    cell_fraction = float(morph[5])
    roughness = float(morph[1])
    rho_norm = float(morph[2])
    sem_texture = float(morph[7])

    threshold = np.quantile(0.65 * base + 0.35 * texture, 1.0 - np.clip(cell_fraction, 0.02, 0.45))
    mask = ((0.65 * base + 0.35 * texture) > threshold).astype(np.float32)
    mask = blur_roll(mask, 1)

    height = normalize01(0.55 * base + 0.25 * base2 + 0.20 * mask + 0.08 * roughness * texture)
    gy, gx = np.gradient(height)
    edge = normalize01(np.sqrt(gx * gx + gy * gy))

    growth_gain = [0.80, 1.00, 1.18][int(growth_state)]
    prep_gain = [0.92, 1.00, 1.16][int(prep_protocol)]
    strain_phase = [0.00, 0.08, -0.06][int(strain_id)]

    clsm_cell = 0.08 + growth_gain * (0.72 * mask + 0.18 * base)
    clsm_cell += rng.normal(0.0, 0.035, size=(size, size)).astype(np.float32)
    clsm_cell = np.clip(clsm_cell, 0.0, 1.0)

    clsm_matrix = 0.06 + matrix_fraction * (0.74 * height + 0.22 * base2) + 0.10 * rho_norm
    clsm_matrix += rng.normal(0.0, 0.030, size=(size, size)).astype(np.float32)
    clsm_matrix = np.clip(clsm_matrix, 0.0, 1.0)

    sem = 0.30 + 0.42 * height + 0.38 * edge * prep_gain + 0.10 * sem_texture * texture + strain_phase
    sem += rng.normal(0.0, 0.030, size=(size, size)).astype(np.float32)
    sem = np.clip(sem, 0.0, 1.0)

    clsm = np.stack([clsm_cell, clsm_matrix], axis=0).astype(np.float32)
    return clsm, sem[None, ...].astype(np.float32), height.astype(np.float32), (mask > 0.42).astype(np.uint8)


def generate_dataset(n: int, seed: int, image_size: int = 64, qmap_size: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = np.linspace(0.005, 0.300, 256, dtype=np.float32)
    z = np.linspace(-50.0, 1000.0, 512, dtype=np.float32)

    log_r_true = np.empty((n, q.size), dtype=np.float32)
    log_r_obs = np.empty_like(log_r_true)
    sigma_log_r = np.empty_like(log_r_true)
    sld = np.empty((n, z.size), dtype=np.float32)

    n_layers = rng.choice(np.array([1, 2, 3]), size=n, p=np.array([0.36, 0.35, 0.29])).astype(np.int64)
    layer_d = np.zeros((n, 3), dtype=np.float32)
    layer_rho = np.zeros((n, 3), dtype=np.float32)
    interface_sigma = np.zeros((n, 4), dtype=np.float32)
    substrate_choices = np.array([2.07e-6, 3.47e-6, 6.35e-6], dtype=np.float32)
    substrate_rho = rng.choice(substrate_choices, size=n, p=np.array([0.30, 0.42, 0.28])).astype(np.float32)
    scale = np.clip(rng.lognormal(mean=0.0, sigma=0.045, size=n), 0.88, 1.16).astype(np.float32)
    background = np.power(10.0, rng.uniform(-9.0, -6.2, size=n)).astype(np.float32)

    clsm = np.empty((n, 2, image_size, image_size), dtype=np.float16)
    sem = np.empty((n, 1, image_size, image_size), dtype=np.float16)
    height_map = np.empty((n, image_size, image_size), dtype=np.float16)
    qmap = np.empty((n, 1, qmap_size, qmap_size), dtype=np.float16)
    mask = np.empty((n, image_size, image_size), dtype=np.uint8)

    morph_features = np.empty((n, 8), dtype=np.float32)
    growth_state = np.empty(n, dtype=np.int64)
    prep_protocol = rng.choice(np.array([0, 1, 2]), size=n, p=np.array([0.34, 0.44, 0.22])).astype(np.int64)
    strain_id = rng.choice(np.array([0, 1, 2]), size=n, p=np.array([0.30, 0.34, 0.36])).astype(np.int64)

    for i in range(n):
        nl = int(n_layers[i])
        layer_d[i, :nl] = rng.uniform(25.0, 350.0, size=nl).astype(np.float32)
        if layer_d[i, :nl].sum() > 930.0:
            layer_d[i, :nl] *= 930.0 / layer_d[i, :nl].sum()

        base_rho = rng.uniform(0.2e-6, 8.0e-6, size=nl)
        if rng.random() < 0.45 and nl > 1:
            base_rho = np.sort(base_rho) if rng.random() < 0.5 else np.sort(base_rho)[::-1]
        layer_rho[i, :nl] = base_rho.astype(np.float32)
        interface_sigma[i, : nl + 1] = rng.uniform(2.0, 25.0, size=nl + 1).astype(np.float32)

        total_thickness = float(layer_d[i, :nl].sum())
        used_rho = layer_rho[i, :nl]
        used_sigma = interface_sigma[i, : nl + 1]
        rough_norm = np.clip(used_sigma.mean() / 30.0, 0.0, 1.0)
        rho_norm = np.clip(used_rho.mean() / 8.0e-6, 0.0, 1.0)
        contrast_norm = np.clip((used_rho.max() - min(used_rho.min(), substrate_rho[i])) / 8.0e-6, 0.0, 1.0)
        total_norm = np.clip(total_thickness / 750.0, 0.0, 1.0)
        matrix_fraction = np.clip(0.18 + 0.55 * rho_norm + 0.18 * total_norm + rng.normal(0, 0.055), 0.16, 0.95)
        cell_fraction = np.clip(0.035 + 0.16 * (1.0 - matrix_fraction) + 0.08 * contrast_norm + rng.normal(0, 0.018), 0.02, 0.32)
        height_std = np.clip(0.095 + 0.10 * rough_norm + 0.025 * total_norm + rng.normal(0, 0.006), 0.08, 0.24)
        sem_texture = np.clip(0.090 + 0.070 * rough_norm + 0.012 * int(prep_protocol[i] == 2) + rng.normal(0, 0.006), 0.07, 0.19)
        morph_features[i] = np.array(
            [total_norm, rough_norm, rho_norm, contrast_norm, matrix_fraction, cell_fraction, height_std, sem_texture],
            dtype=np.float32,
        )

        growth_score = 0.55 * matrix_fraction + 0.30 * total_norm + 0.15 * rho_norm + rng.normal(0.0, 0.08)
        growth_state[i] = int(np.digitize(growth_score, bins=np.array([0.38, 0.68])))

        sld[i] = make_sld_profile(z, nl, layer_d[i], layer_rho[i], interface_sigma[i], float(substrate_rho[i]))
        log_r_true[i] = parratt_log_reflectivity(
            q,
            nl,
            layer_d[i],
            layer_rho[i],
            interface_sigma[i],
            float(substrate_rho[i]),
            float(scale[i]),
            float(background[i]),
        )

        sigma = 0.012 + 0.075 * (q / q.max()) ** 1.35 + rng.uniform(-0.003, 0.003)
        sigma = np.clip(sigma, 0.010, 0.095).astype(np.float32)
        sigma_log_r[i] = sigma
        noise = rng.normal(0.0, sigma).astype(np.float32)
        log_r_obs[i] = np.clip(log_r_true[i] + noise, -9.3, 0.0)

        c, s, h, m = make_images(
            morph_features[i],
            int(growth_state[i]),
            int(prep_protocol[i]),
            int(strain_id[i]),
            rng,
            size=image_size,
        )
        clsm[i] = c.astype(np.float16)
        sem[i] = s.astype(np.float16)
        height_map[i] = h.astype(np.float16)
        mask[i] = m.astype(np.uint8)
        qmap[i, 0] = make_qmap(q, log_r_true[i], morph_features[i], rng, size=qmap_size).astype(np.float16)

        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"generated {i + 1}/{n}", flush=True)

    meta_json = json.dumps(
        {
            "description": "Synthetic specular reflectometry slab dataset for JEPA/inverse POC.",
            "forward_model": "Simple Parratt recursion with Nevot-Croce roughness.",
            "q_units": "1/Angstrom",
            "z_units": "Angstrom",
            "sld_units": "1/Angstrom^2",
            "input_recommendation": "Use q plus logR_obs; sigma_logR can be used for likelihood/ranking.",
            "target_recommendation": "Use sld as the structural target; layer_* arrays are optional labels.",
            "max_layers": 3,
            "seed": seed,
            "n": n,
        },
        indent=2,
    )
    image_meta_json = json.dumps(
        {
            "description": "Synthetic AIM-Bio-like multimodal extension of a slab reflectometry POC dataset.",
            "modalities": {
                "logR_obs": "1D noisy reflectometry/scattering profile",
                "sld": "1D scattering length density profile",
                "clsm": "2-channel CLSM-like image proxy: channel 0 cells, channel 1 matrix/EPS-like signal",
                "sem": "1-channel SEM-like surface morphology proxy",
                "height_map": "latent topographic/film-height proxy shared by image generators",
                "qmap": "2D reciprocal-space intensity-map proxy generated from logR and morphology parameters",
                "mask": "simple synthetic segmentation/region mask",
                "morph_features": "[total_thickness_norm, roughness_norm, rho_norm, contrast_norm, matrix_fraction, cell_fraction, height_std, sem_texture]",
            },
            "growth_state_labels": {
                "0": "sparse/planktonic-like",
                "1": "weakly organized",
                "2": "matrix-rich/biofilm-like",
            },
            "prep_protocol_labels": {
                "0": "hydrated-like",
                "1": "fixed-like",
                "2": "dried/artifact-prone-like",
            },
            "strain_id_labels": {
                "0": "Pseudomonas-aeruginosa proxy",
                "1": "Vibrio-cholerae proxy",
                "2": "Pseudomonas-fluorescens proxy",
            },
            "config": {
                "image_size": image_size,
                "qmap_size": qmap_size,
                "seed": seed,
                "n": n,
                "generator": Path(__file__).name,
            },
            "note": "These images are procedural proxies for model/method development and proposal demonstration, not biological ground truth.",
        },
        indent=2,
    )

    return {
        "q": q,
        "z": z,
        "logR_obs": log_r_obs,
        "logR_true": log_r_true,
        "sigma_logR": sigma_log_r,
        "sld": sld,
        "n_layers": n_layers,
        "layer_d": layer_d,
        "layer_rho": layer_rho,
        "interface_sigma": interface_sigma,
        "substrate_rho": substrate_rho,
        "scale": scale,
        "background": background,
        "meta_json": np.array(meta_json),
        "source_index": np.arange(n, dtype=np.int64),
        "clsm": clsm,
        "sem": sem,
        "height_map": height_map,
        "qmap": qmap,
        "mask": mask,
        "morph_features": morph_features,
        "growth_state": growth_state,
        "prep_protocol": prep_protocol,
        "strain_id": strain_id,
        "image_meta_json": np.array(image_meta_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Number of synthetic samples.")
    parser.add_argument("--seed", type=int, default=20260617, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("aimbio_multimodal_poc_5000.npz"))
    parser.add_argument("--uncompressed", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    args = parser.parse_args()

    start = time.time()
    data = generate_dataset(args.n, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.uncompressed:
        np.savez(args.output, **data)
    else:
        np.savez_compressed(args.output, **data)
    elapsed = time.time() - start
    size_mb = args.output.stat().st_size / 1024**2
    print(f"saved {args.output} ({size_mb:.1f} MB) in {elapsed:.1f} s")

    for key in ["logR_obs", "sld", "clsm", "sem", "qmap", "morph_features"]:
        arr = data[key].astype(np.float64)
        print(
            f"{key:16s} shape={data[key].shape!s:18s} dtype={data[key].dtype} "
            f"min={np.nanmin(arr):.4g} mean={np.nanmean(arr):.4g} max={np.nanmax(arr):.4g}"
        )
    for key in ["growth_state", "prep_protocol", "strain_id", "n_layers"]:
        values, counts = np.unique(data[key], return_counts=True)
        print(f"{key:16s}", dict(zip(values.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
