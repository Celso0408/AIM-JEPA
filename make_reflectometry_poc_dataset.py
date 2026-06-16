#!/usr/bin/env python3
"""
Generate a small synthetic specular reflectometry dataset for inverse-model / JEPA POCs.

The dataset contains paired:
  x: noisy reflectivity curves log10(R_obs(q))
  y: SLD depth profiles rho(z)
  theta: slab parameters used to generate each curve

Units:
  q: 1/Angstrom
  z, thickness, roughness: Angstrom
  SLD rho: 1/Angstrom^2

This script uses a simple Parratt recursion with Nevot-Croce roughness. It is intended
for ML prototyping, not beamline-grade data reduction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover
    _erf = np.vectorize(math.erf)


def parratt_reflectivity(
    q: np.ndarray,
    rho: np.ndarray,
    thickness: np.ndarray,
    roughness: np.ndarray,
) -> np.ndarray:
    """Compute specular reflectivity R(q) for a slab stack.

    Parameters
    ----------
    q:
        Momentum-transfer grid, shape (nq,), in 1/Angstrom.
    rho:
        SLD for [ambient, layer_1, ..., layer_N, substrate], shape (N+2,),
        in 1/Angstrom^2.
    thickness:
        Thicknesses for [ambient, layer_1, ..., layer_N, substrate], shape (N+2,).
        Ambient and substrate thickness should be 0.
    roughness:
        Interfacial roughnesses, shape (N+1,), for interfaces
        ambient/layer_1, layer_1/layer_2, ..., layer_N/substrate.

    Returns
    -------
    R:
        Reflectivity, shape (nq,).
    """
    q = np.asarray(q, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    roughness = np.asarray(roughness, dtype=np.float64)

    # kz_j = sqrt(q^2/4 - 4*pi*rho_j). Add 0j so total-reflection regions are complex.
    kz = np.sqrt((q[:, None] ** 2) / 4.0 - 4.0 * np.pi * rho[None, :] + 0j)

    r_total = np.zeros_like(q, dtype=np.complex128)
    n_media = len(rho)

    # Work upward from substrate. Interface j lies between medium j and medium j+1.
    for j in range(n_media - 2, -1, -1):
        k_j = kz[:, j]
        k_next = kz[:, j + 1]
        denom = k_j + k_next
        r_j = (k_j - k_next) / np.where(np.abs(denom) == 0, 1e-30 + 0j, denom)

        sigma = float(roughness[j])
        if sigma > 0:
            # Nevot-Croce roughness factor.
            r_j *= np.exp(-2.0 * k_j * k_next * sigma**2)

        phase = np.exp(2j * k_next * thickness[j + 1])
        r_total = (r_j + r_total * phase) / (1.0 + r_j * r_total * phase)

    R = np.abs(r_total) ** 2
    return np.clip(R.real, 1e-12, 1.0)


def render_sld_profile(
    z: np.ndarray,
    rho: np.ndarray,
    thickness: np.ndarray,
    roughness: np.ndarray,
) -> np.ndarray:
    """Render a smooth SLD profile rho(z) using error-function interfaces."""
    z = np.asarray(z, dtype=np.float64)
    profile = np.full_like(z, float(rho[0]), dtype=np.float64)

    # Interfaces at z=0, d1, d1+d2, ..., sum(d_i).
    interface_z = [0.0]
    running = 0.0
    for d in thickness[1:-1]:
        running += float(d)
        interface_z.append(running)

    for j, z0 in enumerate(interface_z):
        sigma = max(float(roughness[j]), 1e-6)
        arg = (z - z0) / (math.sqrt(2.0) * sigma)
        profile += 0.5 * (float(rho[j + 1]) - float(rho[j])) * (1.0 + _erf(arg))

    return profile


def sample_one_stack(
    rng: np.random.Generator,
    max_layers: int = 3,
    max_total_thickness: float = 900.0,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one physically plausible-ish slab stack."""
    n_layers = int(rng.integers(1, max_layers + 1))

    # Rejection-sample thicknesses so everything fits on the fixed z-grid.
    for _ in range(100):
        d_layers = rng.uniform(25.0, 350.0, size=n_layers)
        if d_layers.sum() <= max_total_thickness:
            break
    else:
        d_layers *= max_total_thickness / d_layers.sum()

    # Generic neutron/X-ray-like SLD range. The values are intentionally broad.
    # Negative neutron SLDs can be added later; keep first POC positive and simple.
    rho_layers = rng.uniform(0.2e-6, 8.0e-6, size=n_layers)
    rho_substrate = rng.choice([2.07e-6, 3.47e-6, 4.15e-6, 6.35e-6])
    rho = np.concatenate([[0.0], rho_layers, [rho_substrate]])

    thickness = np.concatenate([[0.0], d_layers, [0.0]])

    roughness = np.empty(n_layers + 1, dtype=np.float64)
    for j in range(n_layers + 1):
        # Limit roughness relative to adjacent finite layer thicknesses.
        adjacent = []
        if 1 <= j <= n_layers:
            adjacent.append(d_layers[j - 1])
        if j + 1 <= n_layers:
            adjacent.append(d_layers[j])
        limit = 0.25 * min(adjacent) if adjacent else 20.0
        roughness[j] = rng.uniform(2.0, min(25.0, max(2.5, limit)))

    return n_layers, rho, thickness, roughness


def generate_dataset(
    n_samples: int,
    seed: int,
    nq: int = 256,
    nz: int = 512,
    max_layers: int = 3,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    q = np.linspace(0.005, 0.30, nq, dtype=np.float64)
    z = np.linspace(-50.0, 1000.0, nz, dtype=np.float64)

    logR_obs = np.empty((n_samples, nq), dtype=np.float32)
    logR_true = np.empty((n_samples, nq), dtype=np.float32)
    sigma_logR = np.empty((n_samples, nq), dtype=np.float32)
    sld = np.empty((n_samples, nz), dtype=np.float32)

    n_layers_arr = np.empty(n_samples, dtype=np.int64)
    layer_d = np.zeros((n_samples, max_layers), dtype=np.float32)
    layer_rho = np.zeros((n_samples, max_layers), dtype=np.float32)
    interface_sigma = np.zeros((n_samples, max_layers + 1), dtype=np.float32)
    substrate_rho = np.empty(n_samples, dtype=np.float32)
    scale = np.empty(n_samples, dtype=np.float32)
    background = np.empty(n_samples, dtype=np.float32)

    for i in range(n_samples):
        n_layers, rho, thickness, roughness = sample_one_stack(rng, max_layers=max_layers)

        R_true = parratt_reflectivity(q, rho, thickness, roughness)

        # Experimental nuisance terms.
        scale_i = float(rng.lognormal(mean=0.0, sigma=0.04))
        bg_i = float(10 ** rng.uniform(-9.0, -6.2))

        # Multiplicative noise in natural-log units; larger at high q.
        sigma_ln = 0.03 + 0.16 * (q / q.max()) ** 2 + rng.uniform(0.0, 0.015)
        R_noisy = (scale_i * R_true + bg_i) * np.exp(rng.normal(0.0, sigma_ln, size=nq))
        R_noisy = np.clip(R_noisy, 1e-12, 1.0)

        profile = render_sld_profile(z, rho, thickness, roughness)

        logR_obs[i] = np.log10(R_noisy).astype(np.float32)
        logR_true[i] = np.log10(np.clip(scale_i * R_true + bg_i, 1e-12, 1.0)).astype(np.float32)
        sigma_logR[i] = (sigma_ln / np.log(10.0)).astype(np.float32)
        sld[i] = profile.astype(np.float32)

        n_layers_arr[i] = n_layers
        layer_d[i, :n_layers] = thickness[1:-1].astype(np.float32)
        layer_rho[i, :n_layers] = rho[1:-1].astype(np.float32)
        interface_sigma[i, : n_layers + 1] = roughness.astype(np.float32)
        substrate_rho[i] = np.float32(rho[-1])
        scale[i] = np.float32(scale_i)
        background[i] = np.float32(bg_i)

    meta = {
        "description": "Synthetic specular reflectometry slab dataset for JEPA/inverse POC.",
        "forward_model": "Simple Parratt recursion with Nevot-Croce roughness.",
        "q_units": "1/Angstrom",
        "z_units": "Angstrom",
        "sld_units": "1/Angstrom^2",
        "input_recommendation": "Use q plus logR_obs; sigma_logR can be used for likelihood/ranking.",
        "target_recommendation": "Use sld as the structural target; layer_* arrays are optional labels.",
        "max_layers": max_layers,
        "seed": seed,
    }

    return {
        "q": q.astype(np.float32),
        "z": z.astype(np.float32),
        "logR_obs": logR_obs,
        "logR_true": logR_true,
        "sigma_logR": sigma_logR,
        "sld": sld,
        "n_layers": n_layers_arr,
        "layer_d": layer_d,
        "layer_rho": layer_rho,
        "interface_sigma": interface_sigma,
        "substrate_rho": substrate_rho,
        "scale": scale,
        "background": background,
        "meta_json": np.array(json.dumps(meta, indent=2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of samples")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--out", type=Path, default=Path("reflectometry_poc_dataset.npz"))
    args = parser.parse_args()

    data = generate_dataset(n_samples=args.n, seed=args.seed)
    np.savez_compressed(args.out, **data)
    print(f"Wrote {args.out} with {args.n} samples")
    print("Arrays:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:16s} {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
