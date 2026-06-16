# Reflectometry POC synthetic dataset

Generated with `make_reflectometry_poc_dataset.py`.

## Purpose

Starter dataset for a JEPA-style inverse reflectometry proof of concept:

- input: `logR_obs[i] = log10(R_obs(q))`
- structural target: `sld[i] = SLD_i(z)`
- optional parametric labels: `layer_d`, `layer_rho`, `interface_sigma`, `substrate_rho`

## Arrays in `reflectometry_poc_initial_3000.npz`

| key | shape | units / meaning |
|---|---:|---|
| `q` | `(256,)` | momentum transfer, 1/Å |
| `z` | `(512,)` | depth grid, Å |
| `logR_obs` | `(3000, 256)` | noisy observed reflectivity, log10 scale |
| `logR_true` | `(3000, 256)` | noiseless model reflectivity after scale/background, log10 scale |
| `sigma_logR` | `(3000, 256)` | approximate uncertainty in log10 reflectivity |
| `sld` | `(3000, 512)` | SLD profile, 1/Å² |
| `n_layers` | `(3000,)` | number of finite layers, 1--3 |
| `layer_d` | `(3000, 3)` | layer thicknesses, Å; zero-padded |
| `layer_rho` | `(3000, 3)` | layer SLDs, 1/Å²; zero-padded |
| `interface_sigma` | `(3000, 4)` | interfacial roughnesses, Å; zero-padded |
| `substrate_rho` | `(3000,)` | substrate SLD, 1/Å² |
| `scale` | `(3000,)` | multiplicative reflectivity scale |
| `background` | `(3000,)` | additive reflectivity background |
| `meta_json` | scalar string | generation metadata |

## Suggested split

For quick testing:

- train: first 2400 samples
- validation: next 300 samples
- test: last 300 samples

For a real POC, regenerate 50k--200k samples using the script and create a separate out-of-distribution test set by changing thickness ranges, SLD ranges, noise, q-range, or max layer count.

## Caveat

This is a simple slab simulator using Parratt recursion with Nevot-Croce roughness. It is good for ML scaffolding, not for final scientific validation.
