# AIM-Bio Multimodal AIM-JEPA Synthetic POC

This file documents the synthetic dataset and the methodology used in `AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb`.

The repository is intended to be shared as a reproducible proof of concept for AIM-JEPA: a graph-based Joint-Embedding Predictive Architecture for multimodal scientific imaging, scattering, and metadata-rich datasets.

The current data are **synthetic procedural proxies**. They are useful for method development, proposal de-risking, and reviewer-facing demonstrations of the workflow, but they are not biological validation and should not be described as real biofilm data.

## Main Files

| File | Role |
|---|---|
| `AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb` | Main notebook implementing GNN + KAN AIM-JEPA. |
| `make_aimbio_multimodal_poc_dataset.py` | Generator for the 5000-sample synthetic dataset. |
| `aimbio_multimodal_poc_5000.npz` | Main synthetic AIM-Bio-style multimodal benchmark. |
| `aimbio_multimodal_poc_512.npz` | Smaller synthetic dataset for fast checks and compatibility. |
| `README.md` | Repository-level overview. |

## Generating `aimbio_multimodal_poc_5000.npz`

The 5000-sample dataset is generated from scratch with:

```bash
python3 make_aimbio_multimodal_poc_dataset.py \
  --n 5000 \
  --seed 20260617 \
  --output aimbio_multimodal_poc_5000.npz
```

The script creates independent synthetic samples. It does not simply augment the 512-sample file.

The generator uses:

- random layered SLD structures with 1-3 layers,
- synthetic layer thickness, density, interface roughness, substrate, scale, and background parameters,
- SLD profile construction on a depth grid,
- a simple Parratt recursion with Nevot-Croce roughness to generate synthetic reflectometry/scattering curves,
- q-dependent noise to create `logR_obs`,
- shared synthetic morphology factors,
- procedural CLSM-like, SEM-like, height-map, q-map, and mask proxies,
- categorical growth state, preparation protocol, and strain proxy metadata.

Optional uncompressed output:

```bash
python3 make_aimbio_multimodal_poc_dataset.py \
  --n 5000 \
  --seed 20260617 \
  --output aimbio_multimodal_poc_5000_uncompressed.npz \
  --uncompressed
```

## Dataset Schema

The first dimension is `N`. The main file uses `N=5000`.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `q` | `(256,)` | `float32` | q-grid in inverse Angstrom. |
| `z` | `(512,)` | `float32` | depth grid in Angstrom. |
| `logR_obs` | `(N, 256)` | `float32` | noisy log10 scattering/reflectometry profile. |
| `logR_true` | `(N, 256)` | `float32` | noiseless synthetic curve. |
| `sigma_logR` | `(N, 256)` | `float32` | approximate q-dependent uncertainty. |
| `sld` | `(N, 512)` | `float32` | synthetic structural SLD target. |
| `n_layers` | `(N,)` | `int64` | number of synthetic layers. |
| `layer_d` | `(N, 3)` | `float32` | layer thickness parameters. |
| `layer_rho` | `(N, 3)` | `float32` | layer SLD/density parameters. |
| `interface_sigma` | `(N, 4)` | `float32` | interface roughness parameters. |
| `substrate_rho` | `(N,)` | `float32` | substrate SLD proxy. |
| `scale` | `(N,)` | `float32` | reflectometry scale factor. |
| `background` | `(N,)` | `float32` | reflectometry background term. |
| `source_index` | `(N,)` | `int64` | generated sample index. |
| `clsm` | `(N, 2, 64, 64)` | `float16` | two-channel CLSM-like proxy: cells and matrix/EPS-like signal. |
| `sem` | `(N, 1, 64, 64)` | `float16` | SEM-like surface/morphology proxy. |
| `height_map` | `(N, 64, 64)` | `float16` | latent topography proxy used by the generator. |
| `qmap` | `(N, 1, 64, 64)` | `float16` | reciprocal-space intensity-map proxy. |
| `mask` | `(N, 64, 64)` | `uint8` | synthetic segmentation/region mask. |
| `morph_features` | `(N, 8)` | `float32` | compact morphology vector. |
| `growth_state` | `(N,)` | `int64` | sparse/planktonic-like, weakly organized, or matrix-rich/biofilm-like proxy. |
| `prep_protocol` | `(N,)` | `int64` | hydrated-like, fixed-like, or dried/artifact-prone-like proxy. |
| `strain_id` | `(N,)` | `int64` | synthetic organism/strain proxy. |
| `meta_json` | scalar string | Unicode | reflectometry generator metadata. |
| `image_meta_json` | scalar string | Unicode | multimodal image generator metadata. |

The 8 morphology features are:

```text
total_thickness_norm
roughness_norm
rho_norm
contrast_norm
matrix_fraction
cell_fraction
height_std
sem_texture
```

## What the Notebook Implements

The notebook implements a two-stage AIM-JEPA-style latent prediction benchmark.

### Stage 1: SLD Autoencoder

The SLD profile is treated as the hidden structural target. The notebook first trains an autoencoder:

```text
SLD(z) -> SLD encoder -> z_sld -> SLD decoder -> reconstructed SLD(z)
```

After training, the SLD encoder is frozen. Its latent representation is the target for JEPA training:

```text
z_target = frozen_sld_encoder(SLD)
```

The SLD decoder is used only for diagnostics and plots.

### Stage 2: GNN + KAN AIM-JEPA

The context model receives:

```text
logR_obs, sigma_logR, CLSM-like image, SEM-like image, qmap, growth_state, prep_protocol, strain_id, morph_features
```

The context model does **not** receive the true `sld` profile or the frozen SLD latent during graph construction.

The model predicts:

```text
z_pred = GNN_KAN_AIMJEPA(context_graph)
```

and trains against:

```text
z_target = frozen_sld_encoder(SLD)
```

This is a JEPA-style latent prediction task. The model is not trained to directly reconstruct raw pixels, raw q-map intensities, raw scattering curves, or raw SLD profiles.

## Graph Construction

Each multimodal sample is represented as a fixed graph with 66 nodes:

| Node type | Count | Node information |
|---|---:|---|
| Scattering nodes | 16 | q-interval statistics from `logR_obs`, q coordinates, and uncertainty. |
| CLSM nodes | 16 | 4x4 patch grid over the two-channel CLSM-like image. |
| SEM nodes | 16 | 4x4 patch grid over the SEM-like image. |
| q-map nodes | 16 | 4x4 patch grid over reciprocal-space map. |
| Metadata node | 1 | learnable embeddings for categorical metadata plus morphology features. |
| Global node | 1 | learnable context/readout node. |

The full graph has 530 directed edges. Edge types encode:

- q-adjacency between scattering intervals,
- spatial image-patch adjacency,
- cross-modal patch correspondence,
- metadata-to-context links,
- global-node links.

The implementation uses pure PyTorch fixed-graph batching:

```text
node_features: (B, V, node_dim)
edge_index:    (2, E)
edge_attr:     (E, edge_attr_dim)
```

PyTorch Geometric is not required.

## GNN + KAN Model

The main model is `AIMJEPAGraphKAN`.

It contains:

- modality feature builders,
- node-type embeddings,
- edge-attribute embeddings,
- KAN-style graph message passing,
- graph/global-node readout,
- KAN predictor head to the SLD latent dimension.

A graph message-passing layer follows:

```text
message_ij = KAN_message([h_source, edge_attr_ij])
agg_j      = aggregate_i(message_ij)
h_j_new    = LayerNorm(h_j + KAN_update([h_j, agg_j]))
```

## KAN-Style Basis Functions

The notebook uses self-contained KAN-style basis-function layers:

```text
y = base_linear(x) + basis_projection(B_k(x))
```

Available basis choices:

```text
fourier
chebyshev
legendre
hermite
laguerre
gegenbauer
rbf
```

The current checked-in configuration uses:

```python
cfg.kan_basis = "legendre"
cfg.num_basis = 8
```

These layers are used for node projection, message functions, update functions, the SLD-latent predictor, and the masked-node predictor. They are used as flexible basis-function layers, not as a proven source of scientific interpretability.

## Masked-Node JEPA With EMA Teacher

The notebook includes an auxiliary missing-node objective:

```text
masked observed graph -> student node embeddings
unmasked observed graph -> EMA teacher node embeddings
student predicts teacher embeddings for masked nodes
```

This branch does not use the SLD target. It is intended to test missing-node and missing-modality behavior in the observed graph while the main AIM-JEPA task predicts the frozen SLD latent.

## Losses

The Stage 2 loss can include:

- normalized latent MSE,
- cosine alignment,
- InfoNCE contrastive retrieval loss,
- SIGReg-style random-projection regularization,
- random-projection variance anti-collapse loss,
- target norm/std matching,
- optional KAN L1 regularization,
- masked-node JEPA loss,
- multifractal entropy spectral regularization.

## Multifractal Entropy Regularization

The notebook includes a multifractal entropy regularizer over the latent covariance spectrum.

Given normalized covariance eigenvalue probabilities `p_i`, it uses:

```text
beta = mf_d * q
pi_beta = softmax(beta * log(p_i))
S_beta = -sum_i pi_beta_i log(pi_beta_i)
effective_rank_beta = exp(S_beta)
```

Modes:

- `min_effective_rank`: prevent spectral collapse without forcing full isotropy,
- `target_effective_rank`: target a chosen intrinsic dimensionality,
- `maximize_entropy`: push toward a flatter spectrum.

The trace guard prevents scale collapse because entropy alone is scale-invariant.

The current checked-in notebook configuration is:

```python
cfg.lambda_sigreg = 0.0
cfg.lambda_mhe = 0.10
cfg.lambda_trace = 0.05
cfg.mhe_mode = "min_effective_rank"
cfg.mhe_mf_d = 1.0
cfg.mhe_q_values = (0.5, 1.0, 2.0, 4.0)
```

Common ablations:

```python
# SIGReg baseline
cfg.lambda_sigreg = 0.05
cfg.lambda_mhe = 0.0
cfg.lambda_trace = 0.0

# SIGReg + small MHE
cfg.lambda_sigreg = 0.05
cfg.lambda_mhe = 0.005
cfg.lambda_trace = 0.0

# MHE replacement
cfg.lambda_sigreg = 0.0
cfg.lambda_mhe = 0.10
cfg.lambda_trace = 0.05
```

## Notebook Presets

| Preset | AE epochs | JEPA epochs | Batch size | Hidden dim | GNN layers | Basis terms |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 1 | 1 | 16 | 64 | 1 | 6 |
| `short` | 3 | 5 | 16 | 64 | 1 | 6 |
| `normal` | 50 | 100 | 32 | 128 | 1 | 8 |

The default data path in Colab is:

```python
cfg.data_path = "/content/aimbio_multimodal_poc_5000.npz"
```

## Running the Notebook

In Colab:

1. Open `AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb`.
2. Upload `aimbio_multimodal_poc_5000.npz`, or place it at `/content/aimbio_multimodal_poc_5000.npz`.
3. Use `cfg.run_preset = "smoke"` for a path/package check.
4. Use `cfg.run_preset = "normal"` for a serious single-seed run.

Local execution should also work if the dataset is beside the notebook. The notebook uses only NumPy, pandas, matplotlib, and PyTorch.

## Metrics and Diagnostics

The notebook reports:

- SLD autoencoder train/validation loss,
- SLD reconstruction MAE,
- AIM-JEPA train/validation loss,
- latent MSE,
- latent cosine similarity,
- InfoNCE loss,
- Recall@1, Recall@5, Recall@10,
- median retrieval rank,
- decoded SLD MAE,
- top-1 and top-5 retrieved SLD MAE,
- predicted and target latent standard deviation,
- collapse warning,
- graph edge ablations,
- modality/context ablations,
- node-modality ablations,
- KAN coefficient statistics,
- KAN gradient norm,
- message-passing node norms/stds,
- global-node norm,
- oversmoothing score,
- MHE trace, participation ratio, condition number, per-beta entropy, and effective rank.

## Recent Single-Seed Result

The latest checked notebook output used the synthetic 5000-sample dataset with:

```python
cfg.kan_basis = "legendre"
cfg.lambda_sigreg = 0.0
cfg.lambda_mhe = 0.10
cfg.lambda_trace = 0.05
cfg.seed = 7
```

Final all-context test summary:

| Metric | Value |
|---|---:|
| latent cosine | 0.7968 |
| Recall@1 | 0.1947 |
| Recall@5 | 0.5640 |
| Recall@10 | 0.7387 |
| decoded SLD MAE | 0.7006 |
| top-5 retrieved SLD MAE | 0.4280 |
| predicted latent std | 0.0954 |
| target latent std | 0.0862 |
| collapse warning | false |
| MHE participation ratio | 18.15 |
| MHE condition number | 65.27 |

These values are single-seed synthetic benchmark diagnostics. They should not be treated as biological validation.

## Interpretation

Evidence that the synthetic method is working includes:

- non-collapsed predicted latents,
- latent cosine above random alignment,
- retrieval above random baseline,
- decoded or retrieved SLD candidates that resemble the true synthetic SLD,
- performance degradation under meaningful graph or modality ablations,
- finite KAN coefficients and stable gradients.

The main methodological claim is conservative:

> As a controlled methodological benchmark, AIM-JEPA represents synthetic scattering profiles, microscopy-like images, reciprocal-space maps, and metadata as attributed graphs. A graph neural network with KAN-style basis-function message passing predicts latent structural SLD embeddings, enabling evaluation of cross-modal latent prediction, retrieval, ablation sensitivity, spectral anti-collapse diagnostics, and missing-node behavior before transfer to real AIM-Bio biofilm data.

## Limitations

- The data are synthetic and procedurally generated.
- The microscopy-like arrays are not real CLSM or SEM data.
- Metadata categories are synthetic and may be easier to exploit than real experimental metadata.
- The SLD target is a synthetic scattering structure proxy.
- KAN coefficient diagnostics are not validated scientific interpretations.
- Single-seed improvements are not enough to claim a robust method improvement.
- Real AIM-Bio biological validation remains future work.

## Recommended Next Steps

1. Run SIGReg baseline, SIGReg+MHE, and MHE replacement over at least 3 seeds.
2. Add best-validation checkpoint selection.
3. Add post-training spectral diagnostics for every run, including SIGReg-only runs.
4. Add explicit missing-modality target prediction tasks.
5. Add q-sector latent target prediction tasks.
6. Replace synthetic proxies with pilot DESY/CSSB AIM-Bio data when available.
7. Keep proposal language clear: synthetic benchmark first, real biological validation later.
