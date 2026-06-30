# AIM-Bio AIM-JEPA Multimodal Synthetic POC

This repository contains a controlled proof of concept for **AIM-JEPA**, a graph-based Joint-Embedding Predictive Architecture for multimodal scientific imaging and scattering data.

The project is framed around the AIM-Bio proposal: learning latent structural relationships between scattering curves, microscopy-like images, reciprocal-space maps, metadata, and hidden structural profiles. The current data are **synthetic procedural benchmarks** for method development. They are not biological validation and should not be described as real biofilm microscopy or real biofilm scattering data.

## Repository Contents

| File | Purpose |
|---|---|
| `AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb` | Main Colab notebook. Trains the GNN + KAN AIM-JEPA model, runs ablations, diagnostics, retrieval metrics, and visualizations. |
| `make_aimbio_multimodal_poc_dataset.py` | Reproducible generator for `aimbio_multimodal_poc_5000.npz`. |
| `aimbio_multimodal_poc_5000.npz` | Larger synthetic multimodal AIM-Bio-style dataset used by the current notebook. |
| `aimbio_multimodal_poc_512.npz` | Smaller synthetic dataset kept for fast local checks and backward compatibility. |
| `aimbio_multimodal_poc_README.md` | Companion methodology/data README. |
| `AIM_JEPA_GNN_KAN_results_summary.md` | Earlier local smoke-test summary. |
| `Molecular_JEPA_QM9_Colab_Starter.ipynb` | Previous molecular JEPA project used as conceptual inspiration for masked-node JEPA and EMA teacher ideas. |
| `AIM_JEPA_molecular_transfer_recommendations.md` | Notes on method transfer from molecular JEPA to AIM-Bio AIM-JEPA. |

## Scientific Status

This repository demonstrates a **synthetic multimodal method benchmark**, not a biological result.

Supported claims:

- The notebook constructs an attributed graph from synthetic scattering, microscopy-like image patches, reciprocal-space maps, and metadata.
- The model predicts frozen latent structural SLD embeddings from visible multimodal context.
- The workflow can evaluate retrieval, decoded structural diagnostics, graph ablations, modality ablations, masked-node JEPA behavior, KAN coefficient diagnostics, spectral regularization diagnostics, and collapse checks.

Claims not yet supported:

- Biological validation on real biofilm data.
- A demonstrated real-world biofilm marker or structural signature.
- Formal interpretability of KAN coefficients.
- General missing-modality or q-sector prediction beyond the tasks explicitly evaluated in the notebook.

## Dataset Generation

The larger dataset is generated directly by:

```bash
python3 make_aimbio_multimodal_poc_dataset.py \
  --n 5000 \
  --seed 20260617 \
  --output aimbio_multimodal_poc_5000.npz
```

The generator creates independent synthetic samples rather than augmenting the 512-sample file. It uses:

- a synthetic layered scattering/reflectometry model,
- SLD profile generation from layer thickness, density, roughness, and substrate parameters,
- a simple Parratt recursion with Nevot-Croce roughness for `logR_true`,
- heteroscedastic noise for `logR_obs`,
- synthetic morphology factors shared across image-like modalities,
- procedural CLSM-like, SEM-like, q-map, height-map, and mask proxies,
- categorical metadata for growth state, preparation protocol, and strain proxy.

The default script writes a compressed `.npz`. Use `--uncompressed` if you want faster loading at the cost of a larger file:

```bash
python3 make_aimbio_multimodal_poc_dataset.py \
  --n 5000 \
  --seed 20260617 \
  --output aimbio_multimodal_poc_5000_uncompressed.npz \
  --uncompressed
```

## Dataset Arrays

The first dimension is `N`; for the main benchmark `N=5000`.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `q` | `(256,)` | `float32` | q-grid in inverse Angstrom. |
| `z` | `(512,)` | `float32` | depth grid in Angstrom. |
| `logR_obs` | `(N, 256)` | `float32` | noisy log10 scattering/reflectometry curve. |
| `logR_true` | `(N, 256)` | `float32` | noiseless synthetic curve. |
| `sigma_logR` | `(N, 256)` | `float32` | q-dependent logR uncertainty estimate. |
| `sld` | `(N, 512)` | `float32` | synthetic structural SLD profile target. |
| `n_layers` | `(N,)` | `int64` | number of synthetic layers. |
| `layer_d` | `(N, 3)` | `float32` | synthetic layer thickness parameters. |
| `layer_rho` | `(N, 3)` | `float32` | synthetic layer SLD/density parameters. |
| `interface_sigma` | `(N, 4)` | `float32` | synthetic interface roughness parameters. |
| `substrate_rho` | `(N,)` | `float32` | substrate SLD proxy. |
| `scale` | `(N,)` | `float32` | reflectometry scale factor. |
| `background` | `(N,)` | `float32` | reflectometry background term. |
| `clsm` | `(N, 2, 64, 64)` | `float16` | two-channel CLSM-like image proxy; channel 0 is cell-like signal and channel 1 is matrix/EPS-like signal. |
| `sem` | `(N, 1, 64, 64)` | `float16` | SEM-like morphology proxy. |
| `height_map` | `(N, 64, 64)` | `float16` | latent topography/film-height proxy used by the generator. |
| `qmap` | `(N, 1, 64, 64)` | `float16` | reciprocal-space intensity-map proxy. |
| `mask` | `(N, 64, 64)` | `uint8` | synthetic segmentation/region mask. |
| `morph_features` | `(N, 8)` | `float32` | compact morphology vector: thickness, roughness, density, contrast, matrix fraction, cell fraction, height spread, SEM texture. |
| `growth_state` | `(N,)` | `int64` | synthetic state label: sparse/planktonic-like, weakly organized, matrix-rich/biofilm-like. |
| `prep_protocol` | `(N,)` | `int64` | synthetic preparation label: hydrated-like, fixed-like, dried/artifact-prone-like. |
| `strain_id` | `(N,)` | `int64` | synthetic organism/strain proxy. |
| `meta_json` | scalar string | Unicode | reflectometry generator metadata. |
| `image_meta_json` | scalar string | Unicode | image/modality generator metadata. |

## Notebook Methodology

`AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb` implements a two-stage latent-prediction workflow.

### Stage 1: Frozen SLD Target Encoder

The notebook first trains an SLD autoencoder:

```text
SLD(z) -> SLD encoder -> z_sld -> SLD decoder -> reconstructed SLD(z)
```

After this stage, the SLD encoder is frozen. Its latent vector `z_sld` becomes the structural target for AIM-JEPA training. The decoder is used only for diagnostics and visualization, such as decoded SLD MAE and predicted-vs-true SLD plots.

### Stage 2: Graph AIM-JEPA Context Model

The context branch receives only observed modalities:

```text
logR_obs + sigma_logR + CLSM + SEM + qmap + metadata
```

It does **not** receive the true SLD profile or frozen SLD latent during graph construction or prediction. The training target is:

```text
z_target = frozen_sld_encoder(SLD)
```

The context model predicts:

```text
z_pred = GNN_KAN_AIMJEPA(context_graph)
```

The key JEPA property is that the model predicts a **latent structural embedding**, not raw pixels, raw q-map intensities, raw scattering curves, or raw SLD values.

## Graph Representation

Each sample is converted into a fixed-size attributed graph with **66 nodes**:

| Node type | Count | Meaning |
|---|---:|---|
| Scattering interval nodes | 16 | `logR_obs` split into q-intervals, with q coordinates, mean/std curve features, and optional uncertainty. |
| CLSM patch nodes | 16 | 4x4 grid over the two-channel CLSM-like image. |
| SEM patch nodes | 16 | 4x4 grid over the SEM-like image. |
| q-map patch nodes | 16 | 4x4 grid over the reciprocal-space map. |
| Metadata node | 1 | Embeddings for growth state, preparation protocol, strain proxy, plus normalized morphology features. |
| Global context node | 1 | Learnable node used for graph readout. |

The fixed graph has **530 directed edges** in the full-graph setting. Edge types include:

- q-adjacency between neighboring scattering intervals,
- spatial 4-neighbor adjacency within CLSM patches,
- spatial 4-neighbor adjacency within SEM patches,
- spatial 4-neighbor adjacency within q-map patches,
- cross-modal patch correspondence between CLSM, SEM, and q-map patches,
- metadata links to all other nodes,
- global-node links to all other nodes.

Edge attributes include an edge-type one-hot vector plus simple relative coordinate/distance features.

The notebook batches graphs without PyTorch Geometric. It keeps a shared fixed `edge_index` and `edge_attr`, and stores node features as:

```text
batch_node_features: (B, V, node_dim)
edge_index:          (2, E)
edge_attr:           (E, edge_attr_dim)
```

Message passing is implemented with pure PyTorch indexing and `index_add`-style aggregation.

## GNN + KAN Architecture

The main context model is `AIMJEPAGraphKAN`.

It contains:

- modality-specific feature builders for scattering intervals, image patches, q-map patches, and metadata,
- node-type embeddings,
- edge-type/edge-attribute embeddings,
- `GraphKANLayer` message-passing blocks,
- global-node or graph-level readout,
- KAN-style predictor head mapping the graph representation to the frozen SLD latent dimension.

The message-passing update is conceptually:

```text
message_ij = KAN_message([h_source, edge_attr_ij])
agg_j      = aggregate_i(message_ij)
h_j_new    = LayerNorm(h_j + KAN_update([h_j, agg_j]))
```

The notebook includes an MLP baseline only for comparison. The default context model is:

```python
cfg.context_model = "gnn_kan"
```

## KAN-Style Basis Layers

The notebook uses self-contained KAN-style basis-function layers. It does not depend on third-party KAN or spline packages.

The layer computes:

```text
y = base_linear(x) + sum_i,k coefficient[o, i, k] * B_k(x_i)
```

Available bases:

```python
fourier
chebyshev
legendre
hermite
laguerre
gegenbauer
rbf
```

The current checked-in notebook configuration uses:

```python
cfg.kan_basis = "legendre"
cfg.num_basis = 8
```

These are KAN-style basis expansions for method testing. The repository reports coefficient magnitudes as diagnostics, but it does not claim formal KAN interpretability.

## Masked-Node JEPA and EMA Teacher

The notebook also includes a masked-node JEPA objective inspired by the molecular JEPA starter project.

During training:

- the student graph receives a partially masked observed-modality graph,
- an EMA graph teacher encodes the unmasked observed graph,
- the student predicts hidden node embeddings,
- the target for this auxiliary task comes from the EMA teacher, not from the SLD target.

This tests missing-node and missing-modality behavior while preserving the main AIM-JEPA target:

```text
visible graph context -> frozen SLD latent target
```

## Losses and Regularization

The Stage 2 objective combines several terms:

- normalized latent MSE between `z_pred` and `z_target`,
- cosine alignment loss,
- InfoNCE retrieval/contrastive loss,
- optional SIGReg-style random-projection distribution regularization,
- random-projection variance anti-collapse penalty,
- target norm/std matching,
- optional KAN coefficient L1 penalty,
- optional masked-node JEPA loss against the EMA graph teacher,
- optional multifractal entropy spectral regularization.

### SIGReg Baseline

The SIGReg baseline can be run with:

```python
cfg.lambda_sigreg = 0.05
cfg.lambda_mhe = 0.0
cfg.lambda_trace = 0.0
```

### Multifractal Entropy Regularization

The notebook includes a multifractal entropy regularizer over the normalized covariance spectrum of predicted latent embeddings.

For probabilities `p_i` from normalized covariance eigenvalues, it computes an escort entropy using:

```text
beta = cfg.mhe_mf_d * q
pi_beta = softmax(beta * log(p_i))
S_beta = -sum_i pi_beta_i log(pi_beta_i)
effective_rank_beta = exp(S_beta)
```

Supported modes:

- `min_effective_rank`: penalize only if the effective rank drops below a floor,
- `target_effective_rank`: penalize distance from a chosen effective rank,
- `maximize_entropy`: push toward a flatter spectrum.

The notebook also includes a trace guard because entropy alone is scale-invariant and cannot prevent all-variance collapse.

The current checked-in configuration is an MHE replacement run:

```python
cfg.lambda_sigreg = 0.0
cfg.lambda_mhe = 0.10
cfg.lambda_trace = 0.05
cfg.mhe_mode = "min_effective_rank"
cfg.mhe_q_values = (0.5, 1.0, 2.0, 4.0)
cfg.mhe_mf_d = 1.0
```

Other useful ablations:

```python
# SIGReg + small MHE
cfg.lambda_sigreg = 0.05
cfg.lambda_mhe = 0.005
cfg.lambda_trace = 0.0

# Predictive cross-spectrum MHE
cfg.lambda_sigreg = 0.05
cfg.lambda_pred_mhe = 0.005
cfg.lambda_pred_trace = 0.0
```

Do not claim an improvement from a single seed. Compare settings using the same split, architecture, optimizer, and training budget, and report seed-to-seed variation.

## Run Presets

The notebook applies presets after creating the config object.

| Preset | AE epochs | JEPA epochs | Batch size | Hidden dim | GNN layers | Basis terms |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 1 | 1 | 16 | 64 | 1 | 6 |
| `short` | 3 | 5 | 16 | 64 | 1 | 6 |
| `normal` | 50 | 100 | 32 | 128 | 1 | 8 |

The current notebook default is:

```python
cfg.run_preset = "normal"
cfg.data_path = "/content/aimbio_multimodal_poc_5000.npz"
```

For a quick check, change:

```python
cfg.run_preset = "smoke"
```

## Running in Colab

1. Open `AIM_JEPA_GNN_KAN_multimodal_POC_colab.ipynb`.
2. Upload `aimbio_multimodal_poc_5000.npz` to Colab, or place it at `/content/aimbio_multimodal_poc_5000.npz`.
3. Start with `cfg.run_preset = "smoke"` to verify paths and package availability.
4. Use `cfg.run_preset = "normal"` for a serious single-seed run.
5. Outputs are saved under `/content/aim_jepa_gnn_kan_outputs` in Colab, or `aim_jepa_gnn_kan_outputs` locally.

The notebook uses:

- Python,
- NumPy,
- pandas,
- matplotlib,
- PyTorch.

PyTorch Geometric is not required.

## Outputs

The notebook writes:

| Output | Meaning |
|---|---|
| `aim_jepa_gnn_kan_checkpoint.pt` | Model checkpoint, graph spec, config, normalization stats, split indices. |
| `aim_jepa_gnn_kan_metrics_history.csv` | JEPA training/validation history. |
| `sld_autoencoder_metrics_history.csv` | Stage 1 autoencoder history. |
| `aim_jepa_gnn_kan_context_summary.csv` | Context-modality ablations. |
| `aim_jepa_gnn_kan_graph_ablation_summary.csv` | Edge-type graph ablations. |
| `aim_jepa_gnn_kan_node_ablation_summary.csv` | Node-modality ablations. |
| `aim_jepa_gnn_kan_diagnostics.csv` | KAN and spectral diagnostics. |
| `config.json` | Final runtime configuration. |

## Metrics to Inspect

Core metrics:

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
- predicted latent standard deviation,
- target latent standard deviation,
- collapse warning.

Graph and modality diagnostics:

- all modalities,
- scattering only,
- images only,
- microscopy only,
- q-map only,
- metadata only,
- no scattering,
- full graph,
- no cross-modal edges,
- no metadata edges,
- no q-adjacency edges,
- no spatial image adjacency edges,
- only global-node edges,
- removed scattering/CLSM/SEM/qmap/metadata nodes.

KAN and message-passing diagnostics:

- mean and max absolute basis coefficient value,
- finite KAN coefficient flag,
- KAN parameter count,
- KAN gradient norm,
- node embedding norm/std after message passing,
- global-node norm,
- oversmoothing score.

MHE diagnostics when enabled:

- `mhe_loss`,
- trace guard loss,
- covariance trace,
- participation ratio,
- condition number,
- per-beta entropy,
- per-beta normalized entropy,
- per-beta effective rank.

## Recent Single-Seed Notebook Result

The latest checked notebook output used:

```python
cfg.kan_basis = "legendre"
cfg.lambda_sigreg = 0.0
cfg.lambda_mhe = 0.10
cfg.lambda_trace = 0.05
cfg.seed = 7
```

On the synthetic 5000-sample dataset, the final all-context test summary was approximately:

| Metric | Test value |
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

These are single-seed synthetic benchmark numbers. They are useful for method debugging and proposal de-risking, but not sufficient for a biological or statistical performance claim.

## Interpreting the POC

A useful run should show:

- validation/test latent cosine above a random or undertrained baseline,
- Recall@5 and Recall@10 above random retrieval,
- decoded or retrieved SLD profiles that improve over random candidates,
- nonzero predicted latent spread,
- finite KAN coefficients and gradients,
- meaningful degradation under graph and modality ablations,
- no collapse warning.

The most important distinction is:

```text
Synthetic multimodal benchmark != AIM-Bio biological validation
```

The benchmark demonstrates that the proposed architecture and diagnostics can be exercised before real DESY/CSSB AIM-Bio data are available.

## Conservative Proposal Wording

As a controlled methodological benchmark, we implemented an AIM-JEPA prototype in which synthetic scattering profiles, microscopy-like images, reciprocal-space maps, and metadata are represented as attributed graphs. A graph neural network with KAN-style basis-function message passing predicts latent structural SLD embeddings, enabling evaluation of cross-modal latent prediction, retrieval, ablation sensitivity, spectral anti-collapse diagnostics, and missing-node behavior before transfer to real AIM-Bio biofilm data.

## Known Limitations

- The dataset is synthetic and procedurally generated.
- Metadata and morphology features are generator-derived and can be more informative than real experimental metadata would be.
- CLSM/SEM/qmap arrays are image-like proxies, not real microscopy.
- The SLD profile is a synthetic scattering target, not a directly measured biological structure.
- KAN basis coefficients are reported as diagnostics, not as validated scientific explanations.
- Current results are mostly single-seed experiments; use multi-seed comparisons before claiming improvement.
- The current target is the SLD latent. Additional tasks are needed to fully demonstrate hidden q-sector prediction or arbitrary missing-modality target prediction.

## Recommended Next Steps

1. Run the main configurations over at least 3 seeds.
2. Compare MHE replacement, SIGReg baseline, and SIGReg+MHE using the same split and budget.
3. Add best-validation checkpoint selection instead of relying only on final epoch.
4. Add post-training spectral diagnostics for every run, including SIGReg-only baselines.
5. Add explicit missing-modality target prediction tasks.
6. Add q-sector latent target prediction tasks.
7. Replace procedural proxies with pilot DESY/CSSB AIM-Bio data when available.
8. Keep proposal language conservative until biological validation is complete.
