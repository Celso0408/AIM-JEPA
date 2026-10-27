# Periodic Field JEPA – figure coverage

Input directory: `/home/rafael/Documentos/Pos_doc-BIO-IA/Task-1/single-element_GNN-JEPA/V33/outputs/periodic_jepa_al_fe_to_ni_v33_phase_geometry_equivariant_physical/seed_42_v33_test-epochs`
Figures: 211 files (211 distinct plots); formats: png.
Observed stage histories: autoencoder, D_latent, E_density, E_density_potential, E_full, F_periodic_ramp, P_cohesive_refinement, G_cohesive_calibration, G_adapter_warmup, G_joint_finetune.
Observed per-configuration physical validations: autoencoder, D_latent, E_density, E_density_potential, E_full, F_periodic_ramp, P_cohesive_refinement, G_cohesive_calibration, G_adapter_warmup.

## Interpretation boundaries

- Every physical-validation curve in these files is validation, not held-out test or OOD performance.
- `P_cohesive_refinement`: Accepted P checkpoint (head-only refinement); validation data, not held-out test.
- `G_cohesive_calibration` includes per-configuration physical validation.
- Raw density/potential voxel maps, atomic/graph activations, individual Fourier coefficients, per-sample latent embeddings, causal interventions, baseline-model runs, and uncertainty calibration cannot be reconstructed from these summary files alone.
- `G_joint_finetune` recorded 0 accepted checkpoints; its epoch curves are validation diagnostics.
- Held-out ID/OOD performance was omitted because no verified inference CSV was available.

## Missing stage / test inputs (conditional plots omitted)

- `history_G_retention_repair.csv`
- `verified G_adapter_warmup_all_test_predictions/predictions_per_configuration.csv`

## Read / plotting warnings

- Held-out predictions unavailable: /home/rafael/Documentos/Pos_doc-BIO-IA/Task-1/single-element_GNN-JEPA/V33/outputs/periodic_jepa_al_fe_to_ni_v33_phase_geometry_equivariant_physical/seed_42_v33_test-epochs/G_adapter_warmup_all_test_predictions/predictions_per_configuration.csv
