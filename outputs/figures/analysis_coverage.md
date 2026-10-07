# Periodic Field JEPA – figure coverage

Input directory: `/home/rafael/Documentos/Pos_doc-BIO-IA/Task-1/single-element_GNN-JEPA/V31/outputs/periodic_jepa_al_fe_to_ni_v31_gate_driven_curriculum_physical/seed_42_v31_development_physical_r1`
Figures: 176 files (176 distinct plots); formats: png.
Observed stage histories: autoencoder, D_latent, E_density, E_density_potential, E_full, F_periodic_ramp, P_cohesive_refinement, G_cohesive_calibration.
Observed per-configuration physical validations: autoencoder, D_latent, E_density, E_density_potential, E_full, F_periodic_ramp, P_cohesive_refinement.

## Interpretation boundaries

- Every physical-validation curve in these files is validation, not held-out test or OOD performance.
- `P_cohesive_refinement` is a candidate evaluation. Its checkpoint must not be treated as adopted unless the stage audit reports acceptance.
- `G_cohesive_calibration` in this run has a history but no Ni per-configuration physical-validation CSV. Ni gate graphs use its epoch summaries.
- Raw density/potential voxel maps, atomic/graph activations, individual Fourier coefficients, per-sample latent embeddings, causal interventions, baseline-model runs, and uncertainty calibration cannot be reconstructed from these summary files alone.
- No downstream Ni joint-training, held-out test, or family-OOD performance is inferred from a missing output.

## Missing stage / test inputs (conditional plots omitted)

- `history_G_adapter_warmup.csv`
- `history_G_joint_finetune.csv`
- `history_G_retention_repair.csv`
- `physical_validation_G_cohesive_calibration.csv`
- `ni_id_holdout_test.csv`
- `ni_family_ood_test.csv`

## Read / plotting warnings

- None.
