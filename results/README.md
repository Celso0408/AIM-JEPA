# V12 result generation

Every stored result belongs to one run bundle, named after the run in `outputs/`:

```
results/
  generate_standard_results.py           # stored-data generator for the STANDARD bundle
  generate_results/                      # notebooks that produce the paper figures
  true_jepa_standard_fbc1677c_aee2f53f/  # run used by the paper
    01_metrics/  02_figures/  03_spectral_analysis/  generation_manifest.json
  true_jepa_full_5ba6bc71_aee2f53f/      # FULL run (same code, FULL preset)
    02_figures/
```

The authoritative stored-data generator for the completed V12 STANDARD run is:

```bash
/home/clouduser/.psijepavenv/bin/python results/generate_standard_results.py
```

It reads `outputs/true_jepa_standard_fbc1677c_aee2f53f` and writes the
`results/true_jepa_standard_fbc1677c_aee2f53f` bundle. It performs neither
training nor model inference.

The heatmap, training-OOD heatmap, curriculum, curriculum-dashboard,
spectral-tension, result-copy, and real-data Figure 3 notebooks in
`generate_results/` read the STANDARD run and write into its bundle;
`figure3_direct_vs_ritz_real_data_FULL.ipynb` does the same for the FULL run.

Paper figure sources (STANDARD bundle, `02_figures/`):

| Paper | Source |
|---|---|
| Fig. 2 | `figure4_residual_vs_n.csv`, `figure4_node_accuracy_vs_n.csv` |
| Fig. 4 (a,b) | `figure3_selected_case.csv`, `figure3_nodes.csv`, `figure3_selected_case_metadata.csv` |
| Fig. 4 (c), Table III | `figure5_family_vs_n_heatmap.csv`, `figure5_idrank_family_list.csv` |
| Fig. 6, Table IV | `fidelity_heatmap_regular_training_ood_no_eckart.*` |

Removed on 2026-11-02: root-level `01_metrics/` and `03_spectral_analysis/`
(they matched none of the runs in `outputs/`), the physicless comparison
(`02_figures/ablation/` and its two notebooks; there is no V12 physicless run
with a matching split), duplicated root heatmap CSVs, and the legacy
`generate_results/Untitled-2.ipynb` scratch notebook.
