"""Final provenance, checkpoint, split, and metric audits for the ablation."""

assert (OUT/"best.pt").exists() and (OUT/"last.pt").exists()
assert (OUT/"solution_autoencoder.pt").exists()
assert (OUT/"training_history.csv").exists()
assert (OUT/"per_state_metrics.csv").exists()
assert loaded_checkpoint["checkpoint_role"]=="validation_best"
assert loaded_checkpoint["experiment_name"]==EXPERIMENT_NAME
assert loaded_checkpoint["split_manifest_hash"]==SPLIT_MANIFEST_HASH
assert loaded_checkpoint["dataset_manifest_hash"]==DATASET_MANIFEST_HASH
assert model.decoder_type=="unconstrained" and model.primary_output=="direct_unconstrained"
assert not model.refinement_enabled and not list(model.residual_correction.parameters())
assert set(per_state_output_metrics.output_route)=={"direct_unconstrained"}
assert not (REMOVED_LOSSES&set(DEFAULT_LOSS_WEIGHTS))
assert all(not (REMOVED_LOSSES&set(stage["weights"])) for stage in stage_definitions)

_id_splits={part:{record["group_id"] for record in rows} for part,rows in splits.items()}
assert not (_id_splits["train"]&_id_splits["val"])
assert not (_id_splits["train"]&_id_splits["test"])
assert not (_id_splits["val"]&_id_splits["test"])
assert set(id_metrics.group_id)==_id_splits["test"]

_baseline_matches=[]
for _run in (PROJECT_DIR.parent/"outputs").glob("true_jepa_*"):
    _manifest=_run/"split_manifest.json"
    _config=_run/"config.json"
    if not (_manifest.exists() and _config.exists()): continue
    try:
        _baseline_config=json.loads(_config.read_text(encoding="utf-8"))
        _baseline_split=json.loads(_manifest.read_text(encoding="utf-8"))
    except (OSError,ValueError): continue
    if _baseline_config.get("mode")==CFG.mode and _baseline_config.get("seed")==CFG.seed:
        _baseline_hash=hashlib.sha256(json.dumps(_baseline_split,sort_keys=True).encode()).hexdigest()
        _baseline_matches.append({"run":_run.name,"split_manifest_hash":_baseline_hash,
                                  "identical_split":_baseline_hash==SPLIT_MANIFEST_HASH})
if _baseline_matches and not any(row["identical_split"] for row in _baseline_matches):
    raise AssertionError("No matching V08 run has the same train/val/test split")

_training=pd.read_csv(OUT/"training_history.csv")
assert not _training.empty and np.isfinite(_training[["train_loss","val_fidelity","validation_score"]].to_numpy()).all()
_expected_stages=[stage["name"] for stage in stage_definitions if stage["epochs"]>0]
assert set(_expected_stages)<=set(_training.stage)
_summary={
    "experiment_name":EXPERIMENT_NAME,"run_mode":CFG.mode,"seed":CFG.seed,
    "architecture_version":ARCHITECTURE_VERSION,"source_sha256":SOURCE_SHA256,
    "config_hash":CONFIG_HASH,"code_hash":CODE_HASH,
    "dataset_manifest_hash":DATASET_MANIFEST_HASH,"split_manifest_hash":SPLIT_MANIFEST_HASH,
    "checkpoint_role":"validation_best","selected_stage":best_stage,
    "selected_epoch":int(best_epoch),"optimizer_updates":int(GRADIENT_UPDATE_AUDIT["optimizer_updates"]),
    "parameter_count":int(sum(p.numel() for p in model.parameters())),
    "trainable_parameter_count":int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    "id_hamiltonians":int(id_metrics.group_id.nunique()),
    "ood_categories":int(ood_metrics.category.nunique()) if not ood_metrics.empty else 0,
    "id_mean_fidelity":float(id_metrics.fidelity.mean()),
    "id_mean_independent_residual":float(id_metrics.independent_residual_head.mean()),
    "id_raw_node_accuracy":float(id_metrics.raw_node_correct.mean()),
    "id_mean_gram_error":float(id_metrics.drop_duplicates("group_id").gram_max_error.mean()),
    "reference_baselines":_baseline_matches,
    "removed_principles":["Schrodinger residual training","Sturm nodal architecture and supervision",
        "positive spectral gaps","state orthogonality constraint","Rayleigh-Ritz variational path"],
    "retained_principles":["Dirichlet boundary","unit normalization","density",
        "H1 supervision","tail localization","reflection consistency","spectral subspace supervision"],
    "audit":"PASS",
}
(OUT/"run_card.json").write_text(json.dumps(_summary,indent=2,default=str))
(OUT/"physicless_ablation_audit.json").write_text(json.dumps({
    "run":EXPERIMENT_NAME,"status":"PASS","same_fem_solver":SOLVER_VERSION,
    "same_split_as_available_baseline":any(row["identical_split"] for row in _baseline_matches)
        if _baseline_matches else None,
    "split_disjoint":True,"target_free_inference":True,
    "checkpoint_integrity":True,"finite_metrics":True,
    "removed_training_losses_absent":True,"no_ritz_or_refinement":True,
},indent=2))
print("PHYSICLESS FULL RUN AUDIT: PASS",_summary)
