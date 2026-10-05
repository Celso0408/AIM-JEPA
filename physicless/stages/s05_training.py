"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 61]
AE_NODE_THRESHOLDS=(1e-8,1e-5,3e-5,1e-4,3e-4,1e-3,2e-3)
AE_GATES={"reconstruction_fidelity_min":.995,"aligned_l2_max":.02,"h1_max":.05,"h1_seminorm_relative_max":.05,
          "normalization_error_max":1e-3,"boundary_error_max":1e-7,
          "latent_variance_min":1e-4,"latent_variance_max":100.0,"effective_rank_min":3.0,
          "active_covariance_rank_min":4,"active_condition_number_max":1e6,
          "near_zero_variance_fraction_max":.95,"dominant_covariance_fraction_max":.95}
AE_NODE_POLICY={"hard_gate":None,
                "diagnostic_only":["raw_node_accuracy@1e-8","persistent_node_accuracy@2e-3","orthogonality_max"],
                "reason":"Sturm and orthogonality gates are disabled only for the reduced-physics ablation"}


def autoencoder_metrics(m:TrueSchrodingerJEPA,loader:DataLoader)->dict[str,Any]:
    rows=[]; latents=[]; stability_thresholds=[]; latent_population=0
    latent_sample_per_batch=max(256,65536//max(len(loader),1))
    m.eval()
    with torch.no_grad():
        for raw in loader:
            b=to_device(raw); z,recon=m.encode_online(b); aligned,fid=globally_align_torch(recon,b["psi"],b["w"])
            raw_recon=spectral_to_wave(m.solution_reconstruction_decoder.head(z),b); _,raw_fid=globally_align_torch(raw_recon,b["psi"],b["w"])
            derivative_error_sq,derivative_target_sq=dimensionless_derivative_errors(aligned,b["psi"],b)
            l2_error_sq=torch.sum(b["w"][:,None]*(aligned-b["psi"]).square(),dim=-1)
            l2_target_sq=torch.sum(b["w"][:,None]*b["psi"].square(),dim=-1)
            l2=torch.sqrt(l2_error_sq/l2_target_sq.clamp_min(1e-12))
            h1_relative=torch.sqrt((l2_error_sq+derivative_error_sq)/(l2_target_sq+derivative_target_sq).clamp_min(1e-12))
            h1_seminorm_relative=torch.sqrt(derivative_error_sq/derivative_target_sq.clamp_min(1e-12))
            gram=torch.einsum("bkn,bn,bjn->bkj",recon,b["w"],recon); raw_gram=torch.einsum("bkn,bn,bjn->bkj",raw_recon,b["w"],raw_recon); norm_error=torch.abs(torch.sum(b["w"][:,None]*recon.square(),dim=-1)-1)
            flat_latent=z[b["state_mask"]].detach().cpu(); latent_population+=len(flat_latent)
            if len(flat_latent)>latent_sample_per_batch:
                sample_index=torch.linspace(0,len(flat_latent)-1,latent_sample_per_batch).round().long(); flat_latent=flat_latent[sample_index]
            latents.append(flat_latent)
            for index in range(len(raw["operator_records"])):
                valid=b["state_mask"][index]; n=int(b["lengths"][index]); recon_np=recon[index,valid,:n].cpu().numpy()
                target_np=b["psi"][index,valid,:n].cpu().numpy()
                node_curve=np.asarray([[persistent_nodes(pred,threshold)==persistent_nodes(target,threshold)
                                        for threshold in AE_NODE_THRESHOLDS] for pred,target in zip(recon_np,target_np)],dtype=bool)
                per_state_stability=[]
                for correct in node_curve:
                    stable=next((AE_NODE_THRESHOLDS[j] for j in range(len(AE_NODE_THRESHOLDS)) if bool(np.all(correct[j:]))),
                                2*AE_NODE_THRESHOLDS[-1])
                    per_state_stability.append(float(stable))
                stability_thresholds.extend(per_state_stability)
                curve_mean=node_curve.mean(axis=0)
                raw_node_accuracy=float(curve_mean[0]); noise_floor_node_accuracy=float(curve_mean[1]); persistent_node_accuracy=float(curve_mean[-1])
                gram_i=gram[index][valid][:,valid]; identity=torch.eye(int(valid.sum()),device=DEVICE)
                rows.append({"reconstruction_fidelity":float(fid[index,valid].mean()),"aligned_l2":float(l2[index,valid].mean()),
                             "h1":float(h1_relative[index,valid].mean()),"h1_seminorm_relative":float(h1_seminorm_relative[index,valid].mean()),"raw_node_accuracy":raw_node_accuracy,"noise_floor_node_accuracy":noise_floor_node_accuracy,
                             "persistent_node_accuracy":persistent_node_accuracy,"raw_projection_fidelity":float(raw_fid[index,valid].mean()),
                             **{f"node_accuracy_{threshold:.0e}":float(value) for threshold,value in zip(AE_NODE_THRESHOLDS,curve_mean)},
                             "normalization_error":float(norm_error[index,valid].max()),"orthogonality_max":float((gram_i-identity).abs().max()),"raw_projection_orthogonality_max":float((raw_gram[index][valid][:,valid]-identity).abs().max()),
                             "boundary_error":float(recon[index,valid][:,[0,n-1]].abs().max())})
    frame=pd.DataFrame(rows); diag=latent_diagnostics(torch.cat(latents,dim=0),None)
    result={key:float(frame[key].mean()) for key in ("reconstruction_fidelity","raw_projection_fidelity","aligned_l2","h1","h1_seminorm_relative","raw_node_accuracy","noise_floor_node_accuracy","persistent_node_accuracy")}
    result.update({key:float(frame[key].max()) for key in ("normalization_error","orthogonality_max","raw_projection_orthogonality_max","boundary_error")})
    result["node_accuracy_curve"]={f"{threshold:.0e}":float(frame[f"node_accuracy_{threshold:.0e}"].mean()) for threshold in AE_NODE_THRESHOLDS}
    result["node_stability_threshold_median"]=float(np.quantile(stability_thresholds,.50))
    result["node_stability_threshold_p95"]=float(np.quantile(stability_thresholds,.95))
    result["node_stability_fraction_by_2e-3"]=float(np.mean(np.asarray(stability_thresholds)<=AE_NODE_THRESHOLDS[-1]))
    result["latent_diagnostic_population_size"]=int(latent_population); result["latent_diagnostic_sample_size"]=int(sum(len(item) for item in latents))
    result.update(diag); return result


ae_initial_encoder_state={name:p.detach().cpu().clone() for name,p in model.solution_online_encoder.named_parameters()}
ae_optimizer=torch.optim.AdamW(list(model.solution_online_encoder.parameters())+list(model.solution_reconstruction_decoder.parameters()),lr=CFG.ae_learning_rate,weight_decay=CFG.weight_decay,foreach=False)
def ae_core_checks(metrics:Mapping[str,Any])->dict[str,bool]:
    scalar_keys=("reconstruction_fidelity","aligned_l2","h1","h1_seminorm_relative",
                 "normalization_error","boundary_error","latent_variance",
                 "effective_rank","active_covariance_rank","active_condition_number","near_zero_variance_fraction","dominant_covariance_fraction")
    finite=all(key in metrics and np.isfinite(float(metrics[key])) for key in scalar_keys)
    if not finite: return {"all_required_metrics_finite":False}
    return {"all_required_metrics_finite":True,
            "reconstruction_fidelity":metrics["reconstruction_fidelity"]>=AE_GATES["reconstruction_fidelity_min"],
            "aligned_l2":metrics["aligned_l2"]<=AE_GATES["aligned_l2_max"],
            "h1":metrics["h1"]<=AE_GATES["h1_max"],
            "h1_seminorm":metrics["h1_seminorm_relative"]<=AE_GATES["h1_seminorm_relative_max"],
            "normalization":metrics["normalization_error"]<=AE_GATES["normalization_error_max"],
            "boundary":metrics["boundary_error"]<=AE_GATES["boundary_error_max"],
            "latent_variance_lower":metrics["latent_variance"]>AE_GATES["latent_variance_min"],
            "latent_variance_upper":metrics["latent_variance"]<AE_GATES["latent_variance_max"],
            "effective_rank":metrics["effective_rank"]>AE_GATES["effective_rank_min"],
            "active_covariance_rank":metrics["active_covariance_rank"]>=AE_GATES["active_covariance_rank_min"],
            "active_condition_number":metrics["active_condition_number"]<AE_GATES["active_condition_number_max"],
            "near_zero_variance_fraction":metrics["near_zero_variance_fraction"]<AE_GATES["near_zero_variance_fraction_max"],
            "dominant_covariance_fraction":metrics["dominant_covariance_fraction"]<AE_GATES["dominant_covariance_fraction_max"]}


def ae_selection_score(metrics:Mapping[str,Any])->float:
    # Do not select the ablation's autoencoder using the removed priors.
    return (4*(1-metrics["reconstruction_fidelity"])+
            2*metrics["aligned_l2"]+2*metrics["h1"]+metrics["h1_seminorm_relative"])


def ae_candidate_key(metrics:Mapping[str,Any])->tuple[float,...]:
    checks=ae_core_checks(metrics); failures=sum(not passed for passed in checks.values())
    score=ae_selection_score(metrics); score=score if np.isfinite(score) else float("inf")
    noise=float(metrics.get("noise_floor_node_accuracy",0.0)); noise=noise if np.isfinite(noise) else 0.0
    raw=float(metrics.get("raw_node_accuracy",0.0)); raw=raw if np.isfinite(raw) else 0.0
    return (float(failures>0),float(failures),score)


def assess_autoencoder_gate(metrics:Mapping[str,Any],encoder_update:float,eligible_checkpoint_selected:bool,
                            selection_score_improved:bool=True)->dict[str,bool]:
    checks=ae_core_checks(metrics)
    checks.update({"encoder_parameters_updated":bool(np.isfinite(encoder_update) and encoder_update>0),
                   "eligible_checkpoint_selected":bool(eligible_checkpoint_selected)})
    return checks


AE_LOG_KEYS=("reconstruction_fidelity","aligned_l2","h1","persistent_node_accuracy","orthogonality_max",
             "latent_variance","effective_rank","active_covariance_rank","dominant_covariance_fraction")
def compact_autoencoder_log(metrics:Mapping[str,Any])->dict[str,Any]:
    return {key:metrics[key] for key in AE_LOG_KEYS if key in metrics}


ae_checkpoint_path=OUT/"solution_autoencoder.pt"
ae_resume_payload=None if os.environ.get("PSI_JEPA_RESTART","0")=="1" else load_torch_checkpoint(ae_checkpoint_path)
if ae_resume_payload is not None:
    ae_resume_expected={"config_hash":CONFIG_HASH,"code_hash":CODE_HASH,
                        "dataset_manifest_hash":DATASET_MANIFEST_HASH,"split_manifest_hash":SPLIT_MANIFEST_HASH,
                        "normalization_hash":NORMALIZATION_HASH,"architecture_version":ARCHITECTURE_VERSION,
                        "solver_version":SOLVER_VERSION,"source_sha256":SOURCE_SHA256,
                        "experiment_name":EXPERIMENT_NAME,"output_directory":str(OUT),
                        "stage":"B_solution_autoencoder","checkpoint_role":"solution_autoencoder_best"}
    ae_resume_mismatches={key:(ae_resume_payload.get(key),value) for key,value in ae_resume_expected.items()
                          if ae_resume_payload.get(key)!=value}
    checkpoint_state=ae_resume_payload.get("model_state")
    current_state=model.state_dict()
    state_compatible=(isinstance(checkpoint_state,Mapping) and set(checkpoint_state)==set(current_state) and
                      all(torch.is_tensor(checkpoint_state[name]) and
                          checkpoint_state[name].shape==current_state[name].shape
                          for name in current_state))
    gate_compatible=(ae_resume_payload.get("autoencoder_gate_thresholds")==AE_GATES and
                     ae_resume_payload.get("node_policy")==AE_NODE_POLICY and
                     isinstance(ae_resume_payload.get("autoencoder_gate_checks"),Mapping) and
                     all(bool(value) for value in ae_resume_payload["autoencoder_gate_checks"].values()))
    required_payload_fields=("optimizer_state","validation_metrics","epoch",
                             "solution_online_encoder_state","solution_target_encoder_state")
    payload_complete=all(field in ae_resume_payload for field in required_payload_fields)
    if ae_resume_mismatches or not state_compatible or not gate_compatible or not payload_complete:
        print("Ignoring incompatible solution-autoencoder checkpoint",
              {"identity_mismatches":ae_resume_mismatches,"state_compatible":state_compatible,
               "gate_compatible":gate_compatible,"payload_complete":payload_complete})
        ae_resume_payload=None


ae_initial_validation=None; ae_best_validation=None; ae_best_state=None; ae_best_optimizer_state=None; ae_best_epoch=-1; ae_best_key=None; ae_epoch_metrics=[]
if ae_resume_payload is None:
    ae_initial_validation=autoencoder_metrics(model,loaders["val"])
    print("AE validation before optimization",compact_autoencoder_log(ae_initial_validation))
for epoch in range(0 if ae_resume_payload is not None else CFG.preset.ae_epochs):
    loaders["train"].dataset.set_epoch(epoch)
    model.solution_online_encoder.train(); model.solution_reconstruction_decoder.train(); ae_epoch_losses=[]
    for raw in loaders["train"]:
        b=to_device(raw); ae_optimizer.zero_grad(set_to_none=True); z,recon=model.encode_online(b); aligned,_=globally_align_torch(recon,b["psi"],b["w"])
        pointwise=torch.sum((aligned-b["psi"]).square()*b["node_mask"][:,None],dim=-1)/b["node_mask"].sum(-1)[:,None].clamp_min(1)
        amplitude=b["psi"].abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)
        relative_pointwise=torch.sum(((aligned-b["psi"])/(b["psi"].abs()+.02*amplitude)).square()*b["node_mask"][:,None],dim=-1)/b["node_mask"].sum(-1)[:,None].clamp_min(1)
        identity_target=torch.eye(CFG.spectral_modes,CFG.latent_dim,device=DEVICE,dtype=model.solution_reconstruction_decoder.head.weight.dtype)
        decoder_regularizer=(model.solution_reconstruction_decoder.head.weight-identity_target).square().mean()+model.solution_reconstruction_decoder.head.bias.square().mean()
        derivative_error,derivative_target=dimensionless_derivative_errors(aligned,b["psi"],b)
        loss=(masked_state_mean(torch.sum(b["w"][:,None]*(aligned-b["psi"]).square(),dim=-1),b["state_mask"])
              +.10*masked_state_mean(derivative_error/(1+derivative_target).clamp_min(1e-8),b["state_mask"])
              +.2*masked_state_mean(pointwise,b["state_mask"])+.02*masked_state_mean(relative_pointwise,b["state_mask"])+.5*decoder_regularizer)
        if not bool(torch.isfinite(loss).detach()): raise FloatingPointError(f"non-finite AE loss at epoch {epoch}")
        loss.backward(); nn.utils.clip_grad_norm_(list(model.solution_online_encoder.parameters())+list(model.solution_reconstruction_decoder.parameters()),CFG.gradient_clip_norm,error_if_nonfinite=True); ae_optimizer.step(); ae_epoch_losses.append(float(loss.detach()))
        with torch.no_grad(): model.solution_reconstruction_decoder.head.bias.zero_()
    candidate=autoencoder_metrics(model,loaders["val"]); candidate["training_loss_mean"]=float(np.mean(ae_epoch_losses)); candidate_key=ae_candidate_key(candidate); ae_epoch_metrics.append({"epoch":epoch,"eligible":not bool(candidate_key[0]),"selection_score":ae_selection_score(candidate),**candidate})
    print("AE epoch",epoch,{**compact_autoencoder_log(candidate),"training_loss_mean":candidate["training_loss_mean"],
                            "core_gate_failures":[name for name,passed in ae_core_checks(candidate).items() if not passed]})
    if ae_best_key is None or candidate_key<ae_best_key:
        ae_best_validation=dict(candidate); ae_best_epoch=epoch; ae_best_key=candidate_key
        ae_best_state=cpu_snapshot({"encoder":model.solution_online_encoder.state_dict(),"decoder":model.solution_reconstruction_decoder.state_dict()})
        ae_best_optimizer_state=cpu_snapshot(ae_optimizer.state_dict())
if ae_resume_payload is not None:
    model.load_state_dict(ae_resume_payload["model_state"],strict=True)
    ae_validation=dict(ae_resume_payload["validation_metrics"])
    ae_best_epoch=int(ae_resume_payload["epoch"])
    ae_best_optimizer_state=ae_resume_payload["optimizer_state"]
    ae_gate_checks=dict(ae_resume_payload["autoencoder_gate_checks"])
    ae_gate_passed=all(bool(value) for value in ae_gate_checks.values())
    _assert(ae_gate_passed and all(ae_core_checks(ae_validation).values()),
            "compatible solution-autoencoder checkpoint no longer passes its scientific gate")
    history_path=OUT/"solution_autoencoder_history.json"
    if history_path.exists():
        try: ae_epoch_metrics=json.loads(history_path.read_text())
        except Exception: ae_epoch_metrics=[]
    print("RESUMING FROM SOLUTION AUTOENCODER",{"path":str(ae_checkpoint_path),"selected_epoch":ae_best_epoch,
                                                  **compact_autoencoder_log(ae_validation)})
else:
    _assert(ae_initial_validation is not None,"fresh autoencoder training requires its initial validation")
    _assert(ae_best_state is not None and ae_best_optimizer_state is not None and ae_best_epoch>=0,"no optimized autoencoder epoch was selected")
    model.solution_online_encoder.load_state_dict(ae_best_state["encoder"]); model.solution_reconstruction_decoder.load_state_dict(ae_best_state["decoder"])
    ae_validation=autoencoder_metrics(model,loaders["val"]); ae_validation["selected_epoch"]=ae_best_epoch
    ae_validation["encoder_parameter_update_l2"]=math.sqrt(sum(float((p.detach().cpu()-ae_initial_encoder_state[name]).square().sum()) for name,p in model.solution_online_encoder.named_parameters()))
    ae_validation["selection_score"]=ae_selection_score(ae_validation); ae_validation["initial_selection_score"]=ae_selection_score(ae_initial_validation)
    ae_validation["selection_score_improved_over_initialization"]=bool(ae_validation["selection_score"]<ae_validation["initial_selection_score"])
    ae_gate_checks=assess_autoencoder_gate(ae_validation,ae_validation["encoder_parameter_update_l2"],bool(ae_best_key is not None and ae_best_key[0]==0),
                                           ae_validation["selection_score_improved_over_initialization"])
    ae_gate_passed=all(ae_gate_checks.values())
    if not ae_gate_passed:
        failed={name:value for name,value in ae_gate_checks.items() if not value}
        raise RuntimeError(f"solution representation gate failed checks={failed}; metrics={ae_validation}; policy={AE_NODE_POLICY}")
    model.solution_target_encoder=initialize_ema_target(model.solution_online_encoder).to(DEVICE)
    AE_MODEL_SNAPSHOT=cpu_snapshot(model.state_dict())
    ae_payload_early={"model_state":AE_MODEL_SNAPSHOT,"optimizer_state":ae_best_optimizer_state,"scheduler_state":{},
                      "solution_online_encoder_state":cpu_snapshot(model.solution_online_encoder.state_dict()),"solution_target_encoder_state":cpu_snapshot(model.solution_target_encoder.state_dict()),
                      "configuration":CONFIG_DICT,"normalization_statistics":NORMALIZATION_METADATA,"normalization_hash":NORMALIZATION_HASH,
                      "split_manifest_hash":SPLIT_MANIFEST_HASH,"dataset_manifest_hash":DATASET_MANIFEST_HASH,"solver_version":SOLVER_VERSION,
                      "architecture_version":ARCHITECTURE_VERSION,"code_hash":CODE_HASH,"config_hash":CONFIG_HASH,"experiment_name":EXPERIMENT_NAME,
                      "output_directory":str(OUT),"epoch":ae_best_epoch,"validation_metrics":ae_validation,"ema_tau":CFG.ema_tau_start,
                      "encoder_parameter_update_l2":ae_validation["encoder_parameter_update_l2"],"dominant_covariance_fraction":ae_validation["dominant_covariance_fraction"],
                      "autoencoder_gate_thresholds":AE_GATES,"autoencoder_gate_checks":ae_gate_checks,"node_policy":AE_NODE_POLICY,
                      "stage":"B_solution_autoencoder","checkpoint_role":"solution_autoencoder_best",
                      "state_origin":{"stage":"B_solution_autoencoder","stage_epoch":ae_best_epoch},
                      "source_sha256":SOURCE_SHA256,"source_revision":SOURCE_REVISION}
    atomic_torch_save(ae_payload_early,ae_checkpoint_path)
    (OUT/"solution_autoencoder_history.json").write_text(json.dumps(ae_epoch_metrics,indent=2,default=float))
    print("SOLUTION AUTOENCODER GATE PASSED",compact_autoencoder_log(ae_validation))
    del AE_MODEL_SNAPSHOT, ae_payload_early
del ae_resume_payload, ae_best_state, ae_best_optimizer_state, ae_optimizer, ae_initial_encoder_state
if DEVICE.type=="cuda": torch.cuda.empty_cache()


def trainable_without_target(m:nn.Module)->list[nn.Parameter]:
    return [p for name,p in m.named_parameters() if p.requires_grad and not name.startswith("solution_target_encoder.")]


def validation_summary(m:TrueSchrodingerJEPA,loader:DataLoader,active_states:int|None=None)->dict[str,float]:
    """Raw validation diagnostics; physics terms are never hidden by training ramps."""
    m.eval(); totals={}; count=0
    with torch.no_grad():
        for raw in loader:
            b=to_device(raw); out=m.forward_training(b,detach_ritz_vectors=True)
            _,terms=compute_losses(out,b,DEFAULT_LOSS_WEIGHTS,loss_model=m,active_states=active_states)
            weight=len(raw["operator_records"]); count+=weight
            for key,value in terms.items():
                if value.ndim==0: totals[key]=totals.get(key,0.0)+weight*float(value)
    return {key:value/max(count,1) for key,value in totals.items()}


stage_definitions=build_curriculum_stage_definitions()
_assert([(stage["name"],stage["active_states"]) for stage in stage_definitions[1:4]]==
        [("E_subspace_3",3),("E_subspace_6",6),("E_supervised_11",CFG.k_states)],"progressive state curriculum is malformed")


def capture_training_rng_state(loader:DataLoader)->dict[str,Any]:
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state(),
           "loader_generator":loader.generator.get_state() if getattr(loader,"generator",None) is not None else None}
    if torch.cuda.is_available(): state["torch_cuda"]=torch.cuda.get_rng_state_all()
    return cpu_snapshot(state)


def batch_node_accuracy(psi:torch.Tensor,batch:Mapping[str,torch.Tensor],active_states:int|None=None)->float:
    correct=[]; arrays=psi.detach().cpu().numpy()
    active=CFG.k_states if active_states is None else active_states
    for index,n in enumerate(batch["lengths"].cpu().tolist()):
        valid=[state for state in torch.where(batch["state_mask"][index])[0].cpu().tolist() if state<active]
        correct.extend(persistent_nodes(arrays[index,state,:n],1e-8)==state for state in valid)
    return float(np.mean(correct)) if correct else float("nan")


curriculum_gradient_groups=curriculum_module_parameter_groups(model)
all_trainable=trainable_without_target(model)
module_lr_multipliers=dict(CURRICULUM_MODULE_LR_MULTIPLIERS)
optimizer=torch.optim.AdamW([{"params":parameters,"lr":CFG.learning_rate*module_lr_multipliers[name],
                               "lr_multiplier":module_lr_multipliers[name],"module_name":name}
                              for name,parameters in curriculum_gradient_groups.items()],weight_decay=CFG.weight_decay,foreach=False)
updates_per_epoch=max(1,math.ceil(len(loaders["train"])/TRAINING_GRADIENT_ACCUMULATION))
total_steps=sum(stage["epochs"]*updates_per_epoch for stage in stage_definitions)
training_history=[]; gradient_update_records=[]; global_step=0; global_epoch=0; best_score=float("inf"); best_state=None
previous_val=validation_summary(model,loaders["val"]); tau=CFG.ema_tau_start; scheduler_state={}
physics_curriculum=initial_physics_curriculum(); ritz_ready_streak=0; LITERAL_LAST_BUNDLE=None
physics_loss_keys=("residual","rayleigh_energy_exact","energy_rayleigh_consistency",
                   "rayleigh","normalization","boundary","tail")
full_checkpoint_stages={"E_supervised_11","F_physics_ramp","G_joint_finetune"}
resume_path=OUT/"training_resume.pt"
resume_payload=None if os.environ.get("PSI_JEPA_RESTART","0")=="1" else load_torch_checkpoint(resume_path)
resume_stage_index=0; resume_epoch=0; resume_stage_state=None
if resume_payload is not None:
    resume_expected={"config_hash":CONFIG_HASH,"code_hash":CODE_HASH,"dataset_manifest_hash":DATASET_MANIFEST_HASH,
                     "split_manifest_hash":SPLIT_MANIFEST_HASH,"normalization_hash":NORMALIZATION_HASH}
    resume_mismatches={key:(resume_payload.get(key),value) for key,value in resume_expected.items()
                       if resume_payload.get(key)!=value}
    prior_microbatch=resume_payload.get("training_microbatch_size")
    prior_accumulation=resume_payload.get("training_gradient_accumulation")
    runtime_batch_compatible=(
        isinstance(prior_microbatch,int) and isinstance(prior_accumulation,int) and
        prior_microbatch*prior_accumulation==
        TRAINING_MICROBATCH_SIZE*TRAINING_GRADIENT_ACCUMULATION
    )
    if not runtime_batch_compatible:
        resume_mismatches["effective_training_batch"]=(
            None if not isinstance(prior_microbatch,int) or not isinstance(prior_accumulation,int)
            else prior_microbatch*prior_accumulation,
            TRAINING_MICROBATCH_SIZE*TRAINING_GRADIENT_ACCUMULATION,
        )
    if resume_mismatches:
        print("Ignoring incompatible training checkpoint",resume_mismatches); resume_payload=None
if resume_payload is not None:
    model.load_state_dict(resume_payload["model_state"]); optimizer.load_state_dict(resume_payload["optimizer_state"])
    training_history=resume_payload["training_history"]; gradient_update_records=resume_payload["gradient_update_records"]
    global_step=int(resume_payload["global_step"]); global_epoch=int(resume_payload["global_epoch"])
    tau=float(resume_payload["tau"]); scheduler_state=resume_payload["scheduler_state"]
    physics_curriculum=resume_payload["physics_curriculum"]; ritz_ready_streak=int(resume_payload["ritz_ready_streak"])
    previous_val=resume_payload["previous_val"]; LITERAL_LAST_BUNDLE=resume_payload.get("literal_last_bundle")
    resume_stage_index=int(resume_payload["stage_index"]); resume_epoch=int(resume_payload["next_epoch"])
    resume_stage_state=resume_payload.get("stage_state")
    best_bundle=resume_payload.get("best_bundle")
    if best_bundle is not None:
        best_score=best_bundle["score"]; best_state=best_bundle["state"]; best_stage=best_bundle["stage"]; best_epoch=best_bundle["epoch"]
        best_global_step=best_bundle["global_step"]; best_global_epoch=best_bundle["global_epoch"]; best_stage_update=best_bundle["stage_update"]
        best_optimizer=best_bundle["optimizer"]; best_scheduler=best_bundle["scheduler"]; best_val=best_bundle["validation"]
        best_tau=best_bundle["tau"]; best_training_state=best_bundle["training_state"]
    rng=resume_payload.get("rng_state")
    if rng is not None:
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in rng: torch.cuda.set_rng_state_all(rng["torch_cuda"])
        if rng.get("loader_generator") is not None: loaders["train"].generator.set_state(rng["loader_generator"])
    print("RESUMING TRAINING",{"stage_index":resume_stage_index,"next_epoch":resume_epoch,
                               "global_step":global_step,"global_epoch":global_epoch,
                               "checkpoint_microbatch_size":prior_microbatch,
                               "training_microbatch_size":TRAINING_MICROBATCH_SIZE,
                               "training_gradient_accumulation":TRAINING_GRADIENT_ACCUMULATION})
for stage_index,stage_spec in enumerate(stage_definitions):
    if stage_index<resume_stage_index: continue
    stage=stage_spec["name"]; epochs=stage_spec["epochs"]; base_weights=stage_spec["weights"]; stage_lr=stage_spec["lr"]; active_states=stage_spec["active_states"]
    stage_updates=max(1,epochs*updates_per_epoch); stage_update=0; stage_best=float("inf"); stage_bad_epochs=0
    stage_best_snapshot=None; stage_best_optimizer=None; stage_best_curriculum=None; stage_best_ritz_streak=0; stage_best_validation=None; stage_best_scheduler=None; stage_best_tau=tau
    min_epochs=max(1,math.ceil((.6 if stage=="D_jepa_latent" else .5)*epochs)); patience=min(CFG.early_stopping_patience,max(2,math.ceil(.35*epochs)))
    epoch_start=0
    if stage_index==resume_stage_index and resume_stage_state is not None:
        stage_update=resume_stage_state["stage_update"]; stage_best=resume_stage_state["stage_best"]; stage_bad_epochs=resume_stage_state["stage_bad_epochs"]
        stage_best_snapshot=resume_stage_state["stage_best_snapshot"]; stage_best_optimizer=resume_stage_state["stage_best_optimizer"]
        stage_best_curriculum=resume_stage_state["stage_best_curriculum"]; stage_best_ritz_streak=resume_stage_state["stage_best_ritz_streak"]
        stage_best_validation=resume_stage_state["stage_best_validation"]; stage_best_scheduler=resume_stage_state["stage_best_scheduler"]
        stage_best_tau=resume_stage_state["stage_best_tau"]; epoch_start=resume_epoch
        val=resume_stage_state["last_validation"]; epoch=resume_stage_state["last_epoch"]
        resume_stage_was_complete=bool(resume_stage_state.get("stage_should_stop",False) or epoch_start>=epochs)
    else:
        resume_stage_was_complete=False
    if resume_stage_was_complete: epoch_start=epochs
    for epoch in range(epoch_start,epochs):
        epoch_started=time.perf_counter(); loaders["train"].dataset.set_epoch(global_epoch); global_epoch+=1
        model.train(); model.solution_target_encoder.eval(); epoch_terms=[]; epoch_losses=[]; epoch_clip_reports=[]; optimizer_updates=0
        physics_ramps=physics_state_ramps(physics_curriculum,active_states).to(DEVICE)
        applied_physics_enabled=[bool(value) for value in physics_curriculum["enabled"]]
        applied_physics_ramps=[float(value) for value in physics_ramps.detach().cpu().tolist()]
        physics_authorized=any(applied_physics_enabled[:active_states]); physics_ramp=max(applied_physics_ramps)
        detach_ritz_vectors=not (stage_spec["allow_ritz_gradient"] and ritz_ready_streak>=CFG.ritz_gradient_authorization_patience)
        weights=dict(base_weights)
        microbatches=len(loaders["train"]); optimizer.zero_grad(set_to_none=True)
        for micro_index,raw in enumerate(loaders["train"]):
            window_start=(micro_index//TRAINING_GRADIENT_ACCUMULATION)*TRAINING_GRADIENT_ACCUMULATION
            window_size=min(TRAINING_GRADIENT_ACCUMULATION,microbatches-window_start)
            b=to_device(raw); out=model.forward_training(b,detach_ritz_vectors=detach_ritz_vectors)
            loss,terms=compute_losses(out,b,weights,loss_model=model,active_states=active_states,physics_state_weights=physics_ramps)
            if not bool(torch.isfinite(loss).detach()): raise FloatingPointError(f"non-finite loss in {stage} epoch {epoch}")
            (loss/window_size).backward(); epoch_losses.append(float(loss.detach())); epoch_terms.append({key:float(value.detach()) for key,value in terms.items() if value.ndim==0})
            update_boundary=((micro_index+1)%TRAINING_GRADIENT_ACCUMULATION==0 or micro_index+1==microbatches)
            if not update_boundary: continue
            warmup=max(1,min(updates_per_epoch,stage_updates//10)); progress=max(0,stage_update-warmup)/max(stage_updates-warmup-1,1)
            multiplier=(stage_update+1)/warmup if stage_update<warmup else .5*(1+math.cos(math.pi*min(progress,1.0)))
            for group in optimizer.param_groups: group["lr"]=stage_lr*multiplier*float(group["lr_multiplier"])
            clip_report=clip_module_gradients(curriculum_gradient_groups); epoch_clip_reports.append(clip_report)
            gradient_update_records.append({"stage":stage,"pre":clip_report["gradient_norm"],"post":clip_report["gradient_norm_post_clip"],
                                            "any_clipped":int(clip_report["any_clipped"]),"clipped_module_events":clip_report["clipped_module_events"],
                                            "active_module_events":clip_report["active_module_events"]})
            optimizer.step(); optimizer.zero_grad(set_to_none=True); tau=ema_tau(global_step,total_steps); model.update_target(tau)
            global_step+=1; stage_update+=1; optimizer_updates+=1
        val=validation_summary(model,loaders["val"],active_states=active_states); previous_val=val
        score=(1-val["fidelity"])+.10*val["h1"]+.05*val["energy"]
        monitor=(val["jepa"] if stage=="D_jepa_latent" else
                  val["subspace_projector"]+.05*val["energy"] if active_states<CFG.k_states else score)
        stage_has_physics=any(float(weights.get(key,0))>0 for key in physics_loss_keys)
        if stage_has_physics: physics_curriculum=update_physics_curriculum(physics_curriculum,val,active_states)
        if stage in full_checkpoint_stages and ritz_gradient_validation_ready(val):
            ritz_ready_streak+=1
        elif stage in full_checkpoint_stages: ritz_ready_streak=0
        active_mask=curriculum_state_mask(b["state_mask"],active_states); latent_epoch=latent_diagnostics(out["z_pred"],active_mask)
        train_means=pd.DataFrame(epoch_terms).mean().to_dict()
        clipped_updates=sum(int(report["any_clipped"]) for report in epoch_clip_reports)
        clipped_module_events=sum(int(report["clipped_module_events"]) for report in epoch_clip_reports)
        active_module_events=sum(int(report["active_module_events"]) for report in epoch_clip_reports)
        module_gradient_means={}; module_gradient_post_means={}; module_caps={}; module_clip_fractions={}; module_active_updates={}
        for name in curriculum_gradient_groups:
            active_reports=[report[name] for report in epoch_clip_reports if report[name]["active"]]
            module_gradient_means[name]=float(np.mean([report["pre"] for report in active_reports])) if active_reports else 0.0
            module_gradient_post_means[name]=float(np.mean([report["post"] for report in active_reports])) if active_reports else 0.0
            module_caps[name]=float(epoch_clip_reports[0][name]["cap"]); module_active_updates[name]=len(active_reports)
            module_clip_fractions[name]=sum(int(report["clipped"]) for report in active_reports)/max(len(active_reports),1)
        scheduler_state={"type":"manual_warmup_cosine","stage":stage,"stage_update":stage_update,"stage_updates":stage_updates,"base_lr":stage_lr,
                         "lr_multipliers":module_lr_multipliers,"last_multiplier":multiplier}
        row={"stage":stage,"epoch":epoch,"global_step":global_step,"optimizer_updates":optimizer_updates,"microbatches":microbatches,
             "global_epoch":global_epoch,"active_states":active_states,"train_loss":float(np.mean(epoch_losses)),
             "gradient_norm":float(np.mean([report["gradient_norm"] for report in epoch_clip_reports])),
             "gradient_norm_post_clip":float(np.mean([report["gradient_norm_post_clip"] for report in epoch_clip_reports])),
             "clipped_updates":clipped_updates,"clipped_module_events":clipped_module_events,"active_module_events":active_module_events,
             "clip_fraction":clipped_updates/max(optimizer_updates,1),"module_clip_fraction":clipped_module_events/max(active_module_events,1),
             "clip_target_met":clipped_updates/max(optimizer_updates,1)<=CFG.gradient_clip_target_fraction,
             "learning_rate":optimizer.param_groups[0]["lr"],"ema_tau":tau,"physics_authorized":physics_authorized,"physics_ramp":physics_ramp,
             "ritz_gradient_enabled":not detach_ritz_vectors,"ritz_ready_streak":ritz_ready_streak,"validation_score":score,
             "epoch_seconds":time.perf_counter()-epoch_started,"train_latent_effective_rank":latent_epoch["effective_rank"],
             "train_node_variance":float(out["z_pred"][active_mask].var(dim=0).mean().detach()),
             "train_pre_orth_node_accuracy":batch_node_accuracy(out["psi_pre_orth"],b,active_states),"train_post_orth_node_accuracy":batch_node_accuracy(out["psi"],b,active_states),
             **{"grad_"+k:v for k,v in module_gradient_means.items()},**{"grad_post_"+k:v for k,v in module_gradient_post_means.items()},
             **{"grad_cap_"+k:v for k,v in module_caps.items()},**{"clip_"+k:v for k,v in module_clip_fractions.items()},
             **{"active_updates_"+k:v for k,v in module_active_updates.items()},
             **{f"physics_enabled_s{index}":applied_physics_enabled[index] for index in range(CFG.k_states)},
             **{f"physics_ramp_s{index}":applied_physics_ramps[index] for index in range(CFG.k_states)},
             **{f"next_physics_enabled_s{index}":bool(physics_curriculum["enabled"][index]) for index in range(CFG.k_states)},
             **{f"next_physics_ramp_s{index}":float(physics_state_ramps(physics_curriculum,active_states)[index]) for index in range(CFG.k_states)},
             **{"train_"+k:v for k,v in train_means.items()},
             **{"weight_"+k:float(v) for k,v in weights.items()},**{"val_"+k:v for k,v in val.items()}}
        training_history.append(row); print({k:(round(v,5) if isinstance(v,float) else v) for k,v in row.items() if k in ("stage","epoch","train_loss","clip_fraction","physics_ramp","val_fidelity","validation_score")})
        if stage in full_checkpoint_stages and score<best_score-CFG.early_stopping_min_delta:
            best_score=score; best_state=cpu_snapshot(model.state_dict()); best_stage=stage; best_epoch=epoch; best_global_step=global_step; best_global_epoch=global_epoch; best_stage_update=stage_update
            best_optimizer=cpu_snapshot(optimizer.state_dict()); best_scheduler=cpu_snapshot(scheduler_state); best_val=dict(val); best_tau=tau
            best_training_state={"physics_curriculum":cpu_snapshot(physics_curriculum),"ritz_ready_streak":ritz_ready_streak,"active_states":active_states}
        if monitor<stage_best-CFG.early_stopping_min_delta:
            stage_best=monitor; stage_bad_epochs=0; stage_best_snapshot=cpu_snapshot(model.state_dict()); stage_best_optimizer=cpu_snapshot(optimizer.state_dict())
            stage_best_curriculum=cpu_snapshot(physics_curriculum); stage_best_ritz_streak=ritz_ready_streak; stage_best_validation=dict(val)
            stage_best_scheduler=cpu_snapshot(scheduler_state); stage_best_tau=tau
        else: stage_bad_epochs+=1
        best_bundle=None if best_state is None else {
            "score":best_score,"state":best_state,"stage":best_stage,"epoch":best_epoch,
            "global_step":best_global_step,"global_epoch":best_global_epoch,"stage_update":best_stage_update,
            "optimizer":best_optimizer,"scheduler":best_scheduler,"validation":best_val,
            "tau":best_tau,"training_state":best_training_state}
        stage_should_stop=bool(epoch+1>=min_epochs and stage_bad_epochs>=patience)
        resume_snapshot={
            "checkpoint_role":"training_resume","config_hash":CONFIG_HASH,"code_hash":CODE_HASH,
            "dataset_manifest_hash":DATASET_MANIFEST_HASH,"split_manifest_hash":SPLIT_MANIFEST_HASH,
            "normalization_hash":NORMALIZATION_HASH,"training_microbatch_size":TRAINING_MICROBATCH_SIZE,
            "training_gradient_accumulation":TRAINING_GRADIENT_ACCUMULATION,
            "stage_index":stage_index,"next_epoch":epoch+1,
            "global_step":global_step,"global_epoch":global_epoch,"tau":tau,
            "model_state":cpu_snapshot(model.state_dict()),"optimizer_state":cpu_snapshot(optimizer.state_dict()),
            "scheduler_state":cpu_snapshot(scheduler_state),"physics_curriculum":cpu_snapshot(physics_curriculum),
            "ritz_ready_streak":ritz_ready_streak,"previous_val":dict(previous_val),
            "training_history":cpu_snapshot(training_history),"gradient_update_records":cpu_snapshot(gradient_update_records),
            "literal_last_bundle":cpu_snapshot(LITERAL_LAST_BUNDLE),"best_bundle":cpu_snapshot(best_bundle),
            "rng_state":capture_training_rng_state(loaders["train"]),
            "stage_state":{"stage_update":stage_update,"stage_best":stage_best,"stage_bad_epochs":stage_bad_epochs,
                "stage_best_snapshot":stage_best_snapshot,"stage_best_optimizer":stage_best_optimizer,
                "stage_best_curriculum":stage_best_curriculum,"stage_best_ritz_streak":stage_best_ritz_streak,
                "stage_best_validation":stage_best_validation,"stage_best_scheduler":stage_best_scheduler,
                "stage_best_tau":stage_best_tau,"last_validation":dict(val),"last_epoch":epoch,
                "stage_should_stop":stage_should_stop}}
        atomic_torch_save(resume_snapshot,resume_path)
        del resume_snapshot
        if stage_should_stop:
            print("EARLY STOP",{"stage":stage,"epoch":epoch,"best_monitor":stage_best,"patience":patience}); break
    LITERAL_LAST_BUNDLE={"model_state":cpu_snapshot(model.state_dict()),"optimizer_state":cpu_snapshot(optimizer.state_dict()),
                         "scheduler_state":cpu_snapshot(scheduler_state),"validation":dict(val),"tau":tau,"stage":stage,"stage_epoch":epoch,
                         "global_step":global_step,"global_epoch":global_epoch,"stage_update":stage_update,"history_row_index":len(training_history)-1,
                         "training_state":{"physics_curriculum":cpu_snapshot(physics_curriculum),"ritz_ready_streak":ritz_ready_streak,
                                           "active_states":active_states,"stage_best_monitor":stage_best,"stage_bad_epochs":stage_bad_epochs,
                                           "dataset_epoch":global_epoch-1,"rng_state":capture_training_rng_state(loaders["train"])}}
    if stage_best_snapshot is not None:
        model.load_state_dict(stage_best_snapshot); optimizer.load_state_dict(stage_best_optimizer); physics_curriculum=cpu_snapshot(stage_best_curriculum)
        ritz_ready_streak=stage_best_ritz_streak; scheduler_state=cpu_snapshot(stage_best_scheduler); tau=stage_best_tau; previous_val=dict(stage_best_validation)
    next_stage_payload={
        "checkpoint_role":"training_resume","config_hash":CONFIG_HASH,"code_hash":CODE_HASH,
        "dataset_manifest_hash":DATASET_MANIFEST_HASH,"split_manifest_hash":SPLIT_MANIFEST_HASH,
        "normalization_hash":NORMALIZATION_HASH,"training_microbatch_size":TRAINING_MICROBATCH_SIZE,
        "training_gradient_accumulation":TRAINING_GRADIENT_ACCUMULATION,
        "stage_index":stage_index+1,"next_epoch":0,
        "global_step":global_step,"global_epoch":global_epoch,"tau":tau,
        "model_state":cpu_snapshot(model.state_dict()),"optimizer_state":cpu_snapshot(optimizer.state_dict()),
        "scheduler_state":cpu_snapshot(scheduler_state),"physics_curriculum":cpu_snapshot(physics_curriculum),
        "ritz_ready_streak":ritz_ready_streak,"previous_val":dict(previous_val),
        "training_history":cpu_snapshot(training_history),"gradient_update_records":cpu_snapshot(gradient_update_records),
        "literal_last_bundle":cpu_snapshot(LITERAL_LAST_BUNDLE),"best_bundle":cpu_snapshot(best_bundle),
        "rng_state":capture_training_rng_state(loaders["train"]),"stage_state":None}
    atomic_torch_save(next_stage_payload,resume_path); del next_stage_payload
training_history=pd.DataFrame(training_history); training_history.to_csv(OUT/"training_history.csv",index=False)
_assert(best_state is not None,"training produced no finite validation-selected checkpoint")
_assert(LITERAL_LAST_BUNDLE is not None and LITERAL_LAST_BUNDLE["history_row_index"]==len(training_history)-1,"literal-last state was not captured before rollback")
gradient_updates=pd.DataFrame(gradient_update_records)
GRADIENT_UPDATE_AUDIT={"optimizer_updates":int(len(gradient_updates)),"clipped_updates":int(gradient_updates.any_clipped.sum()),
                       "clipped_module_events":int(gradient_updates.clipped_module_events.sum()),"active_module_events":int(gradient_updates.active_module_events.sum()),
                       "clip_fraction":float(gradient_updates.any_clipped.mean()),
                       "module_clip_fraction":float(gradient_updates.clipped_module_events.sum()/max(gradient_updates.active_module_events.sum(),1)),
                       "target_fraction":CFG.gradient_clip_target_fraction,
                       "target_met":bool(gradient_updates.any_clipped.mean()<=CFG.gradient_clip_target_fraction),
                       "pre_norm_mean":float(gradient_updates.pre.mean()),"pre_norm_p95":float(gradient_updates.pre.quantile(.95)),"pre_norm_max":float(gradient_updates.pre.max()),
                       "post_norm_mean":float(gradient_updates.post.mean()),"post_norm_p95":float(gradient_updates.post.quantile(.95)),"post_norm_max":float(gradient_updates.post.max()),
                       "by_stage":{name:{"optimizer_updates":int(len(frame)),"clip_fraction":float(frame.any_clipped.mean()),
                                           "module_clip_fraction":float(frame.clipped_module_events.sum()/max(frame.active_module_events.sum(),1))}
                                   for name,frame in gradient_updates.groupby("stage",sort=False)}}
model.load_state_dict(best_state); model.solution_target_encoder.eval()

# %% [notebook cell 63]
def state_dict_sha256(state:Mapping[str,torch.Tensor])->str:
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous(); digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def checkpoint_payload(model:TrueSchrodingerJEPA,epoch:int,validation:Mapping[str,float],optimizer_state:dict,
                       scheduler_state:dict,tau:float,stage:str,global_step_value:int,checkpoint_role:str,
                       state_origin:Mapping[str,Any],training_state:Mapping[str,Any])->dict[str,Any]:
    model_state=cpu_snapshot(model.state_dict()); origin=dict(state_origin)
    if origin.get("stage")!=stage or int(origin.get("stage_epoch",-1))!=int(epoch) or int(origin.get("global_step",-1))!=int(global_step_value):
        raise AssertionError("checkpoint state origin does not match top-level cursor")
    cursor={"stage":stage,"stage_epoch":int(epoch),"global_step":int(global_step_value),
            "global_epoch":int(origin.get("global_epoch",-1)),"stage_update":int(origin.get("stage_update",-1)),
            "history_row_index":int(origin.get("history_row_index",-1)),**cpu_snapshot(dict(training_state))}
    return {"model_state":model_state,"model_state_sha256":state_dict_sha256(model_state),
            "optimizer_state":cpu_snapshot(optimizer_state),"scheduler_state":cpu_snapshot(scheduler_state),
            "solution_online_encoder_state":cpu_snapshot(model.solution_online_encoder.state_dict()),"solution_target_encoder_state":cpu_snapshot(model.solution_target_encoder.state_dict()),
            "configuration":CONFIG_DICT,"normalization_statistics":NORMALIZATION_METADATA,"normalization_hash":NORMALIZATION_HASH,
            "split_manifest_hash":SPLIT_MANIFEST_HASH,"dataset_manifest_hash":DATASET_MANIFEST_HASH,"solver_version":SOLVER_VERSION,
            "architecture_version":ARCHITECTURE_VERSION,"source_sha256":SOURCE_SHA256,"source_revision":SOURCE_REVISION,
            "decoder_type":model.decoder_type,"primary_output":model.primary_output,"topology_safeguard":model.topology_safeguard,
            "code_hash":CODE_HASH,"config_hash":CONFIG_HASH,"experiment_name":EXPERIMENT_NAME,
            "output_directory":str(OUT),"checkpoint_role":checkpoint_role,"state_origin":origin,"training_cursor":cursor,
            "stage":stage,"stage_epoch":int(epoch),"global_step":int(global_step_value),
            "epoch":int(epoch),"validation_metrics":dict(validation),"ema_tau":float(tau)}


_assert((OUT/"solution_autoencoder.pt").exists(),"stage-B checkpoint was not saved before JEPA training")
best_history_matches=training_history.index[(training_history.stage==best_stage)&(training_history.epoch==best_epoch)&(training_history.global_step==best_global_step)].tolist()
_assert(len(best_history_matches)==1,"validation-best cursor does not identify exactly one history row")
best_origin={"stage":best_stage,"stage_epoch":best_epoch,"global_step":best_global_step,"global_epoch":best_global_epoch,
             "stage_update":best_stage_update,"history_row_index":int(best_history_matches[0]),"selection":"thresholded_full_state_validation_selection"}
best_payload=checkpoint_payload(model,best_epoch,best_val,best_optimizer,best_scheduler,best_tau,best_stage,best_global_step,
                                "validation_best",best_origin,best_training_state)
torch.save(best_payload,OUT/"best.pt")
best_model_snapshot=cpu_snapshot(model.state_dict()); model.load_state_dict(LITERAL_LAST_BUNDLE["model_state"])
last_origin={"stage":LITERAL_LAST_BUNDLE["stage"],"stage_epoch":LITERAL_LAST_BUNDLE["stage_epoch"],"global_step":LITERAL_LAST_BUNDLE["global_step"],
             "global_epoch":LITERAL_LAST_BUNDLE["global_epoch"],"stage_update":LITERAL_LAST_BUNDLE["stage_update"],
             "history_row_index":LITERAL_LAST_BUNDLE["history_row_index"],"selection":"literal_last_optimizer_update_before_stage_rollback"}
last_payload=checkpoint_payload(model,LITERAL_LAST_BUNDLE["stage_epoch"],LITERAL_LAST_BUNDLE["validation"],LITERAL_LAST_BUNDLE["optimizer_state"],
                                LITERAL_LAST_BUNDLE["scheduler_state"],LITERAL_LAST_BUNDLE["tau"],LITERAL_LAST_BUNDLE["stage"],LITERAL_LAST_BUNDLE["global_step"],
                                "literal_last",last_origin,LITERAL_LAST_BUNDLE["training_state"])
torch.save(last_payload,OUT/"last.pt")
model.load_state_dict(best_model_snapshot)
CHECKPOINT_SUMMARY={"best":{"role":"validation_best",**best_origin,"model_state_sha256":best_payload["model_state_sha256"]},
                    "last":{"role":"literal_last",**last_origin,"model_state_sha256":last_payload["model_state_sha256"]}}
autoencoder_manifest_payload=torch.load(OUT/"solution_autoencoder.pt",map_location="cpu",weights_only=False)
autoencoder_model_hash=state_dict_sha256(autoencoder_manifest_payload["model_state"])
checkpoint_manifest={"normalization_hash":NORMALIZATION_HASH,"checkpoints":{
    "best.pt":{"role":"validation_best","file_sha256":file_sha256(OUT/"best.pt"),"model_state_sha256":best_payload["model_state_sha256"],"state_origin":best_origin},
    "last.pt":{"role":"literal_last","file_sha256":file_sha256(OUT/"last.pt"),"model_state_sha256":last_payload["model_state_sha256"],"state_origin":last_origin},
    "solution_autoencoder.pt":{"role":"solution_autoencoder_best","file_sha256":file_sha256(OUT/"solution_autoencoder.pt"),
                                "model_state_sha256":autoencoder_model_hash,"state_origin":autoencoder_manifest_payload["state_origin"]}}}
(OUT/"checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest,indent=2))
del best_payload, last_payload, best_model_snapshot, best_state, best_optimizer, best_scheduler, LITERAL_LAST_BUNDLE, autoencoder_manifest_payload
if DEVICE.type=="cuda": torch.cuda.empty_cache()


def load_compatible_checkpoint(path:Path,model:TrueSchrodingerJEPA,migration_mode:bool=False,expected_role:str|None=None)->dict[str,Any]:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    expected={"config_hash":CONFIG_HASH,"code_hash":CODE_HASH,"source_sha256":SOURCE_SHA256,"dataset_manifest_hash":DATASET_MANIFEST_HASH,"split_manifest_hash":SPLIT_MANIFEST_HASH,
              "normalization_hash":NORMALIZATION_HASH,"solver_version":SOLVER_VERSION,"architecture_version":ARCHITECTURE_VERSION,
              "experiment_name":EXPERIMENT_NAME,"output_directory":str(OUT),"decoder_type":model.decoder_type,
              "primary_output":model.primary_output,"topology_safeguard":model.topology_safeguard}
    if expected_role is not None: expected["checkpoint_role"]=expected_role
    mismatches={key:(payload.get(key),value) for key,value in expected.items() if payload.get(key)!=value}
    if mismatches and not migration_mode: raise RuntimeError(f"incompatible checkpoint: {mismatches}")
    actual_state_hash=state_dict_sha256(payload["model_state"])
    if payload.get("model_state_sha256")!=actual_state_hash: raise RuntimeError("checkpoint model-state checksum mismatch")
    model.load_state_dict(payload["model_state"]); return payload


def restore_training_state(path:Path,model:TrueSchrodingerJEPA,optimizer:torch.optim.Optimizer|None=None,
                           loader:DataLoader|None=None,restore_rng:bool=False,migration_mode:bool=False)->dict[str,Any]:
    """Restore a literal-last model/optimizer and return its manual scheduler cursor."""
    payload=load_compatible_checkpoint(path,model,migration_mode=migration_mode,expected_role=None if migration_mode else "literal_last")
    if optimizer is not None: optimizer.load_state_dict(payload["optimizer_state"])
    cursor=cpu_snapshot(payload["training_cursor"])
    if restore_rng:
        rng=cursor.get("rng_state")
        if rng is None: raise RuntimeError("checkpoint lacks RNG state")
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in rng: torch.cuda.set_rng_state_all(rng["torch_cuda"])
        if loader is not None and rng.get("loader_generator") is not None: loader.generator.set_state(rng["loader_generator"])
    return {"checkpoint_role":payload["checkpoint_role"],"state_origin":payload["state_origin"],"training_cursor":cursor,
            "scheduler_state":cpu_snapshot(payload["scheduler_state"]),"ema_tau":float(payload["ema_tau"]),
            "validation_metrics":payload["validation_metrics"]}


loaded_checkpoint=load_compatible_checkpoint(OUT/"best.pt",model,expected_role="validation_best")
_assert(all(p.grad is None for p in model.solution_target_encoder.parameters()),"EMA target accumulated gradients")
print("restored compatible validation-selected checkpoint",{"stage":best_stage,"epoch":best_epoch,"score":best_score})
for temporary_name in ("optimizer", "out", "b", "loss", "terms", "raw"):
    globals().pop(temporary_name, None)
del temporary_name
gc.collect()
if DEVICE.type=="cuda": torch.cuda.empty_cache()
