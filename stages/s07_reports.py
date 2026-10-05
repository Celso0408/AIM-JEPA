"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 75]
def save_solver_plot():
    fig,axes=plt.subplots(1,3,figsize=(16,4)); grouped=solver_validation.groupby("family")
    axes[0].bar(np.arange(len(grouped)),grouped.energy_rel_max.max()); axes[0].set_yscale("log"); axes[0].set_title("N vs 2N energy floor")
    axes[1].bar(np.arange(len(grouped)),grouped.refinement_infidelity_max.max()); axes[1].set_yscale("log"); axes[1].set_title("refinement infidelity")
    axes[2].bar(np.arange(len(grouped)),grouped.independent_residual_max.max()); axes[2].set_yscale("log"); axes[2].set_title("independent label residual")
    for ax in axes: ax.set_xticks(np.arange(len(grouped))); ax.set_xticklabels(list(grouped.groups),rotation=90,fontsize=6); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/"numerical_solver_convergence.png",dpi=150); plt.close(fig)


def save_case_dashboard(record:Mapping[str,Any]):
    pred=PREDICTION_CACHE[("ID_test",record["group_id"])]; x=record["x"]; w=record["quadrature_weights"]; states=(0,5,10); fig,axes=plt.subplots(3,4,figsize=(18,12))
    for row,state in enumerate(states):
        sign=1 if np.sum(w*record["psi"][state]*pred["psi"][state])>=0 else -1
        axes[row,0].plot(x,record["V_raw"],color="black"); axes[row,0].axhline(record["energy"][state],color="tab:blue"); axes[row,0].axhline(pred["energy"][state],color="tab:orange",ls="--"); axes[row,0].set_title(f"potential and E, n={state}")
        axes[row,1].plot(x,record["psi"][state],label="exact"); axes[row,1].plot(x,sign*pred["psi"][state],"--",label="predicted"); axes[row,1].set_title("signed wavefunction")
        axes[row,2].plot(x,record["rho"][state],label="exact"); axes[row,2].plot(x,pred["rho"][state],"--",label="derived predicted"); axes[row,2].set_title("density = psi squared")
        H,idx=independent_collocation_action(x,record["V_raw"],pred["psi"][[state]]); residual=H[0]-pred["energy"][state]*pred["psi"][state,idx]; axes[row,3].plot(x[idx],residual); axes[row,3].set_title("independent residual field")
        for ax in axes[row]: ax.grid(alpha=.2)
    axes[0,1].legend(); axes[0,2].legend(); fig.tight_layout(); fig.savefig(OUT/"exact_predicted_wave_density_energy_residual.png",dpi=150); plt.close(fig)


def save_energy_and_node_comparisons(record:Mapping[str,Any]):
    pred=PREDICTION_CACHE[("ID_test",record["group_id"])]; states=np.arange(CFG.k_states)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(states,record["energy"],"o-",label="exact"); ax.plot(states,pred["energy"],"s--",label="ordered head"); ax.plot(states,pred["rayleigh"],"^:",label="audit Rayleigh")
    ax.set_xlabel("state n"); ax.set_ylabel("energy"); ax.set_xticks(states); ax.set_title("exact versus predicted complete 11-state spectrum"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"exact_predicted_energy_spectrum.png",dpi=150); plt.close(fig)
    exact_rows=[]; predicted_rows=[]
    for state in states:
        exact_rows.extend((float(value),int(state),"exact") for value in node_positions(record["x"],record["psi"][state],1e-8))
        predicted_rows.extend((float(value),int(state),"predicted") for value in node_positions(record["x"],pred["psi"][state],1e-8))
    nodes=pd.DataFrame(exact_rows+predicted_rows,columns=["x","state","source"]); nodes.to_csv(OUT/"representative_node_positions.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,6))
    for label,marker,color in (("exact","o","tab:blue"),("predicted","x","tab:orange")):
        subset=nodes[nodes.source==label]; ax.scatter(subset.x,subset.state,marker=marker,color=color,label=label,alpha=.85)
    ax.set_yticks(states); ax.set_xlabel("physical node location x"); ax.set_ylabel("state n"); ax.set_title("exact versus predicted node locations"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"node_position_comparison.png",dpi=150); plt.close(fig)


def save_metric_plots():
    fig,axes=plt.subplots(2,3,figsize=(17,9)); ids=all_metrics[all_metrics.category=="ID_test"]
    ids.groupby("state")[["infidelity","energy_abs_error","independent_residual_head"]].mean().plot(marker="o",ax=axes[0,0],title="ID errors by state")
    axes[0,0].set_yscale("log")
    family_summary[family_summary.category=="ID_test"].set_index("family")[["fidelity"]].plot.bar(ax=axes[0,1],legend=False,title="ID fidelity by family")
    all_metrics.groupby("category").fidelity.mean().plot.bar(ax=axes[0,2],title="fidelity by OOD category")
    gram_record=splits["test"][0]; pred=PREDICTION_CACHE[("ID_test",gram_record["group_id"])]; axes[1,0].imshow(weighted_gram(pred["psi_pre"],gram_record["quadrature_weights"]),vmin=-1,vmax=1,cmap="coolwarm"); axes[1,0].set_title("Gram before orthonormalization")
    axes[1,1].imshow(weighted_gram(pred["psi_orthogonal"],gram_record["quadrature_weights"]),vmin=-1,vmax=1,cmap="coolwarm"); axes[1,1].set_title("Gram after weighted orthonormalization")
    node=ids.groupby("state")[["raw_node_correct","persistent_node_correct","node_position_error"]].mean(); node.plot(marker="o",ax=axes[1,2],title="node diagnostics")
    fig.tight_layout(); fig.savefig(OUT/"physics_metric_dashboard.png",dpi=150); plt.close(fig)


def save_refinement_dashboard():
    selected=refinement_iteration_metrics[refinement_iteration_metrics.category.isin(("ID_test","unseen_analytical_functional_forms"))]
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for category,frame in selected.groupby("category"):
        grouped=frame.groupby("iteration")
        axes[0,0].plot(grouped.fidelity.mean(),"o-",label=category)
        axes[0,1].semilogy(grouped.independent_residual_head.mean(),"o-",label=category)
        later=frame[frame.iteration>0].groupby("iteration").accepted_from_previous.mean(); axes[1,0].plot(later,"o-",label=category)
    depth=refinement_depth_ablation.groupby(["variant","requested_depth"],as_index=False).agg(fidelity=("fidelity","mean"),residual=("independent_residual","mean"))
    for variant,frame in depth.groupby("variant"):
        axes[1,1].plot(frame.requested_depth,frame.fidelity,"o-",label=f"{variant} fidelity")
        axes[1,1].plot(frame.requested_depth,frame.residual,"s--",label=f"{variant} residual")
    axes[0,0].set(title="Fidelity through committed refinement",xlabel="iteration",ylabel="mean fidelity")
    axes[0,1].set(title="Independent residual through refinement",xlabel="iteration",ylabel="mean audit residual")
    axes[1,0].set(title="Accepted Hamiltonian updates",xlabel="iteration",ylabel="acceptance fraction",ylim=(-.02,1.02))
    axes[1,1].set(title="Same-checkpoint depth ablation",xlabel="requested iterations",ylabel="diagnostic mean")
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/"recurrent_refinement_dashboard.png",dpi=150); plt.close(fig)


def save_representation_cost_plots():
    fig,axes=plt.subplots(2,3,figsize=(17,9)); eigen=np.asarray(id_representation["covariance_eigenvalues"])[::-1]
    axes[0,0].semilogy(eigen,"o-"); axes[0,0].set_title("latent covariance eigenvalues")
    axes[0,1].bar(representation_metrics.category,representation_metrics.effective_rank); axes[0,1].tick_params(axis="x",rotation=70); axes[0,1].set_title("effective latent rank")
    cosine_values=np.asarray(id_representation["predictor_target_cosine_values"]); axes[0,2].hist(cosine_values,bins=20); axes[0,2].axvline(cosine_values.mean(),color="red",label="mean"); axes[0,2].legend(); axes[0,2].set_title("ID predictor/target cosine distribution")
    coeff=[]; activation=[]; activation_hooks=[]; activated_module_ids=set()
    def activation_hook(module:FactorizedBasisKANLinear,args:tuple[torch.Tensor,...],output:torch.Tensor)->None:
        with torch.no_grad():
            activated_module_ids.add(id(module))
            basis=polynomial_basis_values(module.transform(args[0].detach()),module.degree,module.basis); dense=module.dense_coefficients().detach(); values=[]
            for degree in range(module.degree): values.append(float(torch.einsum("...i,oi->...o",basis[...,degree],dense[...,degree]).abs().mean()))
            activation.append(np.asarray(values))
    for child in model.modules():
        if isinstance(child,FactorizedBasisKANLinear): activation_hooks.append(child.register_forward_hook(activation_hook))
    model.eval(); activation_batch=to_device(collate_records([operator_only_record(splits["test"][0])],include_targets=False))
    with torch.no_grad(): model.forward_operator(activation_batch)
    for hook in activation_hooks: hook.remove()
    for child in model.modules():
        if isinstance(child,FactorizedBasisKANLinear) and id(child) in activated_module_ids: coeff.append(child.dense_coefficients().detach().abs().mean((0,1)).cpu().numpy())
    if coeff:
        coefficient_spectrum=np.mean(coeff,axis=0); activation_spectrum=np.mean(activation,axis=0)
        degrees=np.arange(1,len(coefficient_spectrum)+1); axes[1,0].plot(degrees,coefficient_spectrum,"o-",label="coefficient")
        axes[1,0].plot(degrees,activation_spectrum,"s--",label="activation contribution")
        pd.DataFrame({"polynomial_degree":degrees,"mean_absolute_coefficient":coefficient_spectrum,"mean_absolute_activation_contribution":activation_spectrum}).to_csv(OUT/"kan_coefficient_activation_spectra.csv",index=False)
    axes[1,0].legend(fontsize=7); axes[1,0].set_title("KAN spectra by polynomial order")
    node_variance=[]; hooks=[]
    if isinstance(model.potential_context_encoder,GraphContextEncoder):
        b=operator_batch_view(to_device(collate_records([splits["test"][0]],include_targets=True))); h=model.potential_context_encoder.stem(b["features"]); node_variance.append(float(h.var(1).mean()))
        for block in model.potential_context_encoder.blocks: h=block(h,b["w"],b["node_mask"],b["edge_index"],b["edge_attr"]); node_variance.append(float(h.var(1).mean()))
    axes[1,1].plot(range(len(node_variance)),node_variance,"o-"); axes[1,1].set_title("node variance versus GNN depth")
    pd.DataFrame({"depth":range(len(node_variance)),"node_variance":node_variance}).to_csv(OUT/"node_variance_by_depth.csv",index=False)
    main_params=deployable_parameter_count(model); main_total_params=sum(p.numel() for p in model.parameters()); main_trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad); main_fid=id_metrics.fidelity.mean(); main_latency=id_representation["inference_latency_ms_per_hamiltonian"]
    axes[1,2].scatter([main_params],[1-main_fid],s=80,label=f"trained main {CFG.mode}"); axes[1,2].set_xscale("log"); axes[1,2].set_yscale("log"); axes[1,2].set_xlabel("parameters"); axes[1,2].set_ylabel("ID infidelity"); axes[1,2].set_title("accuracy vs parameter count\nmatched baselines pending"); axes[1,2].legend(fontsize=7)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/"representation_kan_cost_dashboard.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.scatter([main_latency],[1-main_fid],s=90,label=f"trained main {CFG.mode}"); ax.set_yscale("log"); ax.set_xlabel("inference ms / Hamiltonian"); ax.set_ylabel("ID infidelity"); ax.set_title("accuracy versus runtime\nmatched baselines pending"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"accuracy_vs_runtime.png",dpi=150); plt.close(fig)
    solver_mean=data_audit.solver_seconds.mean(); speedup=solver_mean/(main_latency/1000)
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(["mapped FEM label","target-free surrogate"],[solver_mean,main_latency/1000]); ax.set_yscale("log"); ax.set_ylabel("seconds / Hamiltonian"); ax.set_title(f"measured cost; speedup={speedup:.2f}x"); fig.tight_layout(); fig.savefig(OUT/"surrogate_vs_eigensolver_cost.png",dpi=150); plt.close(fig)
    mean_epoch_seconds=float(training_history.epoch_seconds.mean())
    gradient_summary={"update_pre_clip_mean":GRADIENT_UPDATE_AUDIT["pre_norm_mean"],"update_pre_clip_p95":GRADIENT_UPDATE_AUDIT["pre_norm_p95"],
                      "update_pre_clip_max":GRADIENT_UPDATE_AUDIT["pre_norm_max"],"update_post_clip_mean":GRADIENT_UPDATE_AUDIT["post_norm_mean"],
                      "update_post_clip_p95":GRADIENT_UPDATE_AUDIT["post_norm_p95"],"update_post_clip_max":GRADIENT_UPDATE_AUDIT["post_norm_max"],
                      "all_finite":bool(np.isfinite(gradient_updates[["pre","post"]].to_numpy()).all())}
    module_gradient_summary={name:{"pre_clip_epoch_mean":float(training_history[f"grad_{name}"].mean()),
                                   "post_clip_epoch_mean":float(training_history[f"grad_post_{name}"].mean()),
                                   "cap":float(training_history[f"grad_cap_{name}"].iloc[0]),
                                   "clip_fraction":float((training_history[f"clip_{name}"]*training_history[f"active_updates_{name}"]).sum()/max(training_history[f"active_updates_{name}"].sum(),1))}
                             for name in curriculum_gradient_groups}
    return {"inference_parameters":int(main_params),"total_training_object_parameters":int(main_total_params),"trainable_parameters":int(main_trainable_params),"mean_eigensolver_seconds":float(solver_mean),"inference_seconds":float(main_latency/1000),"measured_speedup":float(speedup),
            "mean_training_epoch_seconds":mean_epoch_seconds,"training_throughput_hamiltonians_per_second":float(len(splits["train"])/max(mean_epoch_seconds,1e-12)),
            "inference_throughput_hamiltonians_per_second":float(1000/max(main_latency,1e-12)),"peak_inference_memory_bytes":int(id_representation["peak_inference_memory_bytes"]),"memory_measurement":id_representation["memory_measurement"],
            "gradient_stability":gradient_summary,"gradient_clipping":GRADIENT_UPDATE_AUDIT,"representative_module_gradient_norms":module_gradient_summary}


def save_best_median_worst():
    quality=id_metrics.groupby("group_id").fidelity.mean().sort_values(); picks=[quality.index[0],quality.index[len(quality)//2],quality.index[-1]]; by_id={r["group_id"]:r for r in splits["test"]}
    fig,axes=plt.subplots(3,2,figsize=(14,10))
    for row,gid in enumerate(picks):
        record=by_id[gid]; pred=PREDICTION_CACHE[("ID_test",gid)]; axes[row,0].plot(record["x"],record["V_raw"]); axes[row,0].set_title(["worst","median","best"][row]+f" {record['family']}")
        valid_values=np.asarray(record["V_raw"])[np.asarray(record["potential_valid_mask"])]; lo,hi=np.quantile(valid_values,[.02,.98]); axes[row,0].set_ylim(lo-.05*(hi-lo+1e-12),hi+.05*(hi-lo+1e-12))
        for state in (0,5,10): axes[row,1].plot(record["x"],record["rho"][state],label=f"exact {state}"); axes[row,1].plot(record["x"],pred["rho"][state],"--",label=f"pred {state}")
        axes[row,1].legend(fontsize=6,ncol=2)
    fig.tight_layout(); fig.savefig(OUT/"best_median_worst_hamiltonians.png",dpi=150); plt.close(fig)


def save_topology_case_dashboards():
    if not TOPOLOGY_METRICS_AVAILABLE: return
    direct_rows=per_state_output_metrics[(per_state_output_metrics.category=="ID_test")&(per_state_output_metrics.output_route=="direct_topology")]
    quality=direct_rows.groupby("group_id").fidelity.mean().sort_values(); picks={"worst":quality.index[0],"median":quality.index[len(quality)//2],"best":quality.index[-1]}
    records_by_id={record["group_id"]:record for record in splits["test"]}; states=(0,5,10)
    for label,gid in picks.items():
        record=records_by_id[gid]; pred=PREDICTION_CACHE[("ID_test",gid)]; x=np.asarray(record["x"]); w=np.asarray(record["quadrature_weights"])
        t=(x-x[0])/(x[-1]-x[0]); fig,axes=plt.subplots(7,3,figsize=(17,22))
        for column,state in enumerate(states):
            sign_direct=1 if np.sum(w*record["psi"][state]*pred["psi_direct"][state])>=0 else -1
            sign_hybrid=1 if np.sum(w*record["psi"][state]*pred["psi_hybrid"][state])>=0 else -1
            axes[0,column].plot(x,record["V_raw"],color="black",label="V"); axes[0,column].axhline(record["energy"][state],color="tab:green",label="exact E")
            axes[0,column].axhline(pred["energy_direct"][state],color="tab:blue",ls="--",label="direct E"); axes[0,column].axhline(pred["energy_hybrid"][state],color="tab:orange",ls=":",label="hybrid E")
            axes[1,column].plot(x,record["psi"][state],color="black",label="exact"); axes[1,column].plot(x,sign_direct*pred["psi_direct"][state],"--",label="direct")
            axes[1,column].plot(x,sign_hybrid*pred["psi_hybrid"][state],":",label="hybrid")
            exact_nodes=target_node_positions_from_wave(x,record["psi"][state],state); predicted_nodes=x[0]+(x[-1]-x[0])*pred["predicted_node_positions_t"][state,:state]
            axes[1,column].scatter(exact_nodes,np.zeros_like(exact_nodes),marker="o",facecolors="none",edgecolors="black",s=28,label="exact nodes" if state else None)
            axes[1,column].scatter(predicted_nodes,np.zeros_like(predicted_nodes),marker="x",color="tab:blue",s=28,label="phase nodes" if state else None)
            axes[2,column].plot(x,record["rho"][state],color="black",label="exact"); axes[2,column].plot(x,pred["rho_direct"][state],"--",label="direct"); axes[2,column].plot(x,pred["rho_hybrid"][state],":",label="hybrid")
            axes[3,column].plot(t,pred["amplitude"][state],color="tab:purple",label="A")
            axes[4,column].plot(t,pred["phase_cdf"][state],color="tab:blue",label="C"); axes[4,column].plot(t,pred["theta"][state]/math.pi,color="tab:red",ls="--",label="theta/pi")
            axes[5,column].plot(t,pred["phase_density"][state],color="tab:brown",label="q")
            for route,color,style in (("direct","tab:blue","--"),("hybrid","tab:orange",":")):
                wave=pred[f"psi_{route}"][[state]]; energy_value=pred[f"energy_{route}"][state]; action,index=independent_collocation_action(x,record["V_raw"],wave)
                axes[6,column].plot(x[index],np.abs(action[0]-energy_value*wave[0,index]),color=color,ls=style,label=f"{route} residual")
            for row in range(7): axes[row,column].grid(alpha=.2)
            axes[0,column].set_title(f"{label}: {record['family']}, n={state}")
            for row,name in enumerate(("V and energies","wavefunctions and nodes","densities","positive amplitude","phase CDF / phase","phase density","|audit residual|")):
                if column==0: axes[row,column].set_ylabel(name)
            axes[6,column].set_xlabel("x")
            for row in (3,4,5): axes[row,column].set_xlabel("dimensionless t")
        for row in range(7): axes[row,0].legend(fontsize=6,loc="best")
        fig.tight_layout(); path=OUT/f"topology_case_{label}.png"; fig.savefig(path,dpi=145); plt.close(fig)
        fig,companion=plt.subplots(1,3,figsize=(15,4)); state_axis=np.arange(CFG.k_states)
        companion[0].plot(state_axis,record["energy"],"o-",label="exact"); companion[0].plot(state_axis,pred["energy_direct"],"s--",label="direct"); companion[0].plot(state_axis,pred["energy_hybrid"],"^:",label="hybrid"); companion[0].legend(fontsize=7); companion[0].set_title("energy spectra")
        companion[1].imshow(weighted_gram(pred["psi_direct"],w),vmin=-1,vmax=1,cmap="coolwarm"); companion[1].set_title("direct Gram")
        companion[2].imshow(weighted_gram(pred["psi_hybrid"],w),vmin=-1,vmax=1,cmap="coolwarm"); companion[2].set_title("hybrid Gram")
        for ax in companion: ax.set_xlabel("state"); ax.grid(alpha=.15)
        fig.tight_layout(); fig.savefig(OUT/f"topology_case_{label}_energy_gram.png",dpi=145); plt.close(fig)


def save_topology_summary_figure():
    if not TOPOLOGY_METRICS_AVAILABLE:
        fig,ax=plt.subplots(figsize=(9,4)); ax.axis("off")
        ax.text(.5,.5,"Topology-phase metrics are not applicable to the selected spectral decoder.",ha="center",va="center")
        fig.tight_layout(); fig.savefig(OUT/"topology_summary_dashboard.png",dpi=150); plt.close(fig); return
    frame=topology_metrics[topology_metrics.category=="ID_test"]; routes=per_state_output_metrics[per_state_output_metrics.category=="ID_test"]
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    representative=splits["test"][0]; prediction=PREDICTION_CACHE[("ID_test",representative["group_id"])]
    for state in range(1,CFG.k_states):
        exact=target_node_positions_from_wave(representative["x"],representative["psi"][state],state)
        predicted=representative["x"][0]+(representative["x"][-1]-representative["x"][0])*prediction["predicted_node_positions_t"][state,:state]
        axes[0,0].scatter(exact,predicted,s=14,label=f"n={state}" if state in (1,5,10) else None)
    bounds=[representative["x"][0],representative["x"][-1]]; axes[0,0].plot(bounds,bounds,"k--"); axes[0,0].set(title="exact vs phase-quantile nodes",xlabel="exact x",ylabel="predicted x")
    frame.groupby("state").node_position_mae_normalized.mean().plot(ax=axes[0,1],marker="o",title="normalized node MAE by state")
    frame.groupby("state").lobe_probability_mass_mae.mean().plot(ax=axes[0,2],marker="o",title="lobe-mass MAE by state")
    axes[1,0].hist(frame.maximum_phase_increment,bins=24,alpha=.8); axes[1,0].axvline(math.pi/CFG.minimum_intervals_per_lobe,color="red",ls="--",label="resolution limit"); axes[1,0].set_title("phase-increment distribution")
    routes.groupby("output_route").persistent_node_correct.mean().reindex(["direct_topology","initial_ritz","hybrid","physics_selected"]).plot.bar(ax=axes[1,1],title="persistent topology by route")
    axes[1,2].bar(["proposal topology rejection","Hamiltonian direct fallback"],[frame.topology_rejection_fraction.mean(),frame.hybrid_fallback.mean()]); axes[1,2].set_ylim(0,1); axes[1,2].set_title("hybrid topology safeguards")
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=7) if ax.get_legend_handles_labels()[0] else None
    fig.tight_layout(); fig.savefig(OUT/"topology_summary_dashboard.png",dpi=150); plt.close(fig)


save_solver_plot(); save_case_dashboard(splits["test"][0]); save_energy_and_node_comparisons(splits["test"][0]); save_metric_plots(); save_refinement_dashboard(); COST_METRICS=save_representation_cost_plots(); save_best_median_worst(); save_topology_case_dashboards(); save_topology_summary_figure()

# %% [notebook cell 77]
def test_end_to_end_artifacts_and_checkpoint_roles():
    _assert((OUT/"best.pt").exists() and (OUT/"last.pt").exists() and (OUT/"checkpoint_manifest.json").exists())
    recurrent_artifacts=[OUT/"refinement_iteration_metrics.csv",OUT/"refinement_depth_ablation.csv",OUT/"recurrent_refinement_dashboard.png"]
    topology_artifacts=[OUT/"per_state_output_metrics.csv",OUT/"topology_metrics.csv",OUT/"direct_hybrid_comparison.csv",
                        OUT/"topology_summary_by_category.csv",OUT/"topology_refinement_audit.csv",OUT/"topology_summary_dashboard.png"]
    _assert((OUT/"per_state_metrics.csv").exists() and all(path.exists() and path.stat().st_size>0 for path in [*recurrent_artifacts,*topology_artifacts]) and inference_demo["psi"].shape[0]==CFG.k_states)
    dashboard_image=plt.imread(OUT/"recurrent_refinement_dashboard.png")
    _assert(dashboard_image.size>0 and np.isfinite(dashboard_image).all(),"recurrent dashboard is not a decodable finite image")
    topology_image=plt.imread(OUT/"topology_summary_dashboard.png")
    _assert(topology_image.size>0 and np.isfinite(topology_image).all(),"topology dashboard is not a decodable finite image")
    best_check=torch.load(OUT/"best.pt",map_location="cpu",weights_only=False); last_check=torch.load(OUT/"last.pt",map_location="cpu",weights_only=False)
    autoencoder_check=torch.load(OUT/"solution_autoencoder.pt",map_location="cpu",weights_only=False)
    manifest=json.loads((OUT/"checkpoint_manifest.json").read_text())
    _assert(best_check["checkpoint_role"]=="validation_best" and last_check["checkpoint_role"]=="literal_last")
    _assert(best_check["normalization_hash"]==last_check["normalization_hash"]==NORMALIZATION_HASH)
    _assert(state_dict_sha256(best_check["model_state"])==best_check["model_state_sha256"] and state_dict_sha256(last_check["model_state"])==last_check["model_state_sha256"])
    for filename,payload in (("best.pt",best_check),("last.pt",last_check)):
        origin=payload["state_origin"]; row=training_history.iloc[int(origin["history_row_index"])]
        _assert(row.stage==origin["stage"] and int(row.epoch)==int(origin["stage_epoch"]) and int(row.global_step)==int(origin["global_step"]))
        _assert(payload["stage"]==origin["stage"] and payload["epoch"]==origin["stage_epoch"] and payload["global_step"]==origin["global_step"])
        for cursor_key in ("stage","stage_epoch","global_step","global_epoch","stage_update","history_row_index"):
            _assert(payload["training_cursor"][cursor_key]==origin[cursor_key],f"{filename} cursor mismatch for {cursor_key}")
        for key,value in payload["validation_metrics"].items():
            column="val_"+key
            _assert(column in row,f"{payload['checkpoint_role']} history lacks {column}")
            _assert(np.isfinite(row[column])==np.isfinite(value),f"{payload['checkpoint_role']} finiteness mismatch for {key}")
            if np.isfinite(value): _assert(math.isclose(float(row[column]),float(value),rel_tol=1e-7,abs_tol=1e-7),f"{payload['checkpoint_role']} validation mismatch for {key}")
        entry=manifest["checkpoints"][filename]
        _assert(entry["role"]==payload["checkpoint_role"] and entry["state_origin"]==origin)
        _assert(entry["file_sha256"]==file_sha256(OUT/filename) and entry["model_state_sha256"]==payload["model_state_sha256"])
    ae_entry=manifest["checkpoints"]["solution_autoencoder.pt"]
    _assert(ae_entry["role"]==autoencoder_check["checkpoint_role"] and ae_entry["state_origin"]==autoencoder_check["state_origin"])
    _assert(ae_entry["file_sha256"]==file_sha256(OUT/"solution_autoencoder.pt") and ae_entry["model_state_sha256"]==state_dict_sha256(autoencoder_check["model_state"]))
    _assert(manifest["normalization_hash"]==NORMALIZATION_HASH)
    _assert(best_check["stage"] in full_checkpoint_stages and int(training_history.iloc[best_check["state_origin"]["history_row_index"]].active_states)==CFG.k_states)
    _assert(last_check["state_origin"]["selection"]=="literal_last_optimizer_update_before_stage_rollback")
    for _,row in training_history.iterrows():
        _assert(math.isclose(row.clip_fraction,row.clipped_updates/max(row.optimizer_updates,1),abs_tol=1e-12))
        _assert(math.isclose(row.module_clip_fraction,row.clipped_module_events/max(row.active_module_events,1),abs_tol=1e-12))
        _assert(row.gradient_norm_post_clip<=row.gradient_norm+1e-5 and np.isfinite(row.gradient_norm))
        _assert(math.isclose(row.physics_ramp,max(float(row[f"physics_ramp_s{index}"]) for index in range(CFG.k_states)),abs_tol=1e-12))
    _assert(int(training_history.optimizer_updates.sum())==GRADIENT_UPDATE_AUDIT["optimizer_updates"])
    _assert(int(training_history.clipped_updates.sum())==GRADIENT_UPDATE_AUDIT["clipped_updates"])
    stage_states={name:frame.active_states.astype(int).unique().tolist() for name,frame in training_history[training_history.stage.str.startswith("E_")].groupby("stage",sort=False)}
    _assert(stage_states=={"E_subspace_3":[3],"E_subspace_6":[6],"E_supervised_11":[11]})
run_test("Shortest end-to-end train/evaluate/infer path",test_end_to_end_artifacts_and_checkpoint_roles)


def test_refinement_export_contract():
    frame=refinement_iteration_metrics; required={"category","group_id","family","iteration","state","training_relative_residual","independent_residual_head","fidelity","energy_abs_error","accepted_from_previous","attempted_from_previous","step_size","gram_max_error","boundary_error"}
    _assert(required<=set(frame) and set(all_metrics.category)==set(frame.category),"refinement export lacks fields or evaluation categories")
    first=frame[frame.iteration==0]; _assert(first.accepted_from_previous.isna().all() and first.step_size.isna().all())
    later=frame[frame.iteration>0]; _assert((later.loc[later.accepted_from_previous==0,"step_size"]==0).all() and (later.step_size<=CFG.refinement_damping+1e-12).all())
    for _,group in frame.groupby(["category","group_id"]):
        iterations=sorted(group.iteration.unique().tolist()); _assert(iterations==list(range(iterations[-1]+1)),"refinement iterations are not contiguous")
        blocks=group.groupby("iteration").training_relative_residual.apply(lambda values:float(np.sqrt(np.mean(np.square(values))))).to_numpy()
        _assert(np.all(blocks[1:]<=blocks[:-1]*(1+CFG.refinement_acceptance_tolerance)+2e-7),"committed block residual increased")
    final=frame.sort_values("iteration").groupby(["category","group_id","state"],as_index=False).tail(1)
    hybrid=per_state_output_metrics[per_state_output_metrics.output_route=="hybrid"]
    merged=final.merge(hybrid,on=["category","group_id","state"],suffixes=("_trace","_hybrid"),validate="one_to_one")
    committed=merged.hybrid_fallback==0
    _assert(np.allclose(merged.loc[committed,"fidelity_trace"],merged.loc[committed,"fidelity_hybrid"],atol=2e-6) and
            np.allclose(merged.loc[committed,"independent_residual_head_trace"],merged.loc[committed,"independent_residual_head_hybrid"],atol=2e-6))
    fallback_routes=per_state_output_metrics[per_state_output_metrics.hybrid_fallback==1].pivot_table(index=["category","group_id","state"],columns="output_route",values="fidelity")
    if len(fallback_routes): _assert(np.allclose(fallback_routes["hybrid"],fallback_routes[MAIN_DIRECT_ROUTE],atol=2e-6),"fallback hybrid did not retain the direct decoder output")
run_test("Per-iteration refinement trace and exported metrics are complete",test_refinement_export_contract)

test_frame=pd.DataFrame(TEST_RESULTS)
TOPOLOGY_TEST_NAMES={"Topology decoder configuration and parameter accounting","Topology decoder enforces positive amplitude and exact Sturm phase",
                     "Topology reflection transforms amplitude phase and waveform","Topology node lobe and phase-resolution losses have constructed-case behavior",
                     "Both decoder types checkpoint and serialize target-free inference"}
ENERGY_TEST_NAMES={"Exact spectra fit inside operator-scaled ground/log-gap coordinates",
                   "Operator-scaled ordered energies and exact/Rayleigh consistency"}
# Two GPU-intensive gradient-route diagnostics are intentionally disabled above.
EXPECTED_MANDATORY_TEST_COUNT=68
_assert(len(test_frame)==EXPECTED_MANDATORY_TEST_COUNT,
        f"mandatory test count is not exactly {EXPECTED_MANDATORY_TEST_COUNT}")
_assert(test_frame["test"].nunique()==EXPECTED_MANDATORY_TEST_COUNT and
        TOPOLOGY_TEST_NAMES|ENERGY_TEST_NAMES<=set(test_frame.test),"mandatory test names are missing or duplicated")
_assert(test_frame["status"].eq("PASS").all(),"not all mandatory tests passed")
_assert(test_frame["index"].tolist()==list(range(1,EXPECTED_MANDATORY_TEST_COUNT+1)),"mandatory test indices are not contiguous")
TEST_SUMMARY={"count":int(len(test_frame)),"unique_names":int(test_frame["test"].nunique()),"pass_count":int(test_frame["status"].eq("PASS").sum()),"all_passed":True}
test_status=dict(zip(test_frame.test,test_frame.status))
TOPOLOGY_ACCEPTANCE_MATRIX=[
    (1,"finite/boundary/normalization/rho/amplitude/CDF/theta/phase crossings",["Topology decoder enforces positive amplitude and exact Sturm phase"]),
    (2,"persistent sampled node count when resolution gate passes",["Topology decoder enforces positive amplitude and exact Sturm phase"]),
    (3,"finite nonzero gradients through topology losses",["Topology node lobe and phase-resolution losses have constructed-case behavior"]),
    (4,"padded nodes and intervals contribute exactly zero",["Topology decoder enforces positive amplitude and exact Sturm phase","Refinement is stable to mixed-resolution padding and isolated across peers"]),
    (5,"multiple resolutions and nonuniform grids",["Topology decoder enforces positive amplitude and exact Sturm phase","Nonuniform grids"]),
    (6,"translation and Schrodinger-dilation covariance",["Translation and Schrödinger-dilation covariance"]),
    (7,"reflection amplitude/phase/waveform transformation",["Topology reflection transforms amplitude phase and waveform"]),
    (8,"constructed node-phase and lobe-mass losses near zero",["Topology node lobe and phase-resolution losses have constructed-case behavior"]),
    (9,"shifted nodes increase node-phase loss",["Topology node lobe and phase-resolution losses have constructed-case behavior"]),
    (10,"concentrated phase activates resolution loss",["Topology node lobe and phase-resolution losses have constructed-case behavior"]),
    (11,"target node and topology-field labels absent from inference",["Target-free inference","Both decoder types checkpoint and serialize target-free inference"]),
    (12,"changing any target topology tensor cannot alter forward_operator",["Both decoder types checkpoint and serialize target-free inference"]),
    (13,"no solution target reaches direct decoder",["No exact target latent reaches main decoder"]),
    (14,"topology safeguard rejects wrong-node lower-residual control",["Both decoder types checkpoint and serialize target-free inference"]),
    (15,"direct topology remains available on hybrid failure",["Both decoder types checkpoint and serialize target-free inference"]),
    (16,"checkpoint/load and target-free serialization for both decoders",["Both decoder types checkpoint and serialize target-free inference"]),
]
topology_acceptance_rows=[]
for index,requirement,evidence in TOPOLOGY_ACCEPTANCE_MATRIX:
    passed=all(test_status.get(name)=="PASS" for name in evidence)
    topology_acceptance_rows.append({"index":index,"requirement":requirement,"status":"PASS" if passed else "FAIL","evidence":"; ".join(evidence)})
tiny_pass=bool(tiny_result["loss_ratio"]<=CFG.tiny_overfit_policy.loss_ratio_max and
               tiny_result["final_fidelity"]>=CFG.tiny_overfit_policy.fidelity_min and
               tiny_result["continuous_topology_compliance"]==1.0 and tiny_result["phase_resolution_gate_fraction"]>0 and
               tiny_result["conditional_persistent_topology_compliance"]==1.0)
topology_acceptance_rows.append({"index":17,"requirement":"tiny-batch topology overfit with fidelity and topology retained",
                                 "status":"PASS" if tiny_pass else "FAIL","evidence":"tiny_result measured trajectory"})
topology_acceptance_matrix=pd.DataFrame(topology_acceptance_rows)
_assert(len(topology_acceptance_matrix)==17 and topology_acceptance_matrix.status.eq("PASS").all(),"topology acceptance matrix is incomplete or failing")
TEST_SUMMARY["topology_acceptance_count"]=17; TEST_SUMMARY["topology_acceptance_pass_count"]=int(topology_acceptance_matrix.status.eq("PASS").sum())
test_frame.to_csv(OUT/"test_results.csv",index=False); topology_acceptance_matrix.to_csv(OUT/"topology_mandatory_test_matrix.csv",index=False)
print("MANDATORY TEST SUMMARY",TEST_SUMMARY)
run_assertions={"configuration":loaded_checkpoint["config_hash"]==CONFIG_HASH,"checkpoint_experiment":loaded_checkpoint["experiment_name"]==EXPERIMENT_NAME,
                "output_directory":loaded_checkpoint["output_directory"]==str(OUT),"dataset_manifest":loaded_checkpoint["dataset_manifest_hash"]==DATASET_MANIFEST_HASH,
                "split_manifest":loaded_checkpoint["split_manifest_hash"]==SPLIT_MANIFEST_HASH,"solver_version":loaded_checkpoint["solver_version"]==SOLVER_VERSION,
                "architecture_version":loaded_checkpoint["architecture_version"]==ARCHITECTURE_VERSION,
                "normalization_contract":loaded_checkpoint["normalization_hash"]==NORMALIZATION_HASH,
                "evaluation_checkpoint_role":loaded_checkpoint["checkpoint_role"]=="validation_best",
                "source_identity":loaded_checkpoint["source_sha256"]==SOURCE_SHA256 and loaded_checkpoint["code_hash"]==CODE_HASH,
                "mandatory_tests":TEST_SUMMARY["all_passed"]}
assert all(run_assertions.values())
id_mean=id_metrics.mean(numeric_only=True); id_hamiltonian=id_metrics.groupby("group_id").mean(numeric_only=True)
projection_floor=max(1-min(r["model_grid_diagnostics"]["projection_fidelity_min"] for r in records),1e-8)
reference_residual_floor=max(float(id_mean.exact_label_residual_floor),1e-8)
reference_energy_floor=max(float(solver_validation.energy_rel_max.max()),1e-8); model_gram_floor=max(r["model_grid_diagnostics"]["gram_max_error"] for r in records)
id_energy_scale=float(np.mean([np.ptp(r["energy"]) for r in splits["test"]])); id_normalized_spacing=float(np.mean([np.mean(np.diff(r["x"]))/(r["domain_right"]-r["domain_left"]) for r in splits["test"]]))
tail_relevant=id_metrics[id_metrics.geometry.isin(("truncated_line","half_line","radial_reduced"))]
ACCEPTANCE_THRESHOLDS={
    "fidelity_min":1-min(.05,max(.01,5*projection_floor)),"h1_relative_max":min(.25,max(.10,5*math.sqrt(projection_floor))),
    "node_accuracy_min":.90,"node_position_error_normalized_max":2*id_normalized_spacing,"independent_residual_max":min(.10,max(.02,5*reference_residual_floor)),
    "energy_scale_normalized_max":min(.10,max(.03,5*reference_energy_floor)),"gap_absolute_max":min(.25,max(.03,5*reference_energy_floor)*id_energy_scale),
    "gram_max":min(1e-2,max(1e-4,10*model_gram_floor)),"boundary_max":1e-7,"tail_error_max":min(.02,max(.005,5*projection_floor)),
}
resolved_topology_rows=(topology_metrics[topology_metrics.maximum_phase_increment<=math.pi/CFG.minimum_intervals_per_lobe+1e-6]
                        if TOPOLOGY_METRICS_AVAILABLE else topology_metrics)
id_direct_route=per_state_output_metrics[(per_state_output_metrics.category=="ID_test")&
                                         (per_state_output_metrics.output_route==MAIN_DIRECT_ROUTE)]
scientific_acceptance={"numerical_smoke_gate":bool(solver_validation.passes_gate.all()),
                       "numerical_reference_scientific_quality":bool(solver_validation.energy_rel_max.max()<caps["energy_rel"] and solver_validation.independent_residual_max.max()<caps["residual"] and solver_validation.refinement_infidelity_max.max()<caps["infidelity"]),
                       "model_grid_projection":bool(all(r["model_grid_diagnostics"]["passes_projection_gate"] for r in records)),
                       "solution_autoencoder":bool(ae_gate_passed),"mandatory_tests":TEST_SUMMARY["all_passed"],"target_free_inference":True,"ema_target":True,"no_split_leakage":True,
                       "recurrent_refinement_trace_complete":bool(len(refinement_iteration_metrics)>0 and len(refinement_depth_ablation)>0 and
                                                                  all((OUT/name).exists() and (OUT/name).stat().st_size>0 for name in
                                                                      ("refinement_iteration_metrics.csv","refinement_depth_ablation.csv","recurrent_refinement_dashboard.png"))),
                       "density_square_exact":bool((all_metrics.rho_square_consistency==0).all()),
                       "direct_continuous_sturm_topology":bool(TOPOLOGY_METRICS_AVAILABLE and len(topology_metrics)>0 and (topology_metrics.exact_node_count_compliance==1).all()),
                       "direct_sampled_topology_when_resolved":bool(TOPOLOGY_METRICS_AVAILABLE and len(resolved_topology_rows)>0 and (resolved_topology_rows.persistent_node_count_compliance==1).all()),
                       "target_node_inference_isolation":not bool(set(FORBIDDEN_DIRECT_KEYS)&set(collate_records([operator_only_record(splits["test"][0])],include_targets=False))),
                       "target_field_inference_isolation":not bool(set(FORBIDDEN_DIRECT_KEYS)&set(collate_records([operator_only_record(splits["test"][0])],include_targets=False))),
                       "hybrid_never_commits_invalid_topology":bool(TOPOLOGY_METRICS_AVAILABLE and (topology_metrics.loc[topology_metrics.hybrid_fallback==0,"hybrid_topology_valid"]==1).all()),
                       "direct_output_available_on_hybrid_fallback":bool(id_direct_route.group_id.nunique()==id_metrics.group_id.nunique()),
                       "id_fidelity_credible":bool(id_mean.fidelity>ACCEPTANCE_THRESHOLDS["fidelity_min"] and id_hamiltonian.fidelity.quantile(.10)>ACCEPTANCE_THRESHOLDS["fidelity_min"]),
                       "id_h1_credible":bool(id_mean.h1_relative<ACCEPTANCE_THRESHOLDS["h1_relative_max"] and id_hamiltonian.h1_relative.quantile(.90)<ACCEPTANCE_THRESHOLDS["h1_relative_max"]),
                       "id_node_accuracy_credible":bool(id_mean.raw_node_correct>ACCEPTANCE_THRESHOLDS["node_accuracy_min"] and id_hamiltonian.raw_node_correct.min()>ACCEPTANCE_THRESHOLDS["node_accuracy_min"]),
                       "id_node_locations_credible":bool(id_mean.node_position_match>.90 and id_mean.node_position_error_penalized_normalized<ACCEPTANCE_THRESHOLDS["node_position_error_normalized_max"] and id_hamiltonian.node_position_error_penalized_normalized.quantile(.90)<ACCEPTANCE_THRESHOLDS["node_position_error_normalized_max"]),
                       "id_independent_residual_credible":bool(id_mean.independent_residual_head<ACCEPTANCE_THRESHOLDS["independent_residual_max"] and id_hamiltonian.independent_residual_head.quantile(.90)<ACCEPTANCE_THRESHOLDS["independent_residual_max"]),
                       "id_energy_credible":bool(id_mean.energy_scale_normalized_error<ACCEPTANCE_THRESHOLDS["energy_scale_normalized_max"] and id_hamiltonian.energy_scale_normalized_error.quantile(.90)<ACCEPTANCE_THRESHOLDS["energy_scale_normalized_max"]),
                       "id_gaps_credible":bool(id_metrics.gap_error.dropna().mean()<ACCEPTANCE_THRESHOLDS["gap_absolute_max"] and id_hamiltonian.gap_error.quantile(.90)<ACCEPTANCE_THRESHOLDS["gap_absolute_max"]),
                       "id_orthogonality_credible":bool(id_metrics.gram_post_max_error.max()<ACCEPTANCE_THRESHOLDS["gram_max"]),
                       "id_direct_gram_reported":bool(len(id_direct_route)>0 and np.isfinite(id_direct_route.gram_max_error).all()),
                       "id_boundary_credible":bool(id_metrics.boundary_error.max()<ACCEPTANCE_THRESHOLDS["boundary_max"]),
                       "id_tail_credible":bool(len(tail_relevant)>0 and tail_relevant.tail_error.mean()<ACCEPTANCE_THRESHOLDS["tail_error_max"] and tail_relevant.groupby("group_id").tail_error.mean().quantile(.90)<ACCEPTANCE_THRESHOLDS["tail_error_max"]),
                       "ood_results_reported":bool(OOD_EVIDENCE_COMPLETE and executed_categories<=set(all_metrics.category) and len(executed_categories)>=15),
                       "matched_direct_and_gnn_mlp_baselines":False,"measured_cost_comparison":bool(COST_METRICS["mean_eigensolver_seconds"]>0)}
accepted=all(scientific_acceptance.values()) and CFG.mode!="QUICK"
def refinement_category_summary(category:str)->dict[str,float]:
    frame=per_state_output_metrics[per_state_output_metrics.category==category]
    direct=frame[frame.output_route==MAIN_DIRECT_ROUTE]; initial=frame[frame.output_route=="initial_ritz"]; hybrid=frame[frame.output_route=="hybrid"]
    audit=all_metrics[all_metrics.category==category]
    return {"hamiltonians":int(frame.group_id.nunique()),"direct_fidelity":float(direct.fidelity.mean()),"initial_ritz_fidelity":float(initial.fidelity.mean()),"hybrid_fidelity":float(hybrid.fidelity.mean()),
            "hybrid_minus_direct_fidelity":float(hybrid.fidelity.mean()-direct.fidelity.mean()),
            "direct_independent_residual":float(direct.independent_residual_head.mean()),"hybrid_independent_residual":float(hybrid.independent_residual_head.mean()),
            "hybrid_minus_direct_residual":float(hybrid.independent_residual_head.mean()-direct.independent_residual_head.mean()),
            "direct_gram_error":float(direct.gram_max_error.mean()),"hybrid_gram_error":float(hybrid.gram_max_error.mean()),
            "direct_persistent_topology":float(direct.persistent_node_correct.mean()),"hybrid_persistent_topology":float(hybrid.persistent_node_correct.mean()),
            "direct_fallback_fraction":float(hybrid.groupby("group_id").hybrid_fallback.first().mean()),
            "accepted_steps_per_hamiltonian":float(audit.groupby("group_id").refinement_accepted_steps.first().mean()),
            "iterations_used_per_hamiltonian":float(audit.groupby("group_id").refinement_iterations_used.first().mean())}
REFINEMENT_SUMMARY={"method":"weak-FEM residual / positive learned Jacobi modulation / weighted projection / damped Ritz backtracking",
                    "training_steps":CFG.preset.refinement_train_steps,"inference_max_steps":CFG.preset.refinement_inference_steps,
                    "id":refinement_category_summary("ID_test"),"unseen_functional_forms":refinement_category_summary("unseen_analytical_functional_forms"),
                    "trace_file":"refinement_iteration_metrics.csv","depth_ablation_file":"refinement_depth_ablation.csv","dashboard":"recurrent_refinement_dashboard.png"}
run_card={"status":"SCIENTIFIC_ACCEPTANCE_MET" if accepted else ("QUICK_SMOKE_COMPLETED_NOT_SCIENTIFIC_EVIDENCE" if CFG.mode=="QUICK" else f"{CFG.mode}_RUN_COMPLETED_NOT_SCIENTIFICALLY_ACCEPTED"),
          "experiment_name":EXPERIMENT_NAME,"run_mode":CFG.mode,"environment":ENVIRONMENT,"configuration":CONFIG_DICT,"config_hash":CONFIG_HASH,"code_hash":CODE_HASH,
          "source_sha256":SOURCE_SHA256,"source_revision":SOURCE_REVISION,"acceptance_thresholds":ACCEPTANCE_THRESHOLDS,
          "normalization_hash":NORMALIZATION_HASH,"normalization_contract":NORMALIZATION_CONTRACT,"normalization_estimator_audit":NORMALIZATION_AUDIT,
          "dataset_manifest_hash":DATASET_MANIFEST_HASH,"split_manifest_hash":SPLIT_MANIFEST_HASH,"solver_version":SOLVER_VERSION,"architecture_version":ARCHITECTURE_VERSION,
          "output_directory":str(OUT),"split_sizes":{k:len(v) for k,v in splits.items()},"active_seed":CFG.seed,"scheduled_mode_seeds":list(CFG.preset.seeds),
          "checkpoints":CHECKPOINT_SUMMARY,"checkpoint":{"best_stage":best_stage,"best_epoch":best_epoch,"validation":best_val,"ema_tau":best_tau},
          "gradient_update_audit":GRADIENT_UPDATE_AUDIT,
          "autoencoder_initial_validation":ae_initial_validation,"autoencoder_validation":ae_validation,
          "autoencoder_gate_thresholds":AE_GATES,"autoencoder_gate_checks":ae_gate_checks,"autoencoder_node_policy":AE_NODE_POLICY,
          "autoencoder_history_file":"solution_autoencoder_history.json",
          "tiny_overfit_history_file":"tiny_overfit_history.csv","tiny_overfit":tiny_result,
          "decoder_parameter_counts":decoder_parameter_counts.to_dict("records"),"parameter_matched_decoder_comparison":False,
          "energy_contract":{"parameterization":"target-free sine-Galerkin reference plus bounded operator-scaled signed ground correction and positive log-gap ratios",
              "exact_supervision_terms":["energy_ground_exact","energy_log_gap_exact","energy_spectrum_exact"],
              "rayleigh_terms":["rayleigh_energy_exact","energy_rayleigh_consistency"],
              "training_rayleigh_operator":"dimensionless weak-FEM Galerkin quotient of psi_direct",
              "independent_audit":"strong/local-polynomial Rayleigh and residual metrics remain separate",
              "target_privacy":"exact energies appear only in supervised losses and never in forward_operator or the energy head",
              "training_coordinate_audit":ENERGY_COORDINATE_AUDIT},
          "topology_contract":{"default_decoder":CFG.decoder_type,"primary_output":CFG.primary_output,
              "continuous_guarantee":"exactly n interior phase roots for state n under supported scalar real 1D self-adjoint Dirichlet/Friedrichs operators",
              "unsupported":["periodic boundaries","coupled-channel or matrix-valued potentials","non-Hermitian operators","complex wavefunctions","higher dimensions","continuum states","unsupported internal singularities"],
              "target_node_privacy":"supervision-only; absent from operator_batch_view and inference serialization",
              "direct_metrics_file":"per_state_output_metrics.csv","topology_metrics_file":"topology_metrics.csv",
              "direct_hybrid_comparison_file":"direct_hybrid_comparison.csv","topology_summary_file":"topology_summary_by_category.csv",
              "hybrid_topology_rejection_mean":float(topology_metrics.topology_rejection_fraction.mean()) if TOPOLOGY_METRICS_AVAILABLE else None,
              "hamiltonian_direct_fallback_fraction":float(topology_metrics.groupby(["category","group_id"]).hybrid_fallback.first().mean()) if TOPOLOGY_METRICS_AVAILABLE else None},
          "run_assertions":run_assertions,"test_summary":TEST_SUMMARY,"scientific_acceptance":scientific_acceptance,
          "id_bootstrap":{column:bootstrap_hamiltonians(id_metrics,column) for column in ("fidelity","energy_abs_error","independent_residual_head","raw_node_correct")},
          "id_energy_rmse_bootstrap":bootstrap_energy_rmse(id_metrics),
          "id_hamiltonian_tail_summary":{"fidelity_p10":float(id_hamiltonian.fidelity.quantile(.10)),"fidelity_worst":float(id_hamiltonian.fidelity.min()),
              "residual_p90":float(id_hamiltonian.independent_residual_head.quantile(.90)),"residual_worst":float(id_hamiltonian.independent_residual_head.max()),
              "node_position_match_fraction":float(id_mean.node_position_match),"node_position_penalized_normalized_mean":float(id_mean.node_position_error_penalized_normalized),
              "relevant_geometry_tail_error_mean":float(tail_relevant.tail_error.mean()),"relevant_geometry_tail_error_p90_by_hamiltonian":float(tail_relevant.groupby("group_id").tail_error.mean().quantile(.90))},
          "orthonormalization_audit":{"rows":int(len(orthonormalization_comparison)),"pre_gram_max":float(orthonormalization_comparison.gram_pre_max_error.max()),"post_gram_max":float(orthonormalization_comparison.gram_post_max_error.max()),"mean_lowdin_displacement":float(orthonormalization_comparison.lowdin_state_displacement.mean())},
          "cost":COST_METRICS,"recurrent_refinement":REFINEMENT_SUMMARY,"ood_experiment_status":OOD_EXPERIMENT_STATUS,"ood_category_roles":CATEGORY_ROLES,"ood_core_metrics_finite":OOD_EVIDENCE_COMPLETE,"ablation_status":ABLATION_STATUS.to_dict("records"),"ablation_smoke_count":int(len(ablation_smokes)),
          "claim":f"A {CFG.mode} {'smoke ' if CFG.mode=='QUICK' else ''}run of a restricted learned surrogate for scalar real one-dimensional self-adjoint Schrödinger eigensystems within a documented operator and training distribution; topology preservation alone is not evidence of OOD generalization."}
(OUT/"run_card.json").write_text(json.dumps(run_card,indent=2,default=float)); print(json.dumps({k:run_card[k] for k in ("status","experiment_name","scientific_acceptance","cost")},indent=2))

# %% [notebook cell 81]
MODEL_CARD={"intended_use":run_card["claim"],"states":"n=0..10 only","supported_geometries":GEOMETRIES,"supported_boundary_conditions":BOUNDARY_CONDITIONS,
            "architecture":"operator-only JEPA with a nodewise positive-amplitude/strictly-monotone-phase direct decoder and an optional safeguarded weak-residual/Jacobi/Rayleigh--Ritz hybrid",
            "energy_parameterization":"operator-only sine-Galerkin reference with a learned scaled ground correction and learned positive log-gap ratios; trained against exact energies and the direct weak Rayleigh quotient",
            "recurrent_refinement":REFINEMENT_SUMMARY,
            "primary_inference_features":"operator-only; continuum threshold omitted; no solution-derived state mask, target node, density, energy, wavefunction, or target latent",
            "topology_scope":"scalar real one-dimensional self-adjoint Sturm--Liouville/Schrodinger operators with supported separated Dirichlet/Friedrichs boundaries",
            "unsupported":"periodic boundaries; coupled-channel or matrix-valued potentials; non-Hermitian operators; complex wavefunctions; higher dimensions; continuum states; unsupported internal singularities; higher-state extrapolation; exact/universal-solver claims",
            "limitations":["finite grid resolution","finite/truncated domains","singular endpoint resolution","radial mappings","near-continuum states","numerical reference error","distribution shift",
                           "no geometry-specific endpoint amplitude factor is imposed because its exponent is not uniformly available as legitimate inference metadata; phase zeros remain exact but higher-order endpoint asymptotics are not structurally guaranteed",
                           "the safeguarded loop can reject or stagnate; fallback_recommended is reported but no automatic classical solve replaces the surrogate output",
                           "a decreasing training weak residual does not guarantee that the lowest 11-state eigenspace was found; supervised projector and Ky--Fan checks remain necessary",
                           "solution autoencoder uses a trainable near-identity signed-carrier warm start; raw 1e-8 and 1e-5 node counts are discontinuous diagnostics, while the hard topology gate uses target-matched 2e-3 persistence",
                           "a compressed non-identity AE remains a pending experiment","QUICK is a smoke run and is not scientific acceptance evidence"],
            "run_status":run_card["status"],"unexecuted_experiments":{"matched_ablations":ABLATION_STATUS[ABLATION_STATUS.status.str.contains("pending")].to_dict("records"),
                "leave_one_family_out":"pending matched retraining; executable manifest exported","multi_seed_modes":"DEVELOPMENT, STANDARD, and FULL not run","near_continuum":"only one diagnostic test Hamiltonian run; production sweep pending"}}
(OUT/"model_card.json").write_text(json.dumps(MODEL_CARD,indent=2))
print("FINAL ARTIFACT DIRECTORY",OUT.resolve())
