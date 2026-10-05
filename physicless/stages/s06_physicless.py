"""Target-free inference and common V08-compatible diagnostics for the ablation."""

import tracemalloc

OUTPUT_ROUTE="direct_unconstrained"
EVALUATION_ROWS=[]
EVALUATION_TIMING=[]

def evaluate_collection(m:TrueSchrodingerJEPA,rows:Sequence[Mapping[str,Any]],
                        category:str)->pd.DataFrame:
    m.eval()
    metrics=[]
    for record in rows:
        operator=operator_only_record(record)
        input_batch=to_device(collate_records([operator],include_targets=False))
        assert not ({"psi","rho","energy","family","parameters"}&set(input_batch))
        if DEVICE.type=="cuda": torch.cuda.reset_peak_memory_stats(DEVICE)
        tracemalloc.start()
        started=time.perf_counter()
        with torch.inference_mode(): out=m.forward_operator(input_batch)
        elapsed=time.perf_counter()-started
        _,python_peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        device_peak=(int(torch.cuda.max_memory_allocated(DEVICE)) if DEVICE.type=="cuda" else None)
        x=np.asarray(record["x"],dtype=np.float64)
        w=np.asarray(record["quadrature_weights"],dtype=np.float64)
        V=np.asarray(record["V_raw"],dtype=np.float64)
        exact=np.asarray(record["psi"],dtype=np.float64)
        exact_energy=np.asarray(record["energy"],dtype=np.float64)
        psi=out["psi"][0,:,:len(x)].detach().cpu().numpy().astype(np.float64)
        energy=out["energy"][0].detach().cpu().numpy().astype(np.float64)
        rho=psi**2
        assert np.isfinite(psi).all() and np.isfinite(energy).all()
        assert np.max(np.abs(psi[:,[0,-1]]))==0
        gram=weighted_gram(psi,w)
        gram_error=float(np.max(np.abs(gram-np.eye(CFG.k_states))))
        norm=np.sum(w[None,:]*psi**2,axis=1)
        overlap=np.sum(w[None,:]*psi*exact,axis=1)
        aligned=psi*np.where(overlap>=0,1.0,-1.0)[:,None]
        residual=independent_residual(x,w,V,psi,energy)
        valid=np.asarray(record["valid_state_mask"],dtype=bool)
        EVALUATION_TIMING.append({"category":category,"group_id":record["group_id"],
            "family":record["family"],"inference_seconds":elapsed,
            "python_peak_bytes":python_peak,"device_peak_bytes":device_peak})
        for state in np.flatnonzero(valid):
            dx=np.diff(x)
            derivative_error=float(np.sum(dx*(np.diff(aligned[state]-exact[state])/dx)**2))
            derivative_target=float(np.sum(dx*(np.diff(exact[state])/dx)**2))
            l2_error=float(np.sum(w*(aligned[state]-exact[state])**2))
            l2_target=float(np.sum(w*exact[state]**2))
            fidelity=float(overlap[state]**2)
            raw_nodes=int(persistent_nodes(psi[state],1e-8))
            stable_nodes=int(persistent_nodes(psi[state],2e-3))
            metrics.append({
                "category":category,"group_id":record["group_id"],
                "family":record["family"],"geometry":record["geometry"],
                "state":int(state),"output_route":OUTPUT_ROUTE,
                "fidelity":fidelity,"infidelity":1-fidelity,
                "h1_relative":float(np.sqrt((l2_error+derivative_error)/max(l2_target+derivative_target,1e-30))),
                "aligned_relative_l2":float(np.sqrt(l2_error/max(l2_target,1e-30))),
                "energy_abs_error":float(abs(energy[state]-exact_energy[state])),
                "energy_squared_error":float((energy[state]-exact_energy[state])**2),
                "ordering_violation":float(state>0 and energy[state]<=energy[state-1]),
                "independent_residual_head":float(residual[state]),
                "exact_label_residual_floor":float(record["solver_residual"][state]),
                "raw_node_count":raw_nodes,"raw_node_correct":float(raw_nodes==state),
                "persistent_node_count":stable_nodes,
                "persistent_node_correct":float(stable_nodes==state),
                "gram_max_error":gram_error,"density_normalization_error":float(abs(norm[state]-1)),
                "density_integrated_absolute_error":float(np.sum(w*np.abs(rho[state]-exact[state]**2))),
                "boundary_error":float(np.max(np.abs(psi[state,[0,-1]]))),
                "binding_margin":float(record["bound_margin"][state]),
            })
    return pd.DataFrame(metrics)


id_metrics=evaluate_collection(model,splits["test"],"ID_test")
ood_frames=[evaluate_collection(model,rows,category) for category,rows in ood_suites.items() if rows]
ood_metrics=pd.concat(ood_frames,ignore_index=True) if ood_frames else pd.DataFrame(columns=id_metrics.columns)
all_metrics=pd.concat([id_metrics,ood_metrics],ignore_index=True)
per_state_output_metrics=all_metrics.copy()
timing_metrics=pd.DataFrame(EVALUATION_TIMING)

id_ids={row["group_id"] for row in splits["test"]}
assert set(id_metrics.group_id)==id_ids
assert id_metrics.groupby("group_id").state.nunique().eq(CFG.k_states).all()
assert not id_metrics.duplicated(["group_id","family","state"]).any()
assert set(all_metrics.output_route)=={OUTPUT_ROUTE}
core=["fidelity","h1_relative","energy_abs_error","independent_residual_head",
      "raw_node_correct","gram_max_error","density_normalization_error"]
assert np.isfinite(all_metrics[core].to_numpy()).all()
assert all_metrics.fidelity.between(-1e-4,1+1e-3).all()
assert all_metrics.boundary_error.le(1e-7).all()
assert all_metrics.density_normalization_error.le(1e-3).all()

all_metrics.to_csv(OUT/"per_state_metrics.csv",index=False)
per_state_output_metrics.to_csv(OUT/"per_state_output_metrics.csv",index=False)
id_metrics.to_csv(OUT/"id_test_metrics.csv",index=False)
ood_metrics.to_csv(OUT/"ood_metrics.csv",index=False)
timing_metrics.to_csv(OUT/"evaluation_timing.csv",index=False)

SUMMARY_METRICS=("fidelity","h1_relative","energy_abs_error","independent_residual_head",
                 "raw_node_correct","persistent_node_correct","gram_max_error","ordering_violation")
def grouped_summary(frame:pd.DataFrame,keys:list[str])->pd.DataFrame:
    return frame.groupby(keys,dropna=False)[list(SUMMARY_METRICS)].agg(["mean","std","count"]).reset_index().pipe(
        lambda table: table.set_axis(["_".join(part for part in col if part) if isinstance(col,tuple) else col
                                      for col in table.columns],axis=1))

grouped_summary(all_metrics,["category"]).to_csv(OUT/"generalization_summary.csv",index=False)
grouped_summary(all_metrics,["category","family"]).to_csv(OUT/"family_summary.csv",index=False)
grouped_summary(all_metrics,["category","state"]).to_csv(OUT/"state_summary.csv",index=False)
grouped_summary(all_metrics,["category","family","state"]).to_csv(OUT/"family_state_summary.csv",index=False)

EVALUATION_AUDIT={"mode":CFG.mode,"route":OUTPUT_ROUTE,
    "id_hamiltonians":len(id_ids),"id_states":len(id_metrics),
    "ood_categories":sorted(ood_metrics.category.unique().tolist()) if len(ood_metrics) else [],
    "target_free_forward":True,"same_split_manifest_hash":SPLIT_MANIFEST_HASH,
    "finite_core_metrics":True,"no_ritz_or_refinement_route":True}
(OUT/"evaluation_audit.json").write_text(json.dumps(EVALUATION_AUDIT,indent=2))
print("PHYSICLESS EVALUATION: PASS",EVALUATION_AUDIT)
