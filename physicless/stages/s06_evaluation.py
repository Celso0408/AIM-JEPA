"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 65]
PREDICTION_CACHE:dict[tuple[str,str],dict[str,Any]]={}
REFINEMENT_ITERATION_ROWS:list[dict[str,Any]]=[]
OUTPUT_ROUTE_ROWS:list[dict[str,Any]]=[]
TOPOLOGY_HAMILTONIAN_ROWS:list[dict[str,Any]]=[]


def _safe_divide(a:float,b:float)->float: return float(a/max(abs(b),1e-30))


def evaluate_collection(m:TrueSchrodingerJEPA,rows:Sequence[Mapping[str,Any]],category:str)->tuple[pd.DataFrame,dict[str,Any]]:
    metric_rows=[]; predicted_tokens=[]; target_tokens=[]; inference_seconds=0.0; target_seconds=0.0; peak_memory=0; peak_device_memory=0
    refinement_attempts=0; refinement_accepts=0; refinement_fallbacks=0; refinement_iterations=[]; refinement_initial_blocks=[]; refinement_final_blocks=[]
    for record in rows:
        operator=operator_only_record(record); batch=to_device(collate_records([operator],include_targets=False)); target_batch=to_device(collate_records([record],include_targets=True))
        m.eval()
        if DEVICE.type=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tracemalloc.start(); started=time.perf_counter()
        with torch.no_grad(): out=m.forward_operator(batch)
        if DEVICE.type=="cuda": torch.cuda.synchronize()
        inference_seconds+=time.perf_counter()-started; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); peak_memory=max(peak_memory,peak)
        if DEVICE.type=="cuda": peak_device_memory=max(peak_device_memory,int(torch.cuda.max_memory_allocated()))
        if DEVICE.type=="cuda": torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.no_grad(): ztarget=m.encode_target(target_batch)
        if DEVICE.type=="cuda": torch.cuda.synchronize()
        target_seconds+=time.perf_counter()-started
        psi=out["psi"][0].cpu().numpy(); psi_pre=out["psi_pre_orth"][0].cpu().numpy(); psi_orthogonal=out["psi_orthogonal"][0].cpu().numpy()
        psi_hybrid=out["psi_hybrid"][0].cpu().numpy(); energy_hybrid=out["energy_hybrid"][0].cpu().numpy()
        energy=out["energy"][0].cpu().numpy(); rho=psi**2
        psi_initial=out["psi_initial"][0].cpu().numpy(); energy_initial=out["energy_initial"][0].cpu().numpy()
        psi_trace=out["refinement_psi_trace"][0].cpu().numpy(); energy_trace=out["refinement_energy_trace"][0].cpu().numpy()
        residual_trace=out["refinement_residual_trace"][0].cpu().numpy(); acceptance_trace=out["refinement_acceptance_trace"][0].cpu().numpy()
        step_trace=out["refinement_step_size_trace"][0].cpu().numpy(); attempted_trace=out["refinement_attempted_trace"][0].cpu().numpy()
        refinement_attempts+=int(attempted_trace.sum()); refinement_accepts+=int(acceptance_trace.sum())
        refinement_fallbacks+=int(out["refinement_fallback_recommended"][0].cpu()); refinement_iterations.append(int(out["refinement_iterations_used"][0].cpu()))
        refinement_initial_blocks.append(float(np.sqrt(np.mean(residual_trace[0]**2)))); refinement_final_blocks.append(float(np.sqrt(np.mean(residual_trace[-1]**2))))
        exact=np.asarray(record["psi"]); exact_energy=np.asarray(record["energy"]); x=np.asarray(record["x"]); w=np.asarray(record["quadrature_weights"]); V=np.asarray(record["V_raw"])
        valid_state_mask=np.asarray(record["valid_state_mask"],dtype=bool)
        spec=PotentialSpec(record["family"],record["parameters"],record["geometry"],record["boundary_condition"],record["domain_left"],record["domain_right"],record["continuum_threshold"])
        direct_route_name="direct_topology" if m.decoder_type=="topology_phase" else "direct_spectral"
        route_payloads={
            direct_route_name:(out["psi_direct"][0].cpu().numpy(),out["energy_direct"][0].cpu().numpy()),
            "initial_ritz":(out["psi_initial_ritz"][0].cpu().numpy(),out["energy_initial_ritz"][0].cpu().numpy()),
            "hybrid":(out["psi_hybrid"][0].cpu().numpy(),out["energy_hybrid"][0].cpu().numpy()),
        }
        exact_tail_by_state=geometry_tail_probability(spec,x,w,exact)
        symmetric_operator=float(np.linalg.norm(V-V[::-1])/max(np.linalg.norm(V),1e-30))<1e-4
        for route,(route_psi,route_energy) in route_payloads.items():
            route_overlap=np.sum(w[None]*route_psi*exact,axis=1); route_fidelity=route_overlap**2
            route_sign=np.where(route_overlap>=0,1.0,-1.0); route_aligned=route_psi*route_sign[:,None]
            route_rayleigh=independent_rayleigh(x,w,V,route_psi)
            route_residual=independent_residual(x,w,V,route_psi,route_energy)
            route_rayleigh_residual=independent_residual(x,w,V,route_psi,route_rayleigh)
            route_gram=weighted_gram(route_psi,w); route_tail=geometry_tail_probability(spec,x,w,route_psi)
            for state in range(CFG.k_states):
                if not valid_state_mask[state]: continue
                exact_nodes=target_node_positions_from_wave(x,exact[state],state); route_nodes=node_positions(x,route_psi[state],1e-8)
                node_mae=(float(np.mean(np.abs(exact_nodes-route_nodes))) if state and len(route_nodes)==state else (0.0 if state==0 and len(route_nodes)==0 else np.nan))
                node_match=bool(len(route_nodes)==state); node_penalized=node_mae if node_match else float(x[-1]-x[0])
                route_rho=route_psi[state]**2; exact_rho=exact[state]**2
                derivative_error_sq=np.sum(np.diff(x)*(np.diff(route_aligned[state])/np.diff(x)-np.diff(exact[state])/np.diff(x))**2)
                derivative_target_sq=np.sum(np.diff(x)*(np.diff(exact[state])/np.diff(x))**2)
                l2_error_sq=np.sum(w*(route_aligned[state]-exact[state])**2); l2_target_sq=np.sum(w*exact[state]**2)
                mass_pred=np.maximum(w*route_rho,0); mass_exact=np.maximum(w*exact_rho,0)
                mass_pred/=max(float(mass_pred.sum()),1e-30); mass_exact/=max(float(mass_exact.sum()),1e-30)
                mixture=.5*(mass_pred+mass_exact)
                js=.5*(np.sum(np.where(mass_pred>0,mass_pred*np.log((mass_pred+1e-30)/(mixture+1e-30)),0))+
                       np.sum(np.where(mass_exact>0,mass_exact*np.log((mass_exact+1e-30)/(mixture+1e-30)),0)))
                gap_error=(np.nan if state==0 else
                           abs((route_energy[state]-route_energy[state-1])-(exact_energy[state]-exact_energy[state-1])))
                parity_correct=np.nan
                if symmetric_operator:
                    exact_parity=1 if np.sum(w*exact[state]*exact[state,::-1])>=0 else -1
                    route_parity=1 if np.sum(w*route_psi[state]*route_psi[state,::-1])>=0 else -1
                    parity_correct=float(exact_parity==route_parity)
                subspace_error=np.nan
                if state<CFG.k_states-1:
                    q_projector=np.sqrt(w)[None]*exact[state:state+2]; p_projector=np.sqrt(w)[None]*route_psi[state:state+2]
                    subspace_error=float(np.linalg.norm(q_projector.T@q_projector-p_projector.T@p_projector,ord="fro")/math.sqrt(2))
                OUTPUT_ROUTE_ROWS.append({"category":category,"group_id":record["group_id"],"family":record["family"],"geometry":record["geometry"],
                    "state":state,"target_state_valid":1.0,"binding_margin":float(record["bound_margin"][state]),"output_route":route,
                    "fidelity":float(route_fidelity[state]),"infidelity":float(1-route_fidelity[state]),
                    "aligned_relative_l2":float(np.sqrt(l2_error_sq/max(l2_target_sq,1e-30))),
                    "h1_relative":float(np.sqrt((l2_error_sq+derivative_error_sq)/max(l2_target_sq+derivative_target_sq,1e-30))),
                    "linf":float(np.max(np.abs(route_aligned[state]-exact[state]))),
                    "energy_abs_error":float(abs(route_energy[state]-exact_energy[state])),
                    "energy_squared_error":float((route_energy[state]-exact_energy[state])**2),
                    "energy_scale_normalized_error":float(abs(route_energy[state]-exact_energy[state])/(np.ptp(exact_energy)+1e-12)),
                    "gap_error":float(gap_error),"ordering_violation":float(state>0 and route_energy[state]<=route_energy[state-1]),
                    "rayleigh_energy":float(route_rayleigh[state]),"head_rayleigh_disagreement":float(abs(route_energy[state]-route_rayleigh[state])),
                    "independent_residual_head":float(route_residual[state]),"independent_residual_rayleigh":float(route_rayleigh_residual[state]),
                    "raw_node_correct":float(len(route_nodes)==state),
                    "persistent_node_correct":float(persistent_nodes(route_psi[state],2e-3)==state),"node_position_mae":node_mae,
                    "node_position_mae_normalized":node_mae/max(float(x[-1]-x[0]),1e-30) if np.isfinite(node_mae) else np.nan,
                    "node_position_match":float(node_match),"node_position_error_penalized":float(node_penalized),
                    "node_position_error_penalized_normalized":float(node_penalized/max(float(x[-1]-x[0]),1e-30)),
                    "parity_correct":parity_correct,"subspace_projector_error":subspace_error,
                    "density_normalization_error":float(abs(np.sum(w*route_rho)-1)),
                    "density_integrated_absolute_error":float(np.sum(w*np.abs(route_rho-exact_rho))),
                    "density_relative_l2":float(np.sqrt(np.sum(w*(route_rho-exact_rho)**2)/(np.sum(w*exact_rho**2)+1e-30))),
                    "hellinger":float(np.sqrt(.5*np.sum((np.sqrt(mass_pred)-np.sqrt(mass_exact))**2))),
                    "jensen_shannon":float(js),"wasserstein":float(scipy.stats.wasserstein_distance(x,x,u_weights=mass_pred,v_weights=mass_exact)),
                    "rho_square_consistency":float(np.max(np.abs(route_rho-route_psi[state]**2))),
                    "x_moment_error":float(abs(np.sum(w*x*route_rho)-np.sum(w*x*exact_rho))),
                    "x2_moment_error":float(abs(np.sum(w*x*x*route_rho)-np.sum(w*x*x*exact_rho))),
                    "V_expectation_error":float(abs(np.sum(w*V*route_rho)-np.sum(w*V*exact_rho))),
                    "gram_max_error":float(np.max(np.abs(route_gram-np.eye(CFG.k_states)))),
                    "boundary_error":float(np.max(np.abs(route_psi[state,[0,-1]]))),
                    "predicted_tail_probability":float(route_tail[state]),"exact_tail_probability":float(exact_tail_by_state[state]),
                    "tail_error":float(abs(route_tail[state]-exact_tail_by_state[state])),
                    "hybrid_fallback":float(out["hybrid_fallback_recommended"][0].cpu())})

        domain_length=float(x[-1]-x[0]); a=float(x[0]); n_valid=len(x)
        phase_nodes_t=out.get("predicted_node_positions_t")
        if phase_nodes_t is not None:
            phase_nodes_x=a+domain_length*phase_nodes_t[0].cpu().numpy()
            phase_increment=out["phase_increment"][0,:,:n_valid-1].cpu().numpy(); q_values=out["phase_density"][0,:,:n_valid].cpu().numpy()
            amplitude_values=out["amplitude"][0,:,:n_valid].cpu().numpy(); cdf_values=out["phase_cdf"][0,:,:n_valid].cpu().numpy()
            proposal_attempts=int(out["refinement_topology_proposal_attempt_count"][0].sum().cpu())
            proposal_rejections=int(out["refinement_topology_proposal_rejection_count"][0].sum().cpu())
            topology_rejection_fraction=float(proposal_rejections/max(proposal_attempts,1))
            direct_psi=out["psi_direct"][0].cpu().numpy()
            dx=np.diff(x)
            for state in range(CFG.k_states):
                exact_nodes=target_node_positions_from_wave(x,exact[state],state); predicted_nodes=phase_nodes_x[state,:state]
                target_valid=bool(valid_state_mask[state]); errors=(np.abs(predicted_nodes-exact_nodes) if state and target_valid else np.empty(0))
                def probability_cdf(density:np.ndarray)->np.ndarray:
                    increment=.5*(density[1:]+density[:-1])*dx; cumulative=np.r_[0.,np.cumsum(increment)]
                    return cumulative/max(float(cumulative[-1]),1e-30)
                pred_mass_cdf=probability_cdf(direct_psi[state]**2); exact_mass_cdf=probability_cdf(exact[state]**2)
                pred_at=np.interp(exact_nodes,x,pred_mass_cdf); exact_at=np.interp(exact_nodes,x,exact_mass_cdf)
                pred_lobes=np.diff(np.r_[0.,pred_at,1.]); exact_lobes=np.diff(np.r_[0.,exact_at,1.])
                predicted_phase_lobes=np.diff(np.r_[0.,np.interp(predicted_nodes,x,pred_mass_cdf),1.])
                q=q_values[state]; logq=np.log(np.maximum(q,1e-30)); dt=np.diff((x-a)/domain_length)
                boundaries=np.r_[a,predicted_nodes,x[-1]]; spacing=np.diff(boundaries); midpoints=.5*(x[:-1]+x[1:])
                interval_counts=np.asarray([np.count_nonzero((midpoints>=boundaries[lobe])&(midpoints<boundaries[lobe+1]))
                                            for lobe in range(len(boundaries)-1)])
                cdf=cdf_values[state]; monotone=bool(np.all(np.diff(cdf)>0) and cdf[0]==0 and cdf[-1]==1)
                crossing_count=sum(int(np.count_nonzero((cdf[:-1]<j/(state+1))&(cdf[1:]>=j/(state+1))))==1
                                   for j in range(1,state+1))
                continuous_compliance=float(monotone and crossing_count==state and np.all(amplitude_values[state,1:-1]>0))
                TOPOLOGY_HAMILTONIAN_ROWS.append({"category":category,"group_id":record["group_id"],"family":record["family"],"geometry":record["geometry"],
                    "state":state,"target_state_valid":float(target_valid),"binding_margin":float(record["bound_margin"][state]),
                    "phase_crossing_count":int(crossing_count),"exact_node_count_compliance":continuous_compliance,
                    "persistent_node_count_compliance":float(persistent_nodes(direct_psi[state],2e-3)==state),
                    "node_position_mae":float(errors.mean()) if state and target_valid else (0.0 if state==0 and target_valid else np.nan),
                    "node_position_mae_normalized":float(errors.mean()/domain_length) if state and target_valid else (0.0 if state==0 and target_valid else np.nan),
                    "worst_node_position_error":float(errors.max()) if state and target_valid else (0.0 if state==0 and target_valid else np.nan),
                    "lobe_probability_mass_mae":float(np.mean(np.abs(pred_lobes-exact_lobes))) if target_valid else np.nan,
                    "minimum_predicted_lobe_probability_mass":float(predicted_phase_lobes.min()),
                    "minimum_predicted_nodal_spacing":float(spacing.min()),"minimum_grid_intervals_per_lobe":int(interval_counts.min()),
                    "maximum_phase_increment":float(phase_increment[state].max()),"p95_phase_increment":float(np.quantile(phase_increment[state],.95)),
                    "phase_density_max_min_ratio":float(q.max()/max(q.min(),1e-30)),
                    "phase_density_smoothness":float(np.sum(np.diff(logq)**2/np.maximum(dt,1e-30))),
                    "amplitude_dynamic_range":float(amplitude_values[state].max()/max(amplitude_values[state].min(),1e-30)),
                    "topology_proposals_attempted":proposal_attempts,"topology_proposals_rejected":proposal_rejections,
                    "topology_rejection_fraction":topology_rejection_fraction,"hybrid_fallback":float(out["hybrid_fallback_recommended"][0].cpu()),
                    "initial_ritz_topology_valid":float(out["initial_ritz_topology_valid_by_state"][0,state].cpu()),
                    "hybrid_topology_valid":float(out["hybrid_topology_valid_by_state"][0,state].cpu())})
        overlap=np.sum(w[None,:]*psi*exact,axis=1); signs=np.where(overlap>=0,1.0,-1.0); aligned=psi*signs[:,None]
        overlap_initial=np.sum(w[None,:]*psi_initial*exact,axis=1); fidelity_initial=overlap_initial**2
        overlap_hybrid=np.sum(w[None,:]*psi_hybrid*exact,axis=1); fidelity_hybrid=overlap_hybrid**2
        overlap_pre=np.sum(w[None,:]*psi_pre*exact,axis=1); signs_pre=np.where(overlap_pre>=0,1.0,-1.0); aligned_pre=psi_pre*signs_pre[:,None]
        fidelity=overlap**2; fidelity_pre=overlap_pre**2
        rayleigh=independent_rayleigh(x,w,V,psi); rayleigh_pre=independent_rayleigh(x,w,V,psi_pre)
        head_res=independent_residual(x,w,V,psi,energy); rq_res=independent_residual(x,w,V,psi,rayleigh)
        initial_head_res=independent_residual(x,w,V,psi_initial,energy_initial)
        hybrid_head_res=independent_residual(x,w,V,psi_hybrid,energy_hybrid)
        head_res_pre=independent_residual(x,w,V,psi_pre,energy); rq_res_pre=independent_residual(x,w,V,psi_pre,rayleigh_pre); floor=np.asarray(record["solver_residual"])
        gram_pre=weighted_gram(psi_pre,w); gram_post=weighted_gram(psi_orthogonal,w); gram_pre_min_eigenvalue=float(np.linalg.eigvalsh(gram_pre).min())
        lowdin_displacement=np.sqrt(np.sum(w[None,:]*(psi_orthogonal-psi_pre)**2,axis=1)); symmetric=float(np.linalg.norm(V-V[::-1])/max(np.linalg.norm(V),1e-30))<1e-4
        PREDICTION_CACHE[(category,record["group_id"])]={"psi":psi,"psi_pre":psi_pre,"psi_initial":psi_initial,"rho":rho,"energy":energy,"energy_initial":energy_initial,"rayleigh":rayleigh,"rayleigh_pre":rayleigh_pre,"head_residual":head_res,"rq_residual":rq_res,
            "psi_direct":out["psi_direct"][0].cpu().numpy(),"rho_direct":out["rho_direct"][0].cpu().numpy(),"energy_direct":out["energy_direct"][0].cpu().numpy(),
            "psi_orthogonal":psi_orthogonal,
            "psi_hybrid":out["psi_hybrid"][0].cpu().numpy(),"rho_hybrid":out["rho_hybrid"][0].cpu().numpy(),"energy_hybrid":out["energy_hybrid"][0].cpu().numpy(),
            "amplitude":out["amplitude"][0].cpu().numpy() if "amplitude" in out else None,
            "phase_cdf":out["phase_cdf"][0].cpu().numpy() if "phase_cdf" in out else None,
            "theta":out["theta"][0].cpu().numpy() if "theta" in out else None,
            "phase_density":out["phase_density"][0].cpu().numpy() if "phase_density" in out else None,
            "predicted_node_positions_t":out["predicted_node_positions_t"][0].cpu().numpy() if "predicted_node_positions_t" in out else None,
            "hybrid_fallback":bool(out["hybrid_fallback_recommended"][0].cpu())}
        valid_tensor=torch.as_tensor(valid_state_mask,device=DEVICE)
        predicted_tokens.append(out["z_pred"][0,valid_tensor].cpu()); target_tokens.append(ztarget[0,valid_tensor].cpu())
        predicted_tail=geometry_tail_probability(spec,x,w,psi); exact_tail=geometry_tail_probability(spec,x,w,exact)
        for iteration,(iter_psi,iter_energy,iter_training_residual) in enumerate(zip(psi_trace,energy_trace,residual_trace)):
            iter_overlap=np.sum(w[None,:]*iter_psi*exact,axis=1); iter_fidelity=iter_overlap**2
            iter_independent=independent_residual(x,w,V,iter_psi,iter_energy); iter_gram=weighted_gram(iter_psi,w)
            for state in range(CFG.k_states):
                if not valid_state_mask[state]: continue
                REFINEMENT_ITERATION_ROWS.append({"category":category,"group_id":record["group_id"],"family":record["family"],"iteration":iteration,"state":state,
                    "training_relative_residual":float(iter_training_residual[state]),"independent_residual_head":float(iter_independent[state]),
                    "fidelity":float(iter_fidelity[state]),"energy_abs_error":float(abs(iter_energy[state]-exact_energy[state])),
                    "accepted_from_previous":np.nan if iteration==0 else float(acceptance_trace[iteration-1]),
                    "attempted_from_previous":np.nan if iteration==0 else float(attempted_trace[iteration-1]),
                    "step_size":np.nan if iteration==0 else float(step_trace[iteration-1]),
                    "gram_max_error":float(np.max(np.abs(iter_gram-np.eye(CFG.k_states)))),
                    "boundary_error":float(np.max(np.abs(iter_psi[state,[0,-1]])))})
        for state in range(CFG.k_states):
            if not valid_state_mask[state]: continue
            derivative_error_sq=np.sum(np.diff(x)*(np.diff(aligned[state])/np.diff(x)-np.diff(exact[state])/np.diff(x))**2)
            derivative_target_sq=np.sum(np.diff(x)*(np.diff(exact[state])/np.diff(x))**2)
            l2_error_sq=np.sum(w*(aligned[state]-exact[state])**2); l2_target_sq=np.sum(w*exact[state]**2)
            h1_relative=np.sqrt((l2_error_sq+derivative_error_sq)/max(l2_target_sq+derivative_target_sq,1e-30))
            h1_seminorm_relative=np.sqrt(derivative_error_sq/max(derivative_target_sq,1e-30))
            pre_derivative_error_sq=np.sum(np.diff(x)*(np.diff(aligned_pre[state])/np.diff(x)-np.diff(exact[state])/np.diff(x))**2)
            pre_l2_error_sq=np.sum(w*(aligned_pre[state]-exact[state])**2)
            pre_h1_relative=np.sqrt((pre_l2_error_sq+pre_derivative_error_sq)/max(l2_target_sq+derivative_target_sq,1e-30))
            density_exact=exact[state]**2; mass_pred=w*rho[state]; mass_exact=w*density_exact; mass_pred/=mass_pred.sum(); mass_exact/=mass_exact.sum(); mixture=.5*(mass_pred+mass_exact)
            js=.5*(np.sum(np.where(mass_pred>0,mass_pred*np.log((mass_pred+1e-30)/(mixture+1e-30)),0))+np.sum(np.where(mass_exact>0,mass_exact*np.log((mass_exact+1e-30)/(mixture+1e-30)),0)))
            pred_nodes=persistent_nodes(psi[state],1e-8); pred_persistent=persistent_nodes(psi[state],2e-3); pre_nodes=persistent_nodes(psi_pre[state],1e-8); pre_persistent=persistent_nodes(psi_pre[state],2e-3)
            exact_positions=node_positions(x,exact[state],1e-8); pred_positions=node_positions(x,psi[state],1e-8); pre_positions=node_positions(x,psi_pre[state],1e-8)
            position_error=float(np.mean(np.abs(exact_positions-pred_positions))) if len(exact_positions)==len(pred_positions) and len(exact_positions)>0 else (0.0 if len(exact_positions)==len(pred_positions)==0 else np.nan)
            position_match=bool(len(exact_positions)==len(pred_positions)); position_error_penalized=position_error if position_match else float(x[-1]-x[0])
            pre_position_error=float(np.mean(np.abs(exact_positions-pre_positions))) if len(exact_positions)==len(pre_positions) and len(exact_positions)>0 else (0.0 if len(exact_positions)==len(pre_positions)==0 else np.nan)
            pre_position_match=bool(len(exact_positions)==len(pre_positions)); pre_position_penalized=pre_position_error if pre_position_match else float(x[-1]-x[0])
            domain_length=float(x[-1]-x[0])
            parity_correct=np.nan
            if symmetric:
                exact_parity=1 if np.sum(w*exact[state]*exact[state,::-1])>=0 else -1; predicted_parity=1 if np.sum(w*psi[state]*psi[state,::-1])>=0 else -1; parity_correct=float(exact_parity==predicted_parity)
            subspace_error=np.nan
            if state<CFG.k_states-1:
                q=np.sqrt(w)[None,:]*exact[state:state+2]; p=np.sqrt(w)[None,:]*psi[state:state+2]; subspace_error=float(np.linalg.norm(q.T@q-p.T@p,ord="fro")/math.sqrt(2))
            gap_error=np.nan if state==0 else abs((energy[state]-energy[state-1])-(exact_energy[state]-exact_energy[state-1]))
            metric_rows.append({"category":category,"group_id":record["group_id"],"family":record["family"],"state":state,"geometry":record["geometry"],
                "binding_margin":float(record["bound_margin"][state]),"fidelity":float(fidelity[state]),"infidelity":float(1-fidelity[state]),
                "initial_ritz_fidelity":float(fidelity_initial[state]),"hybrid_fidelity":float(fidelity_hybrid[state]),
                "refinement_fidelity_gain":float(fidelity_hybrid[state]-fidelity_initial[state]),
                "initial_ritz_to_primary_fidelity_change":float(fidelity[state]-fidelity_initial[state]),
                "pre_orth_fidelity":float(fidelity_pre[state]),"pre_orth_infidelity":float(1-fidelity_pre[state]),
                "aligned_relative_l2":float(np.sqrt(l2_error_sq/max(l2_target_sq,1e-30))),"linf":float(np.max(np.abs(aligned[state]-exact[state]))),
                "pre_orth_aligned_relative_l2":float(np.sqrt(pre_l2_error_sq/max(l2_target_sq,1e-30))),"pre_orth_h1_relative":float(pre_h1_relative),
                "h1_relative":float(h1_relative),"h1_seminorm_relative":float(h1_seminorm_relative),"raw_node_correct":float(pred_nodes==state),"persistent_node_correct":float(pred_persistent==state),
                "node_position_error":position_error,"node_position_match":float(position_match),"node_position_error_penalized":float(position_error_penalized),
                "node_position_error_normalized":float(position_error/domain_length) if np.isfinite(position_error) else np.nan,"node_position_error_penalized_normalized":float(position_error_penalized/domain_length),
                "pre_orth_raw_node_correct":float(pre_nodes==state),"pre_orth_persistent_node_correct":float(pre_persistent==state),
                "pre_orth_node_position_error":pre_position_error,"pre_orth_node_position_match":float(pre_position_match),"pre_orth_node_position_error_penalized":float(pre_position_penalized),
                "pre_orth_node_position_error_penalized_normalized":float(pre_position_penalized/domain_length),
                "parity_correct":parity_correct,"subspace_projector_error":subspace_error,
                "density_normalization_error":float(abs(np.sum(w*rho[state])-1)),"density_integrated_absolute_error":float(np.sum(w*np.abs(rho[state]-density_exact))),
                "density_relative_l2":float(np.sqrt(np.sum(w*(rho[state]-density_exact)**2)/(np.sum(w*density_exact**2)+1e-30))),
                "hellinger":float(np.sqrt(.5*np.sum((np.sqrt(mass_pred)-np.sqrt(mass_exact))**2))),"jensen_shannon":float(js),
                "wasserstein":float(scipy.stats.wasserstein_distance(x,x,u_weights=mass_pred,v_weights=mass_exact)),
                "x_moment_error":float(abs(np.sum(w*x*rho[state])-np.sum(w*x*density_exact))),
                "x2_moment_error":float(abs(np.sum(w*x*x*rho[state])-np.sum(w*x*x*density_exact))),"rho_square_consistency":float(np.max(np.abs(rho[state]-psi[state]**2))),
                "energy_abs_error":float(abs(energy[state]-exact_energy[state])),"energy_squared_error":float((energy[state]-exact_energy[state])**2),
                "energy_scale_normalized_error":float(abs(energy[state]-exact_energy[state])/(np.ptp(exact_energy)+1e-12)),"gap_error":float(gap_error),
                "ordering_violation":float(state>0 and energy[state]<=energy[state-1]),"rayleigh_energy":float(rayleigh[state]),"head_rayleigh_disagreement":float(abs(energy[state]-rayleigh[state])),
                "independent_residual_head":float(head_res[state]),"independent_residual_rayleigh":float(rq_res[state]),"exact_label_residual_floor":float(floor[state]),
                "initial_ritz_independent_residual_head":float(initial_head_res[state]),"hybrid_independent_residual_head":float(hybrid_head_res[state]),
                "refinement_independent_residual_reduction":float(initial_head_res[state]-hybrid_head_res[state]),
                "initial_ritz_to_primary_residual_reduction":float(initial_head_res[state]-head_res[state]),
                "refinement_training_residual_initial":float(residual_trace[0,state]),"refinement_training_residual_final":float(residual_trace[-1,state]),
                "refinement_accepted_steps":int(acceptance_trace.sum()),"refinement_iterations_used":int(out["refinement_iterations_used"][0].cpu()),
                "pre_orth_independent_residual_head":float(head_res_pre[state]),"pre_orth_independent_residual_rayleigh":float(rq_res_pre[state]),
                "pre_orth_rayleigh_energy":float(rayleigh_pre[state]),"lowdin_state_displacement":float(lowdin_displacement[state]),"pre_gram_min_eigenvalue":gram_pre_min_eigenvalue,
                "gram_pre_max_error":float(np.max(np.abs(gram_pre-np.eye(CFG.k_states)))),"gram_post_max_error":float(np.max(np.abs(gram_post-np.eye(CFG.k_states)))),
                "max_offdiagonal_overlap":float(np.max(np.abs(gram_post-np.diag(np.diag(gram_post))))),"boundary_error":float(np.max(np.abs(psi[state,[0,-1]]))),
                "predicted_tail_probability":float(predicted_tail[state]),"exact_tail_probability":float(exact_tail[state]),
                "tail_error":float(abs(predicted_tail[state]-exact_tail[state])),
                "node_theorem_compliance":float(pred_nodes==state),"x_expectation_error":float(abs(np.sum(w*x*rho[state])-np.sum(w*x*density_exact))),
                "V_expectation_error":float(abs(np.sum(w*V*rho[state])-np.sum(w*V*density_exact)))})
    zpred=torch.cat(predicted_tokens,dim=0); ztarget=torch.cat(target_tokens,dim=0); centered=zpred-zpred.mean(0)
    cov=centered.T@centered/max(len(zpred)-1,1); eigen=torch.linalg.eigvalsh(cov).clamp_min(1e-12); probability=eigen/eigen.sum()
    sample_count=min(len(zpred),512); sample_index=torch.linspace(0,len(zpred)-1,sample_count).round().long(); normalized=F.normalize(zpred[sample_index],dim=-1)
    pairwise=normalized@normalized.T; off_diagonal=~torch.eye(sample_count,dtype=bool)
    representation={"category":category,"latent_variance":float(zpred.var(0,unbiased=False).mean()),"covariance_eigenvalues":eigen.cpu().numpy().tolist(),
                    "effective_rank":float(torch.exp(-(probability*torch.log(probability)).sum())),"covariance_condition_number":float(eigen.max()/eigen.min()),
                    "near_zero_variance_fraction":float((zpred.std(0)<1e-3).float().mean()),"pairwise_cosine_mean":float(pairwise[off_diagonal].mean()) if sample_count>1 else np.nan,
                    "predictor_target_cosine":float(F.cosine_similarity(zpred,ztarget).abs().mean()),
                    "predictor_target_cosine_values":F.cosine_similarity(zpred,ztarget).abs().cpu().numpy().tolist(),
                    "inference_latency_ms_per_hamiltonian":1000*inference_seconds/max(len(rows),1),
                    "inference_throughput_hamiltonians_per_second":len(rows)/max(inference_seconds,1e-12),
                    "excluded_target_encoding_ms_per_hamiltonian":1000*target_seconds/max(len(rows),1),"peak_inference_memory_bytes":int(peak_device_memory if DEVICE.type=="cuda" else peak_memory)}
    representation.update({"refinement_attempts":int(refinement_attempts),"refinement_accepted_steps":int(refinement_accepts),
                           "refinement_acceptance_fraction":float(refinement_accepts/max(refinement_attempts,1)),
                           "refinement_fallback_count":int(refinement_fallbacks),"refinement_iterations_used_mean":float(np.mean(refinement_iterations)),
                           "refinement_initial_block_residual_mean":float(np.mean(refinement_initial_blocks)),
                           "refinement_final_block_residual_mean":float(np.mean(refinement_final_blocks)),
                           "refinement_relative_block_residual_reduction":float((np.mean(refinement_initial_blocks)-np.mean(refinement_final_blocks))/max(np.mean(refinement_initial_blocks),1e-12))})
    representation["memory_measurement"]="cuda_max_memory_allocated" if DEVICE.type=="cuda" else "python_tracemalloc_peak_native_torch_allocations_excluded"
    return pd.DataFrame(metric_rows),representation


id_metrics,id_representation=evaluate_collection(model,splits["test"],"ID_test")
print(id_metrics.groupby("state")[["fidelity","energy_abs_error","independent_residual_head","raw_node_correct"]].mean().round(5))

# %% [notebook cell 67]
PARAMETER_OOD_CATEGORIES=("within_range_unseen_parameter_interpolation","single_parameter_extrapolation","joint_parameter_extrapolation","near_continuum_states")
parameter_metric_frames=[]; parameter_representations=[]
for category in PARAMETER_OOD_CATEGORIES:
    frame,rep=evaluate_collection(model,ood_suites[category],category); parameter_metric_frames.append(frame); parameter_representations.append(rep)
parameter_ood_metrics=pd.concat(parameter_metric_frames,ignore_index=True)

# %% [notebook cell 69]
FUNCTIONAL_OOD_CATEGORIES=("unseen_analytical_functional_forms","seen_basis_reparameterization","random_smooth_potentials","anharmonic_wells","asymmetric_wells","double_wells","multiwell_potentials","barriers","multiple_barriers","localized_defects",
                           "variable_uniform_resolution","nonuniform_resolution","changed_domain_same_physical_operator","controlled_smooth_perturbations",
                           "constant_potential_shift","valid_spatial_reflection")
functional_metric_frames=[]; functional_representations=[]
for category in FUNCTIONAL_OOD_CATEGORIES:
    frame,rep=evaluate_collection(model,ood_suites[category],category); functional_metric_frames.append(frame); functional_representations.append(rep)
functional_ood_metrics=pd.concat(functional_metric_frames,ignore_index=True)
all_metrics=pd.concat([id_metrics,parameter_ood_metrics,functional_ood_metrics],ignore_index=True)
representation_metrics=pd.DataFrame([id_representation,*parameter_representations,*functional_representations])
refinement_iteration_metrics=pd.DataFrame(REFINEMENT_ITERATION_ROWS)
per_state_output_metrics=pd.DataFrame(OUTPUT_ROUTE_ROWS)
topology_metrics=pd.DataFrame(TOPOLOGY_HAMILTONIAN_ROWS)
TOPOLOGY_METRIC_COLUMNS=("category","group_id","family","geometry","state","target_state_valid","binding_margin",
    "phase_crossing_count","exact_node_count_compliance","persistent_node_count_compliance","node_position_mae",
    "node_position_mae_normalized","worst_node_position_error","lobe_probability_mass_mae","minimum_predicted_lobe_probability_mass",
    "minimum_predicted_nodal_spacing","minimum_grid_intervals_per_lobe","maximum_phase_increment","p95_phase_increment",
    "phase_density_max_min_ratio","phase_density_smoothness","amplitude_dynamic_range","topology_proposals_attempted",
    "topology_proposals_rejected","topology_rejection_fraction","hybrid_fallback","initial_ritz_topology_valid","hybrid_topology_valid")
if topology_metrics.empty: topology_metrics=pd.DataFrame(columns=TOPOLOGY_METRIC_COLUMNS)
TOPOLOGY_METRICS_AVAILABLE=not topology_metrics.empty
POTENTIAL_CLASS={"double_well":"double_well","multiwell":"multiwell","single_barrier":"barrier","multiple_barriers":"barrier",
                 "gaussian_barrier":"barrier","localized_defect":"localized_defect"}
per_state_output_metrics["potential_class"]=per_state_output_metrics.family.map(POTENTIAL_CLASS).fillna("other")
topology_metrics["potential_class"]=topology_metrics.family.map(POTENTIAL_CLASS).fillna("other")
topology_metrics["near_continuum_margin_role"]=pd.cut(pd.to_numeric(topology_metrics.binding_margin,errors="coerce"),[-np.inf,.05,.20,np.inf],labels=["near","moderate","well_bound"])
per_state_output_metrics.to_csv(OUT/"per_state_output_metrics.csv",index=False)
topology_metrics.to_csv(OUT/"topology_metrics.csv",index=False)
route_pivot=per_state_output_metrics.pivot_table(index=["category","group_id","family","geometry","state"],columns="output_route",
                                                 values=["fidelity","independent_residual_head","gram_max_error","persistent_node_correct"]).reset_index()
route_pivot.columns=["_".join(str(part) for part in column if str(part)) if isinstance(column,tuple) else str(column) for column in route_pivot.columns]
MAIN_DIRECT_ROUTE="direct_topology" if model.decoder_type=="topology_phase" else "direct_spectral"
for metric in ("fidelity","independent_residual_head","gram_max_error"):
    if f"{metric}_{MAIN_DIRECT_ROUTE}" in route_pivot and f"{metric}_hybrid" in route_pivot:
        route_pivot[f"hybrid_minus_direct_{metric}"]=route_pivot[f"{metric}_hybrid"]-route_pivot[f"{metric}_{MAIN_DIRECT_ROUTE}"]
route_pivot.to_csv(OUT/"direct_hybrid_comparison.csv",index=False)
ROUTE_SUMMARY_METRICS=("fidelity","h1_relative","energy_abs_error","gap_error","independent_residual_head",
                       "density_integrated_absolute_error","persistent_node_correct","node_position_mae_normalized",
                       "gram_max_error","boundary_error")
def output_route_summary(frame:pd.DataFrame,group_columns:Sequence[str])->pd.DataFrame:
    rows=[]
    for key,group in frame.groupby(list(group_columns),observed=True,dropna=False):
        keys=key if isinstance(key,tuple) else (key,); row=dict(zip(group_columns,keys)); row["hamiltonians"]=group.group_id.nunique()
        for metric in ROUTE_SUMMARY_METRICS:
            values=group[metric].dropna()
            if len(values):
                lower_worse=metric in {"fidelity","persistent_node_correct"}
                row.update({f"{metric}_mean":float(values.mean()),f"{metric}_median":float(values.median()),
                            f"{metric}_p90":float(values.quantile(.90)),f"{metric}_p95":float(values.quantile(.95)),
                            f"{metric}_worst":float(values.min() if lower_worse else values.max())})
        rows.append(row)
    return pd.DataFrame(rows)
output_route_summary(per_state_output_metrics,["category","output_route"]).to_csv(OUT/"output_route_summary_by_category.csv",index=False)
output_route_summary(per_state_output_metrics,["category","output_route","state"]).to_csv(OUT/"output_route_summary_by_state.csv",index=False)
output_route_summary(per_state_output_metrics,["category","output_route","family"]).to_csv(OUT/"output_route_summary_by_family.csv",index=False)
output_route_summary(per_state_output_metrics,["category","output_route","geometry"]).to_csv(OUT/"output_route_summary_by_geometry.csv",index=False)

TOPOLOGY_SUMMARY_METRICS=("exact_node_count_compliance","persistent_node_count_compliance","node_position_mae_normalized",
                          "worst_node_position_error","lobe_probability_mass_mae","minimum_predicted_lobe_probability_mass","minimum_predicted_nodal_spacing","minimum_grid_intervals_per_lobe",
                          "maximum_phase_increment","p95_phase_increment","phase_density_max_min_ratio","phase_density_smoothness",
                          "amplitude_dynamic_range","topology_rejection_fraction","hybrid_fallback")
TOPOLOGY_LOWER_IS_WORSE={"exact_node_count_compliance","persistent_node_count_compliance","minimum_predicted_nodal_spacing",
                         "minimum_predicted_lobe_probability_mass","minimum_grid_intervals_per_lobe","initial_ritz_topology_valid","hybrid_topology_valid"}
def topology_group_summary(frame:pd.DataFrame,group_columns:Sequence[str])->pd.DataFrame:
    rows=[]
    for key,group in frame.groupby(list(group_columns),observed=True,dropna=False):
        keys=key if isinstance(key,tuple) else (key,); row=dict(zip(group_columns,keys)); row["hamiltonians"]=group.group_id.nunique()
        for metric in TOPOLOGY_SUMMARY_METRICS:
            values=group[metric].dropna()
            if len(values):
                row.update({f"{metric}_mean":float(values.mean()),f"{metric}_median":float(values.median()),
                            f"{metric}_p90":float(values.quantile(.90)),f"{metric}_p95":float(values.quantile(.95)),
                            f"{metric}_worst":float(values.min() if metric in TOPOLOGY_LOWER_IS_WORSE else values.max())})
        fidelity_by_h=group.groupby("group_id").node_position_mae_normalized.mean().dropna(); row["worst_hamiltonian"]=str(fidelity_by_h.idxmax()) if len(fidelity_by_h) else ""
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[*group_columns,"hamiltonians","worst_hamiltonian"])
topology_summary_by_category=topology_group_summary(topology_metrics,["category"]); topology_summary_by_category.to_csv(OUT/"topology_summary_by_category.csv",index=False)
topology_group_summary(topology_metrics,["state"]).to_csv(OUT/"topology_summary_by_state.csv",index=False)
topology_group_summary(topology_metrics,["family"]).to_csv(OUT/"topology_summary_by_family.csv",index=False)
topology_group_summary(topology_metrics,["geometry"]).to_csv(OUT/"topology_summary_by_geometry.csv",index=False)
topology_group_summary(topology_metrics,["near_continuum_margin_role"]).to_csv(OUT/"topology_summary_by_margin.csv",index=False)
topology_group_summary(topology_metrics,["potential_class"]).to_csv(OUT/"topology_summary_by_potential_class.csv",index=False)
topology_metrics[["category","group_id","state","topology_rejection_fraction","hybrid_fallback","initial_ritz_topology_valid","hybrid_topology_valid"]].to_csv(OUT/"topology_refinement_audit.csv",index=False)
_assert(len(refinement_iteration_metrics)>0 and np.isfinite(refinement_iteration_metrics[["training_relative_residual","independent_residual_head","fidelity","energy_abs_error","gram_max_error","boundary_error"]].to_numpy()).all(),
        "recurrent refinement trace is missing or contains non-finite core metrics")
executed_categories={entry["category"] for entry in OOD_EXPERIMENT_STATUS.values() if entry["status"].startswith("EXECUTED")}
_assert(executed_categories<=set(all_metrics["category"]),f"executed OOD status lacks metric evidence: {executed_categories-set(all_metrics['category'])}")
_assert(all_metrics.groupby("category").group_id.nunique().reindex(sorted(executed_categories)).min()>=1,"an executed OOD category has no Hamiltonian")
OOD_CORE_METRICS=("fidelity","h1_relative","energy_abs_error","independent_residual_head","raw_node_correct","gram_post_max_error")
ood_evidence=all_metrics[all_metrics.category.isin(executed_categories)]
OOD_EVIDENCE_COMPLETE=bool(len(ood_evidence)>0 and np.isfinite(ood_evidence[list(OOD_CORE_METRICS)].to_numpy()).all())
_assert(OOD_EVIDENCE_COMPLETE,"executed OOD/challenge metrics contain non-finite core values")
CATEGORY_ROLES={"ID_test":"untouched_id"}
for entry in OOD_EXPERIMENT_STATUS.values():
    if "category" not in entry: continue
    status=entry["status"]; category=entry["category"]
    if category=="seen_basis_reparameterization": role="seen_basis_reparameterization_diagnostic"
    elif category=="unseen_analytical_functional_forms": role="unseen_functional_form_ood"
    elif "TEST_SLICE" in status or "SAME_FAMILY" in status: role="same_training_family_challenge"
    elif category=="within_range_unseen_parameter_interpolation": role="bracketed_parameter_interpolation"
    elif category in ("single_parameter_extrapolation","joint_parameter_extrapolation"): role="parameter_extrapolation_ood"
    elif category=="near_continuum_states": role="near_continuum_challenge"
    else: role="operator_covariance_or_resolution_challenge"
    CATEGORY_ROLES[category]=role
all_metrics["distribution_role"]=all_metrics.category.map(CATEGORY_ROLES).fillna("documented_challenge")
per_state_output_metrics["distribution_role"]=per_state_output_metrics.category.map(CATEGORY_ROLES).fillna("documented_challenge")
topology_metrics["distribution_role"]=topology_metrics.category.map(CATEGORY_ROLES).fillna("documented_challenge")
per_state_output_metrics.to_csv(OUT/"per_state_output_metrics.csv",index=False)
topology_metrics.to_csv(OUT/"topology_metrics.csv",index=False)
output_route_summary(per_state_output_metrics,["distribution_role","output_route"]).to_csv(OUT/"output_route_summary_by_distribution_role.csv",index=False)
topology_group_summary(topology_metrics,["distribution_role"]).to_csv(OUT/"topology_summary_by_distribution_role.csv",index=False)

# Reflection is an evaluation diagnostic in addition to the decoder-level algebra test.
# It compares two independent full-model inference calls and never enters training.
reflection_topology_rows=[]; id_record_by_group={record["group_id"]:record for record in splits["test"]}
if TOPOLOGY_METRICS_AVAILABLE:
    for reflected_record_item in ood_suites["valid_spatial_reflection"]:
        parent_id=reflected_record_item.get("augmentation_parent")
        if parent_id not in id_record_by_group or ("ID_test",parent_id) not in PREDICTION_CACHE: continue
        reflected_key=("valid_spatial_reflection",reflected_record_item["group_id"])
        if reflected_key not in PREDICTION_CACHE: continue
        parent=id_record_by_group[parent_id]; base_prediction=PREDICTION_CACHE[("ID_test",parent_id)]; reflected_prediction=PREDICTION_CACHE[reflected_key]
        x_ref=np.asarray(reflected_record_item["x"]); w_ref=np.asarray(reflected_record_item["quadrature_weights"]); x_parent=np.asarray(parent["x"])
        mirrored_coordinate=float(parent["domain_left"])+float(parent["domain_right"])-x_ref
        for state in range(CFG.k_states):
            expected_amplitude=np.interp(mirrored_coordinate,x_parent,base_prediction["amplitude"][state])
            expected_cdf=1-np.interp(mirrored_coordinate,x_parent,base_prediction["phase_cdf"][state])
            expected_theta=(state+1)*math.pi-np.interp(mirrored_coordinate,x_parent,base_prediction["theta"][state])
            expected_wave=(-1.0 if state%2 else 1.0)*np.interp(mirrored_coordinate,x_parent,base_prediction["psi_direct"][state])
            actual_wave=reflected_prediction["psi_direct"][state]; overlap=float(np.sum(w_ref*expected_wave*actual_wave))
            reflection_topology_rows.append({"group_id":reflected_record_item["group_id"],"parent_group_id":parent_id,"state":state,
                "amplitude_relative_l2":float(np.linalg.norm(reflected_prediction["amplitude"][state]-expected_amplitude)/max(np.linalg.norm(expected_amplitude),1e-30)),
                "phase_cdf_linf":float(np.max(np.abs(reflected_prediction["phase_cdf"][state]-expected_cdf))),
                "theta_over_pi_linf":float(np.max(np.abs(reflected_prediction["theta"][state]-expected_theta))/math.pi),
                "waveform_sign_invariant_fidelity":overlap**2})
reflection_topology_metrics=pd.DataFrame(reflection_topology_rows,columns=["group_id","parent_group_id","state","amplitude_relative_l2","phase_cdf_linf","theta_over_pi_linf","waveform_sign_invariant_fidelity"])
reflection_topology_metrics.to_csv(OUT/"reflection_topology_consistency.csv",index=False)


def bootstrap_hamiltonians(frame:pd.DataFrame,column:str,repeats:int=CFG.bootstrap_repeats)->dict[str,float]:
    values=frame.groupby("group_id")[column].mean().dropna().to_numpy(); rng=np.random.default_rng(CFG.seed)
    samples=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(repeats)])
    return {"mean":float(values.mean()),"median":float(np.median(values)),"p95":float(np.quantile(values,.95)),"ci95_low":float(np.quantile(samples,.025)),"ci95_high":float(np.quantile(samples,.975))}


def bootstrap_energy_rmse(frame:pd.DataFrame,repeats:int=CFG.bootstrap_repeats)->dict[str,float]:
    values=frame.groupby("group_id").energy_squared_error.mean().dropna().to_numpy(); rng=np.random.default_rng(CFG.seed+31)
    samples=np.sqrt(np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(repeats)]))
    per_hamiltonian=np.sqrt(values)
    return {"mean":float(np.sqrt(values.mean())),"median":float(np.median(per_hamiltonian)),"p95":float(np.quantile(per_hamiltonian,.95)),
            "ci95_low":float(np.quantile(samples,.025)),"ci95_high":float(np.quantile(samples,.975))}


summary_rows=[]
for category,frame in all_metrics.groupby("category"):
    row={"category":category,"hamiltonians":frame.group_id.nunique()}
    for column in ("fidelity","energy_abs_error","independent_residual_head","raw_node_correct","gram_post_max_error","density_integrated_absolute_error"):
        row.update({f"{column}_{key}":value for key,value in bootstrap_hamiltonians(frame,column).items()})
    row.update({f"energy_rmse_{key}":value for key,value in bootstrap_energy_rmse(frame).items()})
    summary_rows.append(row)
generalization_summary=pd.DataFrame(summary_rows)
family_summary=all_metrics.groupby(["category","family"]).agg(initial_fidelity=("initial_ritz_fidelity","mean"),fidelity=("fidelity","mean"),fidelity_gain=("refinement_fidelity_gain","mean"),energy_mae=("energy_abs_error","mean"),initial_residual=("initial_ritz_independent_residual_head","mean"),residual=("independent_residual_head","mean"),residual_reduction=("refinement_independent_residual_reduction","mean"),raw_node_accuracy=("raw_node_correct","mean")).reset_index()
state_summary=all_metrics.groupby(["category","state"]).agg(fidelity=("fidelity","mean"),energy_mae=("energy_abs_error","mean"),residual=("independent_residual_head","mean"),raw_node_accuracy=("raw_node_correct","mean")).reset_index()
family_state_summary=all_metrics.groupby(["category","family","state"])[["fidelity","energy_abs_error","independent_residual_head","raw_node_correct"]].mean().reset_index()
distribution_role_summary=all_metrics.groupby("distribution_role").agg(hamiltonians=("group_id","nunique"),fidelity=("fidelity","mean"),energy_mae=("energy_abs_error","mean"),residual=("independent_residual_head","mean"),raw_node_accuracy=("raw_node_correct","mean")).reset_index()
geometry_summary=all_metrics.groupby(["category","geometry"]).agg(hamiltonians=("group_id","nunique"),fidelity=("fidelity","mean"),h1=("h1_relative","mean"),energy_mae=("energy_abs_error","mean"),residual=("independent_residual_head","mean"),raw_node_accuracy=("raw_node_correct","mean")).reset_index()
finite_margin=all_metrics.binding_margin.replace([np.inf,-np.inf],np.nan); margin_labels=pd.cut(finite_margin,[-np.inf,.1,.5,2,np.inf],labels=["<=0.1","0.1-0.5","0.5-2",">2"])
all_metrics["binding_margin_regime"]=margin_labels.astype("object"); all_metrics.loc[~np.isfinite(all_metrics.binding_margin),"binding_margin_regime"]="no_finite_threshold"
binding_margin_summary=all_metrics.groupby(["category","binding_margin_regime"],dropna=False).agg(hamiltonians=("group_id","nunique"),fidelity=("fidelity","mean"),h1=("h1_relative","mean"),energy_mae=("energy_abs_error","mean"),residual=("independent_residual_head","mean")).reset_index()
orth_keys=["category","group_id","family","geometry","state"]
direct_orth_columns=orth_keys+["fidelity","h1_relative","raw_node_correct","persistent_node_correct","node_position_mae_normalized",
                               "independent_residual_head","independent_residual_rayleigh","rayleigh_energy","head_rayleigh_disagreement","gram_max_error"]
direct_orth=per_state_output_metrics[per_state_output_metrics.output_route==MAIN_DIRECT_ROUTE][direct_orth_columns].copy()
ritz_orth=per_state_output_metrics[per_state_output_metrics.output_route=="initial_ritz"][direct_orth_columns].copy()
orthonormalization_comparison=direct_orth.merge(ritz_orth,on=orth_keys,suffixes=("_direct","_initial_ritz"))
orthonormalization_comparison=orthonormalization_comparison.rename(columns={"gram_max_error_direct":"gram_pre_max_error",
    "gram_max_error_initial_ritz":"gram_post_max_error"})
orth_audit=all_metrics[["category","group_id","state","pre_gram_min_eigenvalue","lowdin_state_displacement"]].copy()
orthonormalization_comparison=orthonormalization_comparison.merge(orth_audit,on=["category","group_id","state"],how="left")
_assert(np.isfinite(orthonormalization_comparison[["gram_pre_max_error","gram_post_max_error","pre_gram_min_eigenvalue","lowdin_state_displacement"]].to_numpy()).all())
_assert(float((orthonormalization_comparison.gram_post_max_error-orthonormalization_comparison.gram_pre_max_error).max())<2e-5,
        "Löwdin audit worsened the weighted Gram matrix")
id_hamiltonian_ranking=id_metrics.groupby(["group_id","family"]).agg(fidelity=("fidelity","mean"),energy_rmse=("energy_squared_error",lambda s:float(np.sqrt(s.mean()))),
    residual=("independent_residual_head","mean"),node_accuracy=("raw_node_correct","mean")).reset_index()
worst_hamiltonians=pd.concat([
    id_hamiltonian_ranking.nsmallest(min(5,len(id_hamiltonian_ranking)),"fidelity").assign(ranked_by="lowest_fidelity"),
    id_hamiltonian_ranking.nlargest(min(5,len(id_hamiltonian_ranking)),"energy_rmse").assign(ranked_by="highest_energy_rmse"),
    id_hamiltonian_ranking.nlargest(min(5,len(id_hamiltonian_ranking)),"residual").assign(ranked_by="highest_residual"),
    id_hamiltonian_ranking.nsmallest(min(5,len(id_hamiltonian_ranking)),"node_accuracy").assign(ranked_by="lowest_node_accuracy")],ignore_index=True)
all_metrics.to_csv(OUT/"per_state_metrics.csv",index=False); refinement_iteration_metrics.to_csv(OUT/"refinement_iteration_metrics.csv",index=False); generalization_summary.to_csv(OUT/"generalization_summary.csv",index=False); family_summary.to_csv(OUT/"family_summary.csv",index=False); state_summary.to_csv(OUT/"state_summary.csv",index=False); family_state_summary.to_csv(OUT/"family_state_summary.csv",index=False); distribution_role_summary.to_csv(OUT/"distribution_role_summary.csv",index=False); geometry_summary.to_csv(OUT/"geometry_summary.csv",index=False); binding_margin_summary.to_csv(OUT/"binding_margin_summary.csv",index=False); orthonormalization_comparison.to_csv(OUT/"orthonormalization_comparison.csv",index=False); worst_hamiltonians.to_csv(OUT/"worst_hamiltonians.csv",index=False); representation_metrics.to_json(OUT/"representation_metrics.json",orient="records",indent=2)
record_by_group={r["group_id"]:r for r in records}
def ood_manifest_row(category:str,record:Mapping[str,Any])->dict[str,Any]:
    parent=record_by_group.get(record.get("augmentation_parent")); shift=None; domain_factor=None
    if category=="constant_potential_shift" and parent is not None and len(parent["V_raw"])==len(record["V_raw"]): shift=float(np.median(np.asarray(record["V_raw"])-np.asarray(parent["V_raw"])))
    if parent is not None: domain_factor=float((record["domain_right"]-record["domain_left"])/(parent["domain_right"]-parent["domain_left"]))
    return {"category":category,"group_id":record["group_id"],"augmentation_parent":record.get("augmentation_parent"),"family":record["family"],"parameters":record["parameters"],
            "geometry":record["geometry"],"boundary_condition":record["boundary_condition"],"domain":[record["domain_left"],record["domain_right"]],"domain_factor_from_parent":domain_factor,
            "grid_nodes":len(record["x"]),"grid_variant":record.get("model_grid_diagnostics",{}).get("ood_grid_variant"),"constant_shift_from_parent":shift,
            "continuum_threshold":record["continuum_threshold"],"valid_states":int(np.asarray(record["valid_state_mask"]).sum()),
            "potential_fingerprint":record["potential_fingerprint"],"affine_fingerprint":record["affine_fingerprint"],"reflection_fingerprint":record["reflection_fingerprint"],"sampled_array_fingerprint":record["sampled_array_fingerprint"],
            "x_sha256":hashlib.sha256(np.asarray(record["x"],dtype="<f8").tobytes()).hexdigest(),"V_sha256":hashlib.sha256(np.asarray(record["V_raw"],dtype="<f8").tobytes()).hexdigest(),"label_checksum":record_label_checksum(record)}
(OUT/"ood_suite_manifest.json").write_text(json.dumps({category:[ood_manifest_row(category,r) for r in rows] for category,rows in ood_suites.items()},indent=2,default=float))
print(generalization_summary[[c for c in generalization_summary if c in ("category","hamiltonians","fidelity_mean","energy_abs_error_mean","independent_residual_head_mean","raw_node_correct_mean")]].round(5).to_string(index=False))

# %% [notebook cell 71]
def deployable_parameter_count(candidate:TrueSchrodingerJEPA)->int:
    modules=[candidate.potential_context_encoder,candidate.predictor,candidate.wavefunction_decoder,candidate.energy_head]
    if candidate.learned_refinement: modules.append(candidate.residual_correction)
    return sum(p.numel() for p in {id(p):p for module in modules for p in module.parameters()}.values())


def run_ablation_smoke(name:str,candidate:TrueSchrodingerJEPA,batch:Mapping[str,Any],weights:Mapping[str,float],route:str="post_lowdin",
                       refinement_steps:int|None=None)->dict[str,Any]:
    b=to_device(batch); candidate=candidate.to(DEVICE).train(); params=trainable_without_target(candidate)
    target_before={key:value.detach().clone() for key,value in candidate.solution_target_encoder.state_dict().items()}
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=CFG.weight_decay); optimizer.zero_grad(set_to_none=True)
    if DEVICE.type=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    tracemalloc.start(); started=time.perf_counter(); out=candidate.forward_training(b,refinement_steps=refinement_steps)
    if route=="loss_only":
        out=dict(out); out["psi"]=out["psi_pre_orth"]; out["rho"]=out["psi"].square(); out["rayleigh_energy"]=training_rayleigh(out["psi"],b)
    loss,terms=compute_losses(out,b,weights,loss_model=candidate); loss.backward()
    gradient_norm=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None))
    optimizer.step(); parameters_finite=all(torch.isfinite(p).all() for p in candidate.parameters())
    if DEVICE.type=="cuda": torch.cuda.synchronize()
    elapsed=time.perf_counter()-started; _,peak_host=tracemalloc.get_traced_memory(); tracemalloc.stop()
    peak_device=int(torch.cuda.max_memory_allocated()) if DEVICE.type=="cuda" else int(peak_host)
    _assert(torch.isfinite(loss) and math.isfinite(gradient_norm) and gradient_norm>0 and parameters_finite,f"non-finite/zero-gradient ablation {name}")
    aligned,fidelity=globally_align_torch(out["psi"],b["psi"],b["w"]); psi_np=out["psi"][0].detach().cpu().numpy(); record=b["operator_records"][0]
    gram_error=float(np.max(np.abs(weighted_gram(psi_np,np.asarray(record["quadrature_weights"]))-np.eye(CFG.k_states))))
    pre_psi_np=out["psi_pre_orth"][0].detach().cpu().numpy(); pre_gram_error=float(np.max(np.abs(weighted_gram(pre_psi_np,np.asarray(record["quadrature_weights"]))-np.eye(CFG.k_states))))
    node_accuracy=float(np.mean([persistent_nodes(row,1e-8)==state for state,row in enumerate(psi_np)]))
    residual=float(np.mean(independent_residual(np.asarray(record["x"]),np.asarray(record["quadrature_weights"]),np.asarray(record["V_raw"]),psi_np,out["energy"][0].detach().cpu().numpy())))
    candidate.eval()
    with torch.no_grad():
        post=candidate.forward_training(b,refinement_steps=refinement_steps)
        if route=="loss_only": post=dict(post); post["psi"]=post["psi_pre_orth"]; post["rho"]=post["psi"].square(); post["rayleigh_energy"]=training_rayleigh(post["psi"],b)
        post_loss,_=compute_losses(post,b,weights,loss_model=candidate); _,post_fidelity=globally_align_torch(post["psi"],b["psi"],b["w"])
    post_psi_np=post["psi"][0].cpu().numpy(); post_pre_np=post["psi_pre_orth"][0].cpu().numpy()
    post_gram_error=float(np.max(np.abs(weighted_gram(post_psi_np,np.asarray(record["quadrature_weights"]))-np.eye(CFG.k_states))))
    post_pre_gram_error=float(np.max(np.abs(weighted_gram(post_pre_np,np.asarray(record["quadrature_weights"]))-np.eye(CFG.k_states))))
    post_node_accuracy=float(np.mean([persistent_nodes(row,1e-8)==state for state,row in enumerate(post_psi_np)]))
    post_residual=float(np.mean(independent_residual(np.asarray(record["x"]),np.asarray(record["quadrature_weights"]),np.asarray(record["V_raw"]),post_psi_np,post["energy"][0].cpu().numpy())))
    finite=bool(torch.isfinite(loss) and torch.isfinite(post_loss) and np.isfinite([gradient_norm,gram_error,residual,post_gram_error,post_residual]).all() and parameters_finite)
    _assert(finite,f"post-step validation failed for {name}")
    _assert(all(p.grad is None for p in candidate.solution_target_encoder.parameters()),f"target received gradients in {name}")
    _assert(all(torch.equal(value,target_before[key]) for key,value in candidate.solution_target_encoder.state_dict().items()),f"optimizer changed target in {name}")
    return {"smoke":name,"route":route,"decoder_type":candidate.decoder_type,"primary_output":candidate.primary_output,
            "topology_safeguard":candidate.topology_safeguard,"parameter_matched":False,"comparison_status":"ONE_STEP_WIRING_SMOKE_MATCHED_TRAINING_PENDING",
            "parameters":deployable_parameter_count(candidate),"total_parameters":sum(p.numel() for p in candidate.parameters()),"trainable_parameters":sum(p.numel() for p in candidate.parameters() if p.requires_grad),"inference_parameters":deployable_parameter_count(candidate),"one_step_seconds":elapsed,"peak_memory_bytes":peak_device,"memory_measurement":"cuda_max_memory_allocated" if DEVICE.type=="cuda" else "python_tracemalloc_peak_native_torch_allocations_excluded",
            "pre_step_loss":float(loss.detach()),"post_step_loss":float(post_loss),"gradient_norm":gradient_norm,"initial_fidelity":float(fidelity.mean()),"post_step_fidelity":float(post_fidelity.mean()),"pre_step_raw_node_accuracy":node_accuracy,"post_step_raw_node_accuracy":post_node_accuracy,
            "pre_step_independent_residual":residual,"post_step_independent_residual":post_residual,"pre_step_pre_gram_max_error":pre_gram_error,"pre_step_gram_max_error":gram_error,"post_step_pre_gram_max_error":post_pre_gram_error,"post_step_gram_max_error":post_gram_error,"finite":finite,"ema_target_frozen":True,
            **{f"term_{key}":float(value.detach()) for key,value in terms.items() if value.ndim==0}}


primary_smoke_batch=collate_records([splits["train"][0]],include_targets=True)
supervised_weights={"wave":1.0,"infidelity":.1,"density":.15,"energy":.25,"gap":.1}
jepa_weights={"jepa":1.0,"wave":.2,"infidelity":.05,"energy":.1}
physics_wiring_weights={**jepa_weights,"residual":.05,"rayleigh":.04,"orthogonality":.05,"tail":.01}
topology_smoke_weights={**supervised_weights,"topology_node_phase":CFG.topology_node_loss_weight,
                        "topology_lobe_mass":CFG.topology_lobe_mass_weight,
                        "topology_phase_cdf_supervision":CFG.topology_phase_cdf_loss_weight,
                        "topology_log_amplitude_supervision":CFG.topology_log_amplitude_loss_weight,
                        "topology_phase_smoothness":CFG.topology_phase_smoothness_weight,
                        "topology_phase_resolution":CFG.topology_phase_resolution_weight,
                        "topology_amplitude_smoothness":CFG.topology_amplitude_smoothness_weight}
ablation_smoke_rows=[]
for backbone in BACKBONE_OPTIONS:
    ablation_smoke_rows.append(run_ablation_smoke(f"backbone_{backbone}",TrueSchrodingerJEPA(backbone=backbone),primary_smoke_batch,supervised_weights))
ablation_smoke_rows.extend([
    run_ablation_smoke("decoder_spectral_sine_residual",TrueSchrodingerJEPA(decoder_type="spectral_sine_residual",primary_output="hybrid"),primary_smoke_batch,supervised_weights),
    run_ablation_smoke("decoder_topology_phase",TrueSchrodingerJEPA(decoder_type="topology_phase",primary_output="direct_topology"),primary_smoke_batch,topology_smoke_weights),
    run_ablation_smoke("topology_without_node_phase_loss",TrueSchrodingerJEPA(),primary_smoke_batch,{**topology_smoke_weights,"topology_node_phase":0.0}),
    run_ablation_smoke("topology_without_lobe_mass_loss",TrueSchrodingerJEPA(),primary_smoke_batch,{**topology_smoke_weights,"topology_lobe_mass":0.0}),
    run_ablation_smoke("topology_without_phase_cdf_supervision",TrueSchrodingerJEPA(),primary_smoke_batch,{**topology_smoke_weights,"topology_phase_cdf_supervision":0.0}),
    run_ablation_smoke("topology_without_log_amplitude_supervision",TrueSchrodingerJEPA(),primary_smoke_batch,{**topology_smoke_weights,"topology_log_amplitude_supervision":0.0}),
    run_ablation_smoke("topology_without_phase_resolution",TrueSchrodingerJEPA(),primary_smoke_batch,{**topology_smoke_weights,"topology_phase_resolution":0.0}),
    run_ablation_smoke("topology_direct_output",TrueSchrodingerJEPA(primary_output="direct_topology"),primary_smoke_batch,topology_smoke_weights),
    run_ablation_smoke("topology_hybrid_output",TrueSchrodingerJEPA(primary_output="hybrid"),primary_smoke_batch,topology_smoke_weights),
    run_ablation_smoke("hybrid_with_topology_safeguard",TrueSchrodingerJEPA(primary_output="hybrid",topology_safeguard=True),primary_smoke_batch,topology_smoke_weights),
    run_ablation_smoke("hybrid_without_topology_safeguard",TrueSchrodingerJEPA(primary_output="hybrid",topology_safeguard=False),primary_smoke_batch,topology_smoke_weights),
    run_ablation_smoke("direct_supervised_without_jepa",TrueSchrodingerJEPA(),primary_smoke_batch,supervised_weights),
    run_ablation_smoke("jepa_without_physics",TrueSchrodingerJEPA(),primary_smoke_batch,jepa_weights),
    run_ablation_smoke("jepa_with_physics_wiring",TrueSchrodingerJEPA(),primary_smoke_batch,physics_wiring_weights),
    run_ablation_smoke("local_graph_only",TrueSchrodingerJEPA(use_global_context=False),collate_records([splits["train"][0]],True,dilations=(1,)),jepa_weights),
    run_ablation_smoke("multiscale_graph_no_global",TrueSchrodingerJEPA(use_global_context=False),collate_records([splits["train"][0]],True,dilations=CFG.graph_dilations),jepa_weights),
    run_ablation_smoke("multiscale_plus_global",TrueSchrodingerJEPA(use_global_context=True),primary_smoke_batch,jepa_weights),
    run_ablation_smoke("orthogonality_loss_only",TrueSchrodingerJEPA(),primary_smoke_batch,{**supervised_weights,"orthogonality":.05},route="loss_only"),
    run_ablation_smoke("regularized_lowdin",TrueSchrodingerJEPA(),primary_smoke_batch,supervised_weights),
    run_ablation_smoke("lowdin_plus_mild_loss",TrueSchrodingerJEPA(),primary_smoke_batch,{**supervised_weights,"orthogonality":.05}),
    run_ablation_smoke("energy_head_only",TrueSchrodingerJEPA(),primary_smoke_batch,{"energy":1.0,"gap":.2}),
    run_ablation_smoke("rayleigh_coupled_energy",TrueSchrodingerJEPA(),primary_smoke_batch,{"energy":1.0,"gap":.2,"rayleigh":.1}),
    run_ablation_smoke("refinement_t0_initializer",TrueSchrodingerJEPA(),primary_smoke_batch,physics_wiring_weights,refinement_steps=0),
    run_ablation_smoke("refinement_t1_hybrid",TrueSchrodingerJEPA(),primary_smoke_batch,{**physics_wiring_weights,"refinement_contraction":.10},refinement_steps=1),
    run_ablation_smoke("refinement_t3_classical",TrueSchrodingerJEPA(learned_refinement=False),primary_smoke_batch,physics_wiring_weights,refinement_steps=3),
    run_ablation_smoke("refinement_t3_hybrid",TrueSchrodingerJEPA(),primary_smoke_batch,{**physics_wiring_weights,"refinement_contraction":.10},refinement_steps=3),
    run_ablation_smoke("refinement_t5_hybrid",TrueSchrodingerJEPA(),primary_smoke_batch,{**physics_wiring_weights,"refinement_contraction":.10},refinement_steps=5),
])
threshold_record=thresholded_test[0]; threshold_batch=collate_records([threshold_record],True,use_continuum_threshold=True)
ablation_smoke_rows.append(run_ablation_smoke("continuum_threshold_assisted",TrueSchrodingerJEPA(in_features=threshold_batch["features"].shape[-1]),threshold_batch,jepa_weights))
ablation_smoke_rows.append(run_ablation_smoke("operator_only_no_threshold",TrueSchrodingerJEPA(),primary_smoke_batch,jepa_weights))
original_basis=CFG.kan_basis
for basis in KAN_BASIS_OPTIONS:
    CFG.kan_basis=basis
    candidate=TrueSchrodingerJEPA(backbone="gnn_kan")
    CFG.kan_basis=original_basis
    ablation_smoke_rows.append(run_ablation_smoke(f"gnn_kan_{basis}",candidate,primary_smoke_batch,jepa_weights))
CFG.kan_basis=original_basis
ablation_smokes=pd.DataFrame(ablation_smoke_rows); _assert(ablation_smokes.finite.all() and ablation_smokes.ema_target_frozen.all())
orth_smokes=ablation_smokes[ablation_smokes.smoke.isin(["orthogonality_loss_only","regularized_lowdin","lowdin_plus_mild_loss"])]
_assert(np.isfinite(orth_smokes[["initial_fidelity","post_step_fidelity","pre_step_raw_node_accuracy","post_step_raw_node_accuracy","pre_step_independent_residual","post_step_independent_residual","post_step_gram_max_error"]].to_numpy()).all())
_assert(bool((orth_smokes.loc[orth_smokes.smoke!="orthogonality_loss_only","post_step_gram_max_error"]<=orth_smokes.loc[orth_smokes.smoke!="orthogonality_loss_only","post_step_pre_gram_max_error"]+2e-5).all()),"regularized Löwdin smoke worsened the Gram matrix")
ablation_smokes.to_csv(OUT/"ablation_smoke_results.csv",index=False)
backbone_smokes=ablation_smokes[ablation_smokes.smoke.str.startswith("backbone_")].copy(); backbone_smokes.to_csv(OUT/"backbone_smoke_comparison.csv",index=False)
basis_smokes=[]
for basis in KAN_BASIS_OPTIONS:
    for degree in (3,5,7):
        for rank in (2,4,8):
            layer=FactorizedBasisKANLinear(8,6,degree,rank,basis); x=torch.randn(4,8,requires_grad=True); y=layer(x); y.square().mean().backward()
            basis_smokes.append({"basis":basis,"degree":degree,"rank":rank,"finite":bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all()),"parameters":sum(p.numel() for p in layer.parameters())})
basis_smokes=pd.DataFrame(basis_smokes); _assert(basis_smokes.finite.all()); basis_smokes.to_csv(OUT/"kan_basis_degree_rank_smokes.csv",index=False)


def run_refinement_depth_ablation(m:TrueSchrodingerJEPA)->pd.DataFrame:
    unseen_by_family=[]
    for family,group in pd.DataFrame([{"family":r["family"],"index":i} for i,r in enumerate(ood_suites["unseen_analytical_functional_forms"])]).groupby("family"):
        del family; unseen_by_family.append(ood_suites["unseen_analytical_functional_forms"][int(group.iloc[0]["index"])])
    selected=[splits["test"][0],*unseen_by_family]; depths=sorted({0,1,3,5}); rows=[]; original=m.learned_refinement
    try:
        for variant,learned in (("classical",False),("hybrid",True)):
            m.learned_refinement=learned; m.eval()
            for depth in depths:
                for record in selected:
                    b=to_device(collate_records([operator_only_record(record)],include_targets=False))
                    if DEVICE.type=="cuda": torch.cuda.synchronize()
                    started=time.perf_counter()
                    with torch.no_grad(): out=m.forward_operator(b,refinement_steps=depth)
                    if DEVICE.type=="cuda": torch.cuda.synchronize()
                    elapsed=time.perf_counter()-started; psi=out["psi_hybrid"][0].cpu().numpy(); energy=out["energy_hybrid"][0].cpu().numpy()
                    exact=np.asarray(record["psi"]); weights=np.asarray(record["quadrature_weights"]); overlap=np.sum(weights[None]*psi*exact,axis=1)
                    residual=independent_residual(np.asarray(record["x"]),weights,np.asarray(record["V_raw"]),psi,energy)
                    rows.append({"variant":variant,"requested_depth":depth,"executed_iterations":int(out["refinement_iterations_used"][0].cpu()),
                                 "group_id":record["group_id"],"family":record["family"],"evaluation_role":"ID" if record is selected[0] else "unseen_functional_form",
                                 "fidelity":float(np.mean(overlap**2)),"energy_mae":float(np.mean(np.abs(energy-np.asarray(record["energy"])))),
                                 "independent_residual":float(np.mean(residual)),"training_block_residual":float(torch.sqrt(out["refinement_residual_trace"][0,-1].square().mean()).cpu()),
                                 "accepted_steps":int(out["refinement_acceptance_trace"][0].sum().cpu()),"fallback_recommended":bool(out["refinement_fallback_recommended"][0].cpu()),
                                 "seconds":elapsed})
    finally: m.learned_refinement=original
    return pd.DataFrame(rows)


refinement_depth_ablation=run_refinement_depth_ablation(model)
_assert(np.isfinite(refinement_depth_ablation[["fidelity","energy_mae","independent_residual","training_block_residual","seconds"]].to_numpy()).all())
refinement_depth_ablation.to_csv(OUT/"refinement_depth_ablation.csv",index=False)

physics_rows=training_history[training_history.stage.isin(("F_physics_ramp","G_joint_finetune"))]
physics_activated=bool(physics_rows.physics_authorized.any())
physics_status=(f"activated in {CFG.mode}; applied weights logged" if physics_activated else
                f"stage reached in {CFG.mode}; physics losses disabled by fidelity gate")
collapse_demonstrated=bool(ae_validation["effective_rank"]<2 or ae_validation["near_zero_variance_fraction"]>.5 or ae_validation["dominant_covariance_fraction"]>.95)
_assert(not CFG.use_sigreg or collapse_demonstrated,"SIGReg cannot be enabled without measured collapse")
ABLATION_STATUS=pd.DataFrame([
    ("Spectral sine-residual decoder","one-step optimizer smoke executed; parameter count reported; matched training pending"),
    ("Topology phase decoder",f"main {CFG.mode} plus one-step optimizer smoke; parameter count reported; matched multi-seed training pending"),
    ("Topology without node-phase loss","one-step optimizer smoke executed; matched training pending"),
    ("Topology without lobe-mass loss","one-step optimizer smoke executed; matched training pending"),
    ("Topology without phase-resolution regularization","one-step optimizer smoke executed; matched training pending"),
    ("Direct topology versus hybrid output","both one-step routes executed from identical split/backbone/seed; matched training pending"),
    ("Hybrid topology safeguard on/off","both one-step routes executed from identical split/backbone/seed; matched training pending"),
    ("Direct strong baseline","one-step optimizer smoke executed; matched training pending"),("GNN with MLP","one-step optimizer smoke executed; matched training pending"),
    ("GNN-KAN Chebyshev",f"main {CFG.mode} plus optimizer smoke; matched multi-seed training pending"),("GNN-KAN Legendre","optimizer smoke executed; matched training pending"),("GNN-KAN normalized Hermite","optimizer smoke executed; matched training pending"),
    ("Direct supervised without JEPA","optimizer smoke executed; matched training pending"),("JEPA without physics","optimizer smoke plus stage D/E trajectory; matched training pending"),("JEPA with physics",f"optimizer wiring smoke executed; main curriculum {physics_status}; matched training pending"),
    ("Local graph only","optimizer smoke executed; matched training pending"),("Multiscale graph","optimizer smoke executed without global context; matched training pending"),("Multiscale plus global context",f"optimizer smoke plus main {CFG.mode}; matched training pending"),
    ("Orthogonality loss only","optimizer smoke executed; matched training pending"),("Explicit orthonormalization","regularized Löwdin and Löwdin-plus-loss optimizer smokes executed; post-Gram result reported; matched training pending"),("Energy head only","optimizer smoke executed; matched training pending"),
    ("Rayleigh-coupled energy","optimizer smoke executed; matched training pending"),("With continuum threshold metadata","assisted optimizer smoke executed; matched training pending"),("Without continuum threshold metadata",f"operator-only optimizer smoke plus primary {CFG.mode}"),
    ("Recurrent eigenspace refinement","T=0/1/3/5 hybrid and T=3 classical optimizer smokes executed; matched DEVELOPMENT retraining comparison pending"),
    ("SIGReg off/on","SIGReg off main; on not executed because collapse was not demonstrated" if not collapse_demonstrated else "collapse demonstrated; controlled pooled-head comparison pending"),("KAN degrees 3/5/7","all recurrence/layer backward smokes executed; matched training pending"),("KAN ranks 2/4/8","all factorized-layer backward smokes executed; matched training pending"),
],columns=["ablation","status"]); ABLATION_STATUS.to_csv(OUT/"ablation_status.csv",index=False)
MATCHED_ABLATION_MATRIX={"split_manifest_hash":SPLIT_MANIFEST_HASH,"dataset_manifest_hash":DATASET_MANIFEST_HASH,"scheduled_seeds":list(CFG.preset.seeds),
    "early_stopping_rule":"identical validation score and checkpoint selection used by the main curriculum","budget_rule":"match trainable parameters as closely as architecture permits",
    "quick_smoke_results":"ablation_smoke_results.csv","full_accuracy_status":"PENDING_MATCHED_MULTI_SEED_TRAINING","experiments":ABLATION_STATUS.to_dict("records")}
(OUT/"matched_ablation_matrix.json").write_text(json.dumps(MATCHED_ABLATION_MATRIX,indent=2))
print(ablation_smokes[["smoke","inference_parameters","trainable_parameters","one_step_seconds","gradient_norm","initial_fidelity","post_step_fidelity","pre_step_raw_node_accuracy","post_step_raw_node_accuracy","pre_step_independent_residual","post_step_independent_residual","post_step_gram_max_error"]].round(5).to_string(index=False)); print(ABLATION_STATUS.to_string(index=False))

# %% [notebook cell 73]
inference_operator=operator_only_record(splits["test"][0])
forbidden_inference_keys={"family","parameters","psi","rho","energy","valid_state_mask","state_mask","node_count","bound_margin","solver_residual","target_latents",
                          "target_node_positions","target_node_mask","target_phase_cdf","target_phase_cdf_mask",
                          "target_log_amplitude","target_log_amplitude_mask"}
assert not forbidden_inference_keys&set(inference_operator)
inference_demo=target_free_predict(model,inference_operator)
assert np.array_equal(inference_demo["rho"],inference_demo["psi"]**2)
assert np.array_equal(inference_demo["rho_direct"],inference_demo["psi_direct"]**2)
assert np.array_equal(inference_demo["rho_hybrid"],inference_demo["psi_hybrid"]**2)
np.savez_compressed(OUT/"inference_only_demo.npz",**inference_demo,x=inference_operator["x"],V=inference_operator["V_raw"])
print({"inference_input_keys":sorted(inference_operator),"output_shapes":{k:v.shape for k,v in inference_demo.items()},"targets_present":False})
