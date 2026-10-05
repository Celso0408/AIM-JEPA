"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 55]
def to_device(batch:Mapping[str,Any])->dict[str,Any]:
    return {key:(value.to(DEVICE,non_blocking=True) if torch.is_tensor(value) else value) for key,value in batch.items()}


def cpu_snapshot(value:Any)->Any:
    """Recursively clone training state onto CPU without retaining CUDA storage."""
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {key:cpu_snapshot(item) for key,item in value.items()}
    if isinstance(value,list): return [cpu_snapshot(item) for item in value]
    if isinstance(value,tuple): return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def masked_state_mean(values:torch.Tensor,state_mask:torch.Tensor)->torch.Tensor:
    while state_mask.ndim<values.ndim: state_mask=state_mask.unsqueeze(-1)
    expanded=state_mask.expand_as(values).to(values.dtype); return (values*expanded).sum()/expanded.sum().clamp_min(1)


def global_sign_alignment_torch(pred:torch.Tensor,target:torch.Tensor,w:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    """Return the per-state sign gauge and sign-invariant fidelity to ``target``."""
    overlap=torch.sum(w[:,None]*pred*target,dim=-1)
    sign=torch.where(overlap.detach()>=0,torch.ones_like(overlap),-torch.ones_like(overlap))
    return sign,overlap.square().clamp(0,1)


def globally_align_torch(pred:torch.Tensor,target:torch.Tensor,w:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    sign,fidelity=global_sign_alignment_torch(pred,target,w)
    return pred*sign[...,None],fidelity


def derivative_values(y:torch.Tensor,x:torch.Tensor,node_mask:torch.Tensor|None=None)->torch.Tensor:
    dx=x[:,1:]-x[:,:-1]
    interval=torch.ones_like(dx,dtype=torch.bool) if node_mask is None else node_mask[:,1:]&node_mask[:,:-1]
    safe_dx=torch.where(interval,dx,torch.ones_like(dx)).clamp_min(1e-8)
    values=(y[...,1:]-y[...,:-1])/safe_dx[:,None]
    return torch.where(interval[:,None],values,torch.zeros_like(values))


def dimensionless_derivative_errors(pred:torch.Tensor,target:torch.Tensor,batch:Mapping[str,torch.Tensor])->tuple[torch.Tensor,torch.Tensor]:
    """Return dilation-invariant derivative error and target seminorm in t=(x-a)/L."""
    length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8)
    scale=torch.sqrt(length)[:,None,None]; pred_phi=pred*scale; target_phi=target*scale
    interval=batch["node_mask"][:,1:]&batch["node_mask"][:,:-1]
    dt=(batch["t"][:,1:]-batch["t"][:,:-1])*interval
    pred_derivative=derivative_values(pred_phi,batch["t"],batch["node_mask"])
    target_derivative=derivative_values(target_phi,batch["t"],batch["node_mask"])
    error=torch.sum(dt[:,None]*(pred_derivative-target_derivative).square(),dim=-1)
    reference=torch.sum(dt[:,None]*target_derivative.square(),dim=-1)
    return error,reference


def gram_loss(psi:torch.Tensor,w:torch.Tensor,state_mask:torch.Tensor)->torch.Tensor:
    gram=torch.einsum("bkn,bn,bjn->bkj",psi,w,psi); valid=state_mask[:,:,None]&state_mask[:,None,:]
    target=torch.eye(CFG.k_states,device=psi.device)[None].expand_as(gram); return ((gram-target).square()*valid).sum()/valid.sum().clamp_min(1)


def latent_diagnostics(z:torch.Tensor,token_mask:torch.Tensor|None=None)->dict[str,Any]:
    flat=z.detach().float().reshape(-1,z.shape[-1])
    if token_mask is not None: flat=flat[token_mask.reshape(-1)]
    if len(flat)<2: return {"latent_variance":0.0,"effective_rank":0.0,"condition_number":float("inf"),"active_condition_number":float("inf"),"active_covariance_rank":0,"dominant_covariance_fraction":1.0,"covariance_eigenvalues":[],"near_zero_variance_fraction":1.0}
    centered=flat-flat.mean(0); cov=centered.T@centered/max(len(flat)-1,1); eigen=torch.linalg.eigvalsh(cov).clamp_min(1e-12); probability=eigen/eigen.sum().clamp_min(1e-12)
    active=eigen[eigen>1e-6*eigen.max()]
    return {"latent_variance":float(flat.var(0,unbiased=False).mean()),"effective_rank":float(torch.exp(-(probability*torch.log(probability)).sum())),
            "condition_number":float(eigen.max()/eigen.min()),"active_condition_number":float(active.max()/active.min()),
            "dominant_covariance_fraction":float(eigen.max()/eigen.sum().clamp_min(1e-12)),"covariance_eigenvalues":eigen.cpu().numpy().tolist(),
            "active_covariance_rank":int(len(active)),"near_zero_variance_fraction":float((flat.std(0)<1e-3).float().mean())}


def kan_coefficient_penalty(module:nn.Module)->torch.Tensor:
    penalties=[]
    for child in module.modules():
        if isinstance(child,DenseBasisKANLinear): penalties.append(child.coefficients.square().mean())
        elif isinstance(child,FactorizedBasisKANLinear):
            dense=child.dense_coefficients(); degree=torch.arange(1,child.degree+1,device=dense.device,dtype=dense.dtype)
            penalties.append((dense.square()*degree.square()).mean())
    return torch.stack(penalties).mean() if penalties else next(module.parameters()).sum()*0


DEFAULT_LOSS_WEIGHTS={"jepa":0.2,"subspace_projector":0.50,"initial_subspace_projector":0.20,"wave":1.0,"infidelity":0.10,"density":0.15,
                      "direct_wave":1.0,"direct_infidelity":0.10,"direct_density":0.15,"direct_h1":0.12,"direct_subspace_projector":0.50,
                      "energy_ground_exact":CFG.energy_ground_loss_weight,
                      "energy_log_gap_exact":CFG.energy_log_gap_loss_weight,
                      "energy_spectrum_exact":CFG.energy_exact_loss_weight,
                      "rayleigh_energy_exact":CFG.rayleigh_exact_loss_weight,
                      "energy_rayleigh_consistency":CFG.energy_rayleigh_consistency_weight,"h1":0.12,
                      "reconstruction":0.03,"orthogonality":0.12,"gram_barrier":0.10,"residual":0.18,
                      "refinement_deep_residual":0.10,"refinement_contraction":0.08,
                      "latent_variance":0.05,"latent_covariance":0.01,"reflection_wave":0.10,"reflection_energy":0.05,
                      # Normalization and explicit boundaries are imposed structurally in
                      # the selected primary decoder. Tail mass remains a learned penalty.
                      "normalization":0.0,"boundary":0.01 if CFG.boundary_mode=="soft" else 0.0,"tail":0.01,"kan":1e-6,
                      "topology_node_phase":CFG.topology_node_loss_weight,
                      "topology_lobe_mass":CFG.topology_lobe_mass_weight,
                      "topology_phase_cdf_supervision":CFG.topology_phase_cdf_loss_weight,
                      "topology_log_amplitude_supervision":CFG.topology_log_amplitude_loss_weight,
                      "topology_phase_smoothness":CFG.topology_phase_smoothness_weight,
                      "topology_phase_resolution":CFG.topology_phase_resolution_weight,
                      "topology_amplitude_smoothness":CFG.topology_amplitude_smoothness_weight}


def latent_regularization(z:torch.Tensor,state_mask:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    flat=z[state_mask]
    if len(flat)<2:
        zero=z.sum()*0; return zero,zero
    centered=flat-flat.mean(0); variance=centered.square().mean(0); variance_loss=(torch.sqrt(variance+1e-4)-1).square().mean()
    standardized=centered/torch.sqrt(variance+1e-4); covariance=standardized.T@standardized/max(len(flat)-1,1)
    off_diagonal=covariance-torch.diag_embed(torch.diagonal(covariance)); covariance_loss=off_diagonal.square().sum()/max(CFG.latent_dim*(CFG.latent_dim-1),1)
    return variance_loss,covariance_loss


def curriculum_state_mask(state_mask:torch.Tensor,active_states:int|None)->torch.Tensor:
    count=state_mask.shape[1] if active_states is None else int(active_states)
    if not 1<=count<=state_mask.shape[1]: raise ValueError(f"active_states must lie in [1,{state_mask.shape[1]}]")
    return state_mask&(torch.arange(state_mask.shape[1],device=state_mask.device)[None]<count)


def ramped_state_mean(values:torch.Tensor,state_mask:torch.Tensor,state_weights:torch.Tensor)->torch.Tensor:
    """State mean whose denominator preserves the physical-ramp amplitude."""
    if values.ndim<2 or state_mask.ndim!=2 or values.shape[:2]!=state_mask.shape: raise ValueError("state reducer expects values shaped (batch,state,...)")
    state_weights=torch.as_tensor(state_weights,device=values.device,dtype=values.dtype)
    if state_weights.shape!=(values.shape[1],) or not bool(torch.isfinite(state_weights).all()) or bool(((state_weights<0)|(state_weights>1)).any()):
        raise ValueError("state_weights must be a finite vector in [0,1]")
    while state_mask.ndim<values.ndim: state_mask=state_mask.unsqueeze(-1)
    weights=state_weights[None]
    while weights.ndim<values.ndim: weights=weights.unsqueeze(-1)
    expanded_mask=state_mask.expand_as(values).to(values.dtype)
    return (values*expanded_mask*weights.expand_as(values).to(values.dtype)).sum()/expanded_mask.sum().clamp_min(1)


def prefix_projector_loss(pred:torch.Tensor,target:torch.Tensor,w:torch.Tensor,state_mask:torch.Tensor,
                          active_states:int|None,prefixes:Sequence[int]=(3,6,11))->tuple[torch.Tensor,dict[str,torch.Tensor]]:
    """Rotation/sign-invariant weighted projector loss for nested low-energy prefixes."""
    count=state_mask.shape[1] if active_states is None else int(active_states)
    if not 1<=count<=state_mask.shape[1]: raise ValueError(f"active_states must lie in [1,{state_mask.shape[1]}]")
    losses=[]; diagnostics={}
    for prefix in prefixes:
        if prefix>count or prefix>state_mask.shape[1]: continue
        valid=state_mask[:,:prefix].all(1); overlap=torch.einsum("bkn,bn,bjn->bkj",pred[:,:prefix],w,target[:,:prefix])
        coverage=overlap.square().sum((-2,-1))/prefix
        loss_per=F.relu(1-coverage)
        loss=(loss_per*valid.to(loss_per.dtype)).sum()/valid.sum().clamp_min(1)
        diagnostics[f"subspace_projector_p{prefix}"]=loss
        diagnostics[f"subspace_fidelity_p{prefix}"]=((coverage.clamp(0,1))*valid.to(coverage.dtype)).sum()/valid.sum().clamp_min(1)
        losses.append(loss)
    zero=pred.sum()*0
    if not losses: return zero,diagnostics
    # Preserve earlier low-energy prefixes while emphasizing the newly introduced
    # subspace: (1), (0.4,0.6), then (0.2,0.2,0.6).
    coefficients=torch.ones(len(losses),device=pred.device,dtype=pred.dtype) if len(losses)==1 else torch.tensor(
        ([.4,.6] if len(losses)==2 else [.2,.2,.6]),device=pred.device,dtype=pred.dtype)
    return torch.sum(torch.stack(losses)*coefficients),diagnostics


def reverse_valid_nodes(values:torch.Tensor,lengths:torch.Tensor)->torch.Tensor:
    result=torch.zeros_like(values)
    for index,n in enumerate(lengths.tolist()): result[index,...,:n]=values[index,...,:n].flip(-1)
    return result


def physics_tail_mask(batch:Mapping[str,torch.Tensor],fraction:float=.05)->torch.Tensor:
    domain=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8)
    left_distance=(batch["x"]-batch["endpoints"][:,0,None])/domain[:,None]
    right_distance=(batch["endpoints"][:,1,None]-batch["x"])/domain[:,None]
    truncated=batch["geometry"]==GEOMETRIES.index("truncated_line")
    one_sided=torch.isin(batch["geometry"],torch.tensor([GEOMETRIES.index("half_line"),GEOMETRIES.index("radial_reduced")],device=batch["x"].device))
    return (((one_sided[:,None]&(right_distance<=fraction))|(truncated[:,None]&((left_distance<=fraction)|(right_distance<=fraction))))&batch["node_mask"])


def refinement_contraction_statistics(proposal_trace:torch.Tensor,previous_trace:torch.Tensor,
                                      state_mask:torch.Tensor,attempted_trace:torch.Tensor,
                                      acceptance_trace:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    """Attempt-masked contraction loss and accepted/attempted block fraction."""
    if proposal_trace.shape!=previous_trace.shape: raise ValueError("proposal and previous residual traces must have identical shapes")
    if attempted_trace.shape!=proposal_trace.shape[:2] or acceptance_trace.shape!=attempted_trace.shape:
        raise ValueError("refinement attempt/acceptance traces have incompatible shapes")
    proposal_mask=state_mask[:,None].expand_as(proposal_trace)&attempted_trace[:,:,None]
    finite=torch.isfinite(proposal_trace)
    sanitized=torch.nan_to_num(proposal_trace,nan=0.0,posinf=0.0,neginf=0.0)
    finite_fallback=previous_trace.detach()+1.0
    safe_proposal=torch.where(finite,sanitized,finite_fallback)
    contraction_error=F.relu(safe_proposal-CFG.refinement_contraction_target*previous_trace.detach()).square()
    contraction=torch.where(proposal_mask,contraction_error,torch.zeros_like(contraction_error)).sum()/proposal_mask.sum().clamp_min(1)
    attempts=attempted_trace.sum(); accepted=(acceptance_trace&attempted_trace).sum()
    acceptance=accepted.to(proposal_trace.dtype)/attempts.clamp_min(1).to(proposal_trace.dtype)
    return contraction,acceptance


def interpolate_field_at_target_nodes(field:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    """Interpolate a nodewise field at detached target roots; field gradients remain live."""
    B,K,N=field.shape; positions=batch["target_node_positions"].detach()
    length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8)
    target_t=((positions-batch["endpoints"][:,0,None,None])/length[:,None,None]).clamp(0,1)
    search=torch.where(batch["node_mask"],batch["t"],torch.full_like(batch["t"],2.0))[:,None].expand(B,K,N).contiguous()
    right=torch.searchsorted(search.detach(),target_t.contiguous(),right=False).clamp(1,N-1); left=right-1
    t_grid=batch["t"][:,None].expand(B,K,N); t0=torch.gather(t_grid,-1,left); t1=torch.gather(t_grid,-1,right)
    f0=torch.gather(field,-1,left); f1=torch.gather(field,-1,right)
    alpha=((target_t-t0)/(t1-t0).clamp_min(1e-8)).clamp(0,1)
    return f0+alpha*(f1-f0)


def normalized_density_cdf(rho:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    device_type=rho.device.type if rho.device.type in ("cpu","cuda") else "cpu"
    with torch.autocast(device_type=device_type,enabled=False):
        rho=rho.float(); interval=batch["node_mask"][:,1:]&batch["node_mask"][:,:-1]
        dx=torch.where(interval,batch["x"][:,1:].float()-batch["x"][:,:-1].float(),torch.zeros_like(batch["x"][:,:-1].float()))
        increment=0.5*(rho[...,1:]+rho[...,:-1])*dx[:,None]*interval[:,None]
        cumulative=torch.cat([torch.zeros_like(rho[...,:1]),deterministic_prefix_sum(increment,-1)],-1)
        total=increment.sum(-1,keepdim=True).clamp_min(1e-12)
        return (cumulative/total)*batch["node_mask"][:,None]


def dimensionless_interval_h1(field:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    interval=batch["node_mask"][:,1:]&batch["node_mask"][:,:-1]
    dt=torch.where(interval,batch["t"][:,1:].float()-batch["t"][:,:-1].float(),torch.ones_like(batch["t"][:,:-1].float())).clamp_min(1e-8)
    difference=torch.where(interval[:,None],field.float()[...,1:]-field.float()[...,:-1],torch.zeros_like(field.float()[...,:-1]))
    return (difference.square()/dt[:,None]).sum(-1)


def topology_supervision_terms(out:Mapping[str,torch.Tensor],batch:Mapping[str,torch.Tensor],mask:torch.Tensor)->dict[str,torch.Tensor]:
    zero=out["psi"].sum()*0
    if "phase_cdf" not in out:
        return {name:zero for name in ("topology_node_phase","topology_lobe_mass",
                                       "topology_phase_cdf_supervision","topology_log_amplitude_supervision",
                                       "topology_phase_smoothness","topology_phase_resolution",
                                       "topology_amplitude_smoothness")}
    target_node_mask=batch["target_node_mask"]&mask[:,:,None]
    cdf_at_nodes=interpolate_field_at_target_nodes(out["phase_cdf"],batch)
    states=torch.arange(CFG.k_states,device=out["psi"].device,dtype=out["psi"].dtype)[None,:,None]
    j=torch.arange(1,CFG.k_states,device=out["psi"].device,dtype=out["psi"].dtype)[None,None,:]
    quantiles=j/(states+1)
    node_error=torch.where(target_node_mask,(cdf_at_nodes-quantiles).square(),torch.zeros_like(cdf_at_nodes))
    node_phase=node_error.sum()/target_node_mask.sum().clamp_min(1)

    pred_cdf=normalized_density_cdf(out["rho_direct"],batch)
    true_cdf=normalized_density_cdf(batch["psi"].square(),batch).detach()
    pred_nodes=interpolate_field_at_target_nodes(pred_cdf,batch); true_nodes=interpolate_field_at_target_nodes(true_cdf,batch)
    pred_nodes=torch.where(batch["target_node_mask"],pred_nodes,torch.ones_like(pred_nodes))
    true_nodes=torch.where(batch["target_node_mask"],true_nodes,torch.ones_like(true_nodes))
    zeros=torch.zeros_like(pred_nodes[...,:1]); ones=torch.ones_like(pred_nodes[...,:1])
    pred_lobes=torch.diff(torch.cat([zeros,pred_nodes,ones],-1),dim=-1)
    true_lobes=torch.diff(torch.cat([zeros,true_nodes,ones],-1),dim=-1)
    node_counts=batch["target_node_mask"].sum(-1); lobe_index=torch.arange(CFG.k_states,device=pred_lobes.device)
    lobe_mask=mask[:,:,None]&(lobe_index[None,None]<=node_counts[:,:,None])
    lobe_error=torch.where(lobe_mask,(pred_lobes-true_lobes).square(),torch.zeros_like(pred_lobes))
    lobe_mass=lobe_error.sum()/lobe_mask.sum().clamp_min(1)

    # Full-field targets are solution-branch labels only.  Float32 quadrature and
    # detached labels keep the numerical path stable while gradients remain live
    # through the predicted phase CDF and log-amplitude fields.
    device_type=out["psi"].device.type if out["psi"].device.type in ("cpu","cuda") else "cpu"
    with torch.autocast(device_type=device_type,enabled=False):
        w_t=batch["w_dimensionless"].float()[:,None]
        node_valid=batch["node_mask"][:,None]&mask[:,:,None]
        phase_valid=batch["target_phase_cdf_mask"]&node_valid
        phase_weight=w_t*phase_valid.to(w_t.dtype)
        phase_error=(out["phase_cdf"].float()-batch["target_phase_cdf"].detach().float()).square()
        phase_per=(phase_weight*phase_error).sum(-1)/phase_weight.sum(-1).clamp_min(1e-12)
        phase_profile=masked_state_mean(phase_per,mask&phase_valid.any(-1))
        amplitude_valid=batch["target_log_amplitude_mask"]&node_valid
        amplitude_weight=w_t*amplitude_valid.to(w_t.dtype)
        amplitude_error=(out["log_amplitude"].float()-batch["target_log_amplitude"].detach().float()).square()
        amplitude_per=(amplitude_weight*amplitude_error).sum(-1)/amplitude_weight.sum(-1).clamp_min(1e-12)
        amplitude_profile=masked_state_mean(amplitude_per,mask&amplitude_valid.any(-1))

    phase_smooth=masked_state_mean(dimensionless_interval_h1(torch.log(out["phase_density"].clamp_min(1e-30)),batch),mask)
    amplitude_smooth=masked_state_mean(dimensionless_interval_h1(out["log_amplitude"],batch),mask)
    interval=batch["node_mask"][:,1:]&batch["node_mask"][:,:-1]
    limit=math.pi/CFG.minimum_intervals_per_lobe
    excess=F.relu(out["phase_increment"]/limit-1).square()*interval[:,None]
    resolution_per=excess.sum(-1)/torch.arange(1,CFG.k_states+1,device=excess.device,dtype=excess.dtype)[None]
    resolution=masked_state_mean(resolution_per,mask)
    return {"topology_node_phase":node_phase,"topology_lobe_mass":lobe_mass,
            "topology_phase_cdf_supervision":phase_profile,
            "topology_log_amplitude_supervision":amplitude_profile,
            "topology_phase_smoothness":phase_smooth,"topology_phase_resolution":resolution,
            "topology_amplitude_smoothness":amplitude_smooth}


def compute_losses(out:Mapping[str,torch.Tensor],batch:Mapping[str,torch.Tensor],weights:Mapping[str,float],loss_model:nn.Module|None=None,
                   active_states:int|None=None,physics_state_weights:torch.Tensor|Sequence[float]|None=None)->tuple[torch.Tensor,dict[str,torch.Tensor]]:
    pred,target,w,full_mask=out["psi"],batch["psi"],batch["w"],batch["state_mask"]
    mask=curriculum_state_mask(full_mask,active_states); active_count=int(mask.shape[1] if active_states is None else active_states)
    if physics_state_weights is None: physics_weights=torch.ones(mask.shape[1],device=pred.device,dtype=pred.dtype)
    else: physics_weights=torch.as_tensor(physics_state_weights,device=pred.device,dtype=pred.dtype)
    if physics_weights.shape!=(mask.shape[1],): raise ValueError(f"physics_state_weights must have shape {(mask.shape[1],)}")
    physics_weights=physics_weights*(torch.arange(mask.shape[1],device=pred.device)<active_count).to(pred.dtype)
    aligned,fidelity=globally_align_torch(pred,target,w); aligned_pre,fidelity_pre=globally_align_torch(out["psi_pre_orth"],target,w)
    aligned_direct,direct_fidelity=globally_align_torch(out["psi_direct"],target,w)
    wave_per=torch.sum(w[:,None]*(aligned-target).square(),dim=-1)
    density_per=torch.sum(w[:,None]*(out["rho"]-target.square()).abs(),dim=-1)
    derivative_error,derivative_target=dimensionless_derivative_errors(aligned,target,batch)
    h1_per=(wave_per+derivative_error)/(1+derivative_target).clamp_min(1e-8)
    direct_wave_per=torch.sum(w[:,None]*(aligned_direct-target).square(),dim=-1)
    direct_density_per=torch.sum(w[:,None]*(out["rho_direct"]-target.square()).abs(),dim=-1)
    direct_derivative_error,_=dimensionless_derivative_errors(aligned_direct,target,batch)
    direct_h1_per=(direct_wave_per+direct_derivative_error)/(1+derivative_target).clamp_min(1e-8)
    # Exact-energy supervision is expressed in operator-only coordinates.  The direct
    # learned head is supervised independently of the selected direct/hybrid reporting
    # route, so a Ritz branch can never hide a poorly calibrated learned spectrum.
    target_dimensionless=batch["energy_dimensionless"].float()
    head_dimensionless=out["head_energy_dimensionless"].float()
    operator_scale=out["energy_operator_scale_dimensionless"].detach().float().clamp_min(CFG.energy_scale_floor)
    ground_scale=out["energy_ground_scale_dimensionless"].detach().float().clamp_min(CFG.energy_scale_floor)
    state_order=torch.arange(1,mask.shape[1]+1,device=pred.device,dtype=torch.float32)
    state_energy_scale=ground_scale[:,None]+operator_scale[:,None]*state_order[None].square()
    selected_energy_per=F.smooth_l1_loss(out["energy_dimensionless"].float()/state_energy_scale,
                                         target_dimensionless/state_energy_scale,reduction="none")
    energy_exact_per=F.smooth_l1_loss(head_dimensionless/state_energy_scale,
                                      target_dimensionless/state_energy_scale,reduction="none")
    target_ground_coordinate=((target_dimensionless[:,0]-out["energy_reference_dimensionless"][:,0].detach().float())/ground_scale).clamp(
        -CFG.energy_ground_residual_clip,CFG.energy_ground_residual_clip)
    ground_exact_per=F.smooth_l1_loss(out["head_ground_residual_scaled"].float(),target_ground_coordinate,reduction="none")
    gap_mask=mask[:,1:]&mask[:,:-1]
    target_gaps=torch.diff(target_dimensionless,dim=1)
    reference_gaps=out["energy_reference_gap_dimensionless"].detach().float().clamp_min(
        CFG.energy_reference_gap_floor*operator_scale[:,None])
    target_log_gap_ratio=torch.log(target_gaps.clamp_min(CFG.energy_reference_gap_floor*operator_scale[:,None])/reference_gaps).clamp(
        -CFG.energy_log_gap_clip,CFG.energy_log_gap_clip)
    log_gap_exact_per=F.smooth_l1_loss(out["head_log_gap_ratio"].float(),target_log_gap_ratio,reduction="none")
    latent_overlap=torch.sum(out["z_pred"]*out["z_target"].detach(),dim=-1); latent_sign=torch.where(latent_overlap.detach()>=0,1.0,-1.0)
    aligned_z_pred=out["z_pred"]*latent_sign[...,None]
    jepa_per=F.smooth_l1_loss(F.layer_norm(aligned_z_pred,aligned_z_pred.shape[-1:]),F.layer_norm(out["z_target"].detach(),out["z_target"].shape[-1:]),reduction="none")
    recon_aligned,_=globally_align_torch(out["reconstruction"],target,w); reconstruction_per=torch.sum(w[:,None]*(recon_aligned-target).square(),dim=-1)
    subspace_projector,subspace_diagnostics=prefix_projector_loss(pred,target,w,full_mask,active_count)
    direct_subspace_projector,_=prefix_projector_loss(out["psi_direct"],target,w,full_mask,active_count)
    initial_subspace_projector,_=prefix_projector_loss(out["psi_initial"],target,w,full_mask,active_count)
    with torch.autocast(device_type=pred.device.type,enabled=False):
        pf=pred.float(); Hpsi,interior=training_hamiltonian_action(pf,batch["x"].float(),batch["V"].float(),batch["kappa"].float(),batch["node_mask"]); core=pf[:,:,1:-1]; wi=w[:,1:-1].float()*interior
        residual=Hpsi-out["energy"].float()[:,:,None]*core
        residual_per=torch.sum(wi[:,None]*residual.square(),dim=-1)/(torch.sum(wi[:,None]*Hpsi.square(),dim=-1)+out["energy"].float().square()+1e-8)
        # The weak-Galerkin quotient is the differentiable training quantity.  It is
        # tied both to the exact energy and to the direct learned head; the independent
        # strong/local-polynomial quotient remains an audit and is not optimized here.
        weak_rayleigh_direct=out["weak_rayleigh_energy_direct_dimensionless"].float()
        rayleigh_exact_per=F.smooth_l1_loss(weak_rayleigh_direct/state_energy_scale,
                                            target_dimensionless/state_energy_scale,reduction="none")
        rayleigh_consistency_per=F.smooth_l1_loss(weak_rayleigh_direct/state_energy_scale,
                                                  head_dimensionless/state_energy_scale,reduction="none")
    gram_pre=torch.einsum("bkn,bn,bjn->bkj",out["psi_pre_orth"],w,out["psi_pre_orth"]); gram_post=torch.einsum("bkn,bn,bjn->bkj",pred,w,pred); identity=torch.eye(CFG.k_states,device=pred.device)[None]
    valid_pair=mask[:,:,None]&mask[:,None,:]; orth=((gram_pre-identity).square()*valid_pair).sum()/valid_pair.sum().clamp_min(1); gram_post_error=((gram_post-identity).square()*valid_pair).sum()/valid_pair.sum().clamp_min(1)
    gram_min=torch.linalg.eigvalsh(gram_pre.float()).amin(-1); gram_barrier=F.relu(.20-gram_min).square().mean()
    normalization_per=torch.abs(torch.sum(w[:,None]*pred.square(),dim=-1)-1)
    boundary_per=torch.sum(pred.square()*batch["boundary_mask"][:,None],dim=-1)/batch["boundary_mask"].sum(-1)[:,None].clamp_min(1)
    tail_shell=physics_tail_mask(batch); tail_per=torch.sum(w[:,None]*pred.square()*tail_shell[:,None],dim=-1)
    pred_variance,pred_covariance=latent_regularization(out["z_pred"],mask); online_variance,online_covariance=latent_regularization(out["z_online"],mask)
    latent_variance=.5*(pred_variance+online_variance); latent_covariance=.5*(pred_covariance+online_covariance)
    reflection_wave=pred.sum()*0; reflection_energy=pred.sum()*0; pairs=batch.get("reflection_pairs")
    if pairs is not None and len(pairs)>0:
        original=pairs[:,0]; reflected=pairs[:,1]; original_reversed=reverse_valid_nodes(pred[original],batch["lengths"][original])
        pair_weight=batch["w"][reflected][:,None]; pair_overlap=torch.sum(pair_weight*pred[reflected]*original_reversed,dim=-1); pair_sign=torch.where(pair_overlap.detach()>=0,1.0,-1.0)
        reflection_wave_per=torch.sum(pair_weight*(pred[reflected]-pair_sign[...,None]*original_reversed).square(),dim=-1)
        reflection_energy_per=F.smooth_l1_loss(out["energy_dimensionless"][reflected],out["energy_dimensionless"][original],reduction="none")
        pair_mask=mask[original]&mask[reflected]
        reflection_wave=masked_state_mean(reflection_wave_per,pair_mask); reflection_energy=masked_state_mean(reflection_energy_per,pair_mask)
    active_gap_values=torch.where(gap_mask,torch.diff(out["energy_dimensionless"],dim=1),torch.full_like(target_gaps,float("inf")))
    ritz_min_gap=active_gap_values.amin(); ritz_min_gap=torch.where(torch.isfinite(ritz_min_gap),ritz_min_gap,ritz_min_gap.new_zeros(()))
    refinement_trace=out["refinement_residual_trace"]; proposal_trace=out["refinement_proposal_residual_trace"]
    trace_mask=mask[:,None].expand(-1,refinement_trace.shape[1],-1)
    refinement_deep_residual=((refinement_trace[:,1:].square()*trace_mask[:,1:]).sum()/trace_mask[:,1:].sum().clamp_min(1)
                              if refinement_trace.shape[1]>1 else pred.sum()*0)
    if proposal_trace.shape[1]>0:
        refinement_contraction,refinement_acceptance=refinement_contraction_statistics(
            proposal_trace,refinement_trace[:,:-1],mask,out["refinement_attempted_trace"],out["refinement_acceptance_trace"])
        accepted_steps=out["refinement_step_size_trace"][out["refinement_acceptance_trace"]]
        refinement_step_size=accepted_steps.mean() if accepted_steps.numel()>0 else pred.sum()*0
    else:
        refinement_contraction=pred.sum()*0; refinement_acceptance=pred.sum()*0; refinement_step_size=pred.sum()*0
    initial_block=torch.sqrt((refinement_trace[:,0].square()*mask).sum()/mask.sum().clamp_min(1))
    final_block=torch.sqrt((refinement_trace[:,-1].square()*mask).sum()/mask.sum().clamp_min(1))
    terms={"jepa":masked_state_mean(jepa_per.mean(-1),mask),"subspace_projector":subspace_projector,
           "initial_subspace_projector":initial_subspace_projector,
           "wave":masked_state_mean(wave_per,mask),"infidelity":masked_state_mean(1-fidelity,mask),
           "density":masked_state_mean(density_per,mask),
           "direct_wave":masked_state_mean(direct_wave_per,mask),
           "direct_infidelity":masked_state_mean(1-direct_fidelity,mask),
           "direct_density":masked_state_mean(direct_density_per,mask),
           "direct_h1":masked_state_mean(direct_h1_per,mask),
           "direct_subspace_projector":direct_subspace_projector,
           "energy_ground_exact":masked_state_mean(ground_exact_per[:,None],mask[:,:1]),
           "energy_log_gap_exact":masked_state_mean(log_gap_exact_per,gap_mask),
           "energy_spectrum_exact":masked_state_mean(energy_exact_per,mask),
           "rayleigh_energy_exact":ramped_state_mean(rayleigh_exact_per,mask,physics_weights),
           "energy_rayleigh_consistency":ramped_state_mean(rayleigh_consistency_per,mask,physics_weights),
           # Backward-compatible aliases used by existing dashboards and ablations.
           "energy":masked_state_mean(selected_energy_per,mask),
           "energy_head":masked_state_mean(energy_exact_per,mask),
           "gap":masked_state_mean(log_gap_exact_per,gap_mask),"h1":masked_state_mean(h1_per,mask),
           "reconstruction":masked_state_mean(reconstruction_per,mask),"orthogonality":orth,"gram_barrier":gram_barrier,
           "residual":ramped_state_mean(residual_per,mask,physics_weights),
           "rayleigh":ramped_state_mean(rayleigh_consistency_per,mask,physics_weights),
           "refinement_deep_residual":refinement_deep_residual,"refinement_contraction":refinement_contraction,
           "refinement_acceptance_fraction":refinement_acceptance,"refinement_mean_step_size":refinement_step_size,
           "refinement_initial_residual":initial_block,"refinement_final_residual":final_block,
           "refinement_relative_residual_reduction":(initial_block-final_block)/initial_block.clamp_min(1e-8),
           "refinement_iterations_used":out["refinement_iterations_used"].float().mean(),
           "refinement_converged_fraction":out["refinement_converged"].float().mean(),
           "refinement_fallback_fraction":out["refinement_fallback_recommended"].float().mean(),
           "hybrid_topology_valid_fraction":out["hybrid_topology_valid"].float().mean(),
           "direct_topology_valid_fraction":out["direct_topology_valid"].float().mean(),
           "physics_selected_topology_valid_fraction":out["physics_selected_topology_valid"].float().mean(),
           "hybrid_topology_fallback_fraction":out["hybrid_fallback_recommended"].float().mean(),
           "refinement_topology_rejection_fraction":(out["refinement_topology_rejection_trace"].float().mean()
                                                       if out["refinement_topology_rejection_trace"].numel() else pred.sum()*0),
           "latent_variance":latent_variance,"latent_covariance":latent_covariance,
           "reflection_wave":reflection_wave,"reflection_energy":reflection_energy,
           "normalization":ramped_state_mean(normalization_per,mask,physics_weights),"boundary":ramped_state_mean(boundary_per,mask,physics_weights),
           "tail":ramped_state_mean(tail_per,mask,physics_weights),"kan":kan_coefficient_penalty(model if loss_model is None else loss_model),
           "fidelity":masked_state_mean(fidelity,mask),"fidelity_all":masked_state_mean(fidelity,full_mask),
           "pre_orth_fidelity":masked_state_mean(fidelity_pre,mask),"pre_orth_gram_mse":orth,"post_orth_gram_mse":gram_post_error,
           "ritz_min_gap":ritz_min_gap,"ritz_relative_min_gap":out["ritz_relative_min_gap"].amin(),
           "ritz_gradient_safe_fraction":out["ritz_gradient_safe"].float().mean(),"active_state_count":pred.new_tensor(float(active_count)),
           **subspace_diagnostics}
    terms.update(topology_supervision_terms(out,batch,mask))
    for state in range(mask.shape[1]):
        state_valid=full_mask[:,state]
        terms[f"fidelity_s{state}"]=(fidelity[:,state]*state_valid.to(fidelity.dtype)).sum()/state_valid.sum().clamp_min(1)
    total=sum(float(weights.get(name,0))*terms[name] for name in weights if name in terms)
    return total,terms


def progressive_supervised_epochs(total:int)->tuple[int,int,int]:
    """Allocate approximately 25/33/42 percent to 3/6/11-state phases."""
    if total<3: raise ValueError("progressive supervised curriculum requires at least three epochs")
    low=max(1,round(.25*total)); middle=max(1,round(total/3)); full=total-low-middle
    if full<1: middle=max(1,middle-(1-full)); full=total-low-middle
    return low,middle,full


def build_curriculum_stage_definitions()->list[dict[str,Any]]:
    """Return the auditable state curriculum used by production training.

    Sturm state labels make individual states identifiable from the first active-state
    stage, so waveform and topology-field objectives start with states 0--2 instead of
    waiting for the 11-state stage.  Strong target-field weights are annealed as more
    states enter; the rotation-invariant projector remains an auxiliary objective.
    """
    low_epochs,middle_epochs,full_epochs=progressive_supervised_epochs(CFG.preset.supervised_epochs)
    statewise={"wave":1.0,"infidelity":1.0,"density":.15,"h1":.12,
               "energy_ground_exact":CFG.energy_ground_loss_weight,
               "energy_log_gap_exact":CFG.energy_log_gap_loss_weight,
               "energy_spectrum_exact":CFG.energy_exact_loss_weight,
               "topology_node_phase":CFG.topology_node_loss_weight,
               "topology_lobe_mass":CFG.topology_lobe_mass_weight}
    subspace_low={"jepa":.20,"subspace_projector":1.0,"initial_subspace_projector":.35,
                  "orthogonality":.03,"gram_barrier":.06,
                  "latent_variance":.05,"latent_covariance":.01,"residual":.05,
                  "rayleigh_energy_exact":.04,"energy_rayleigh_consistency":.04,
                  "refinement_deep_residual":.06,"refinement_contraction":.04,"tail":.005,"kan":1e-6,
                  **statewise,
                  "topology_phase_cdf_supervision":4.0*CFG.topology_phase_cdf_loss_weight,
                  "topology_log_amplitude_supervision":2.0*CFG.topology_log_amplitude_loss_weight,
                  "topology_phase_smoothness":CFG.topology_phase_smoothness_weight,
                  "topology_phase_resolution":CFG.topology_phase_resolution_weight,
                  "topology_amplitude_smoothness":CFG.topology_amplitude_smoothness_weight}
    subspace_middle={**subspace_low,"energy_ground_exact":.18,"energy_log_gap_exact":.18,
                     "energy_spectrum_exact":.18,"orthogonality":.05,"gram_barrier":.08,
                     "reflection_energy":.03,"residual":.07,"rayleigh_energy_exact":.05,
                     "energy_rayleigh_consistency":.05,
                     "topology_phase_cdf_supervision":2.0*CFG.topology_phase_cdf_loss_weight,
                     "topology_log_amplitude_supervision":1.5*CFG.topology_log_amplitude_loss_weight}
    return [
        # The JEPA-only stage does not move an otherwise unanchored decoder toward an
        # arbitrary smooth/uniform field. Decoder supervision starts at E_subspace_3.
        {"name":"D_jepa_latent","epochs":CFG.preset.jepa_epochs,
         "weights":{"jepa":1.0,"latent_variance":.05,"latent_covariance":.01,"kan":1e-6},
         "lr":CFG.learning_rate,"active_states":CFG.k_states,"allow_ritz_gradient":False},
        {"name":"E_subspace_3","epochs":low_epochs,"weights":subspace_low,
         "lr":CFG.learning_rate,"active_states":3,"allow_ritz_gradient":False},
        {"name":"E_subspace_6","epochs":middle_epochs,"weights":subspace_middle,
         "lr":CFG.learning_rate,"active_states":6,"allow_ritz_gradient":False},
        {"name":"E_supervised_11","epochs":full_epochs,
         "weights":{**DEFAULT_LOSS_WEIGHTS,"subspace_projector":.75,"initial_subspace_projector":.30},
         "lr":CFG.learning_rate,"active_states":CFG.k_states,"allow_ritz_gradient":False},
        {"name":"F_physics_ramp","epochs":CFG.preset.physics_epochs,
         "weights":{**DEFAULT_LOSS_WEIGHTS,"subspace_projector":.60,"initial_subspace_projector":.45},
         "lr":CFG.physics_learning_rate,"active_states":CFG.k_states,"allow_ritz_gradient":False},
        {"name":"G_joint_finetune","epochs":CFG.preset.finetune_epochs,
         "weights":{**DEFAULT_LOSS_WEIGHTS,"subspace_projector":.50,"initial_subspace_projector":.60},
         "lr":CFG.finetune_learning_rate,"active_states":CFG.k_states,"allow_ritz_gradient":True},
    ]


CURRICULUM_MODULE_LR_MULTIPLIERS={
    "potential_context":1.0,"predictor":1.0,
    "wavefunction_decoder":CFG.decoder_learning_rate_multiplier,
    "energy_head":1.0,"residual_correction":.50,
    "solution_online_encoder":.05,"reconstruction_decoder":.05,
}


def initial_physics_curriculum()->dict[str,list[Any]]:
    return {"enabled":[False]*CFG.k_states,"streak":[0]*CFG.k_states,"age":[0]*CFG.k_states}


def physics_state_ramps(state:Mapping[str,Sequence[Any]],active_states:int)->torch.Tensor:
    values=[(min(float(state["age"][index])/max(CFG.physics_ramp_epochs,1),1.0) if index<active_states and bool(state["enabled"][index]) else 0.0) for index in range(CFG.k_states)]
    return torch.tensor(values,dtype=torch.float32)


def update_physics_curriculum(state:Mapping[str,Sequence[Any]],validation:Mapping[str,float],active_states:int)->dict[str,list[Any]]:
    updated={key:list(value) for key,value in state.items()}
    for index in range(CFG.k_states):
        if index>=active_states:
            updated["enabled"][index]=False; updated["streak"][index]=0; updated["age"][index]=0; continue
        fidelity=float(validation.get(f"fidelity_s{index}",0.0))
        if bool(updated["enabled"][index]):
            if fidelity<CFG.physics_stop_fidelity:
                updated["enabled"][index]=False; updated["streak"][index]=0; updated["age"][index]=0
            else: updated["age"][index]=int(updated["age"][index])+1
        else:
            updated["streak"][index]=int(updated["streak"][index])+1 if fidelity>=CFG.physics_start_fidelity else 0
            if int(updated["streak"][index])>=CFG.physics_authorization_patience:
                updated["enabled"][index]=True; updated["age"][index]=1
    return updated


def ritz_gradient_validation_ready(validation:Mapping[str,float])->bool:
    state_fidelities=[float(validation.get(f"fidelity_s{index}",0.0)) for index in range(CFG.k_states)]
    return (min(state_fidelities)>=CFG.ritz_gradient_start_fidelity and
            float(validation.get("subspace_projector_p11",float("inf")))<=CFG.ritz_gradient_start_projector_loss and
            float(validation.get("ritz_gradient_safe_fraction",0.0))>=CFG.ritz_gradient_min_safe_fraction)


def curriculum_module_parameter_groups(m:TrueSchrodingerJEPA)->dict[str,list[nn.Parameter]]:
    modules={"potential_context":m.potential_context_encoder,"predictor":m.predictor,"wavefunction_decoder":m.wavefunction_decoder,
             "energy_head":m.energy_head,"residual_correction":m.residual_correction,
             "solution_online_encoder":m.solution_online_encoder,"reconstruction_decoder":m.solution_reconstruction_decoder}
    groups={name:parameters for name,module in modules.items() if (parameters:=[p for p in module.parameters() if p.requires_grad])}
    identifiers=[id(p) for params in groups.values() for p in params]
    expected=[p for name,p in m.named_parameters() if p.requires_grad and not name.startswith("solution_target_encoder.")]
    if len(identifiers)!=len(set(identifiers)) or set(identifiers)!={id(p) for p in expected}:
        raise AssertionError("module gradient groups must be disjoint and cover every non-target trainable parameter")
    return groups


def clip_module_gradients(groups:Mapping[str,Sequence[nn.Parameter]],clip_rms:float=CFG.module_gradient_clip_rms,
                          hard_norm:float=CFG.module_gradient_clip_hard_norm)->dict[str,Any]:
    """Clip independent module norms once per complete accumulation window."""
    metrics={}; pre_squared=0.0; post_squared=0.0; clipped_events=0; active_events=0
    for name,parameters in groups.items():
        params=list(parameters); parameter_count=sum(p.numel() for p in params); cap=min(float(hard_norm),float(clip_rms)*math.sqrt(max(parameter_count,1)))
        active=any(p.grad is not None for p in params)
        if active:
            pre=float(nn.utils.clip_grad_norm_(params,cap,error_if_nonfinite=True,foreach=False)); scale=min(1.0,cap/(pre+1e-6)); post=pre*scale; clipped=scale<1.0
            active_events+=1; clipped_events+=int(clipped); pre_squared+=pre*pre; post_squared+=post*post
        else: pre=post=0.0; scale=1.0; clipped=False
        metrics[name]={"pre":pre,"post":post,"cap":cap,"scale":scale,"parameter_count":parameter_count,"active":active,"clipped":clipped}
    metrics.update({"gradient_norm":math.sqrt(pre_squared),"gradient_norm_post_clip":math.sqrt(post_squared),
                    "clipped_module_events":clipped_events,"active_module_events":active_events,"any_clipped":clipped_events>0})
    return metrics

# %% [notebook cell 57]
TEST_RESULTS=[]
def run_test(name:str,fn:Callable[[],None])->None:
    started=time.perf_counter()
    try:
        fn(); TEST_RESULTS.append({"index":len(TEST_RESULTS)+1,"test":name,"status":"PASS","seconds":time.perf_counter()-started})
    except Exception as exc:
        TEST_RESULTS.append({"index":len(TEST_RESULTS)+1,"test":name,"status":"FAIL","seconds":time.perf_counter()-started,"detail":repr(exc)})
        pd.DataFrame(TEST_RESULTS).to_csv(OUT/"test_results.csv",index=False); raise


def _assert(condition:bool,message:str="assertion failed")->None:
    if not bool(condition): raise AssertionError(message)


def _assert_tensor_close(name:str,actual:torch.Tensor,expected:torch.Tensor,*,atol:float,rtol:float)->None:
    """Named finite-tensor comparison with actionable numerical diagnostics."""
    if actual.shape!=expected.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: non-finite values")
    if torch.allclose(actual,expected,atol=atol,rtol=rtol): return
    delta=(actual-expected).abs(); scale=torch.maximum(actual.abs(),expected.abs()).clamp_min(atol)
    relative=delta/scale; allowed=atol+rtol*expected.abs()
    raise AssertionError(f"{name}: max_abs={float(delta.max()):.3e}, max_rel={float(relative.max()):.3e}, "
                         f"outside_tolerance={int((delta>allowed).sum())}/{delta.numel()}, atol={atol:.1e}, rtol={rtol:.1e}")


def test_hierarchical_u_normalization():
    family_mass=1/len(TRAIN_FAMILIES); stats={row["family"]:row for row in NORMALIZATION_AUDIT["family_statistics"]}
    _assert(set(stats)==set(TRAIN_FAMILIES) and abs(NORMALIZATION_AUDIT["total_assigned_mass"]-1)<1e-12)
    _assert(max(abs(row["assigned_mass"]-family_mass) for row in stats.values())<1e-12,"family masses are unequal")
    for family,row in stats.items():
        expected=family_mass/row["hamiltonians"]
        masses=[mass for group,mass in NORMALIZATION_AUDIT["hamiltonian_assigned_masses"].items() if any(r["group_id"]==group and r["family"]==family for r in splits["train"])]
        _assert(masses and max(abs(mass-expected) for mass in masses)<1e-12,f"Hamiltonian masses are unequal in {family}")
    def synthetic(family:str,group:str,values:Sequence[float],weights:Sequence[float])->dict[str,Any]:
        return {"family":family,"group_id":group,"x":np.linspace(0.,1.,len(values)),
                "V_raw":np.asarray(values,dtype=np.float64),
                "quadrature_weights":np.asarray(weights,dtype=np.float64),"potential_valid_mask":np.ones(len(values),dtype=bool),
                "domain_left":0.0,"domain_right":1.0,"kinetic_coefficient":.5,
                "operator_window":{"potential_ceiling":float(max(values))}}
    rows=[synthetic("a","a0",[0,2,6],[1,2,1]),synthetic("a","a1",[0,3,8],[1,2,1]),synthetic("b","b0",[0,4,10],[1,2,1])]
    duplicated=copy.deepcopy(rows); duplicated[0]=synthetic("a","a0",[0,2,2,6],[1,1,1,1])
    scale,audit=fit_dimensionless_u_scale(rows); duplicate_scale,duplicate_audit=fit_dimensionless_u_scale(duplicated)
    _assert(scale==duplicate_scale and abs(audit["total_assigned_mass"]-1)<1e-12 and abs(duplicate_audit["total_assigned_mass"]-1)<1e-12,
            "quadrature-mass-preserving node duplication changed the fitted scale")
    _assert(NORMALIZATION_HASH==hashlib.sha256(json.dumps(NORMALIZATION_CONTRACT,sort_keys=True).encode()).hexdigest())
run_test("Hierarchical train-only U normalization",test_hierarchical_u_normalization)


def test_state_ramps_and_physics_hysteresis():
    values=torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True); mask=torch.ones(2,3,dtype=torch.bool)
    reduced=ramped_state_mean(values,mask,torch.tensor([1.,.5,0.])); _assert(torch.allclose(reduced,torch.tensor(8.5/6)))
    _assert(torch.allclose(ramped_state_mean(values,mask,torch.full((3,),.5)),values.mean()*.5))
    zero=ramped_state_mean(values,mask,torch.zeros(3)); zero.backward(); _assert(torch.count_nonzero(values.grad)==0,"zero ramp leaked a gradient")
    original=initial_physics_curriculum(); first=update_physics_curriculum(original,{"fidelity_s0":.71,"fidelity_s1":.69,"fidelity_s2":.8},2)
    second=update_physics_curriculum(first,{"fidelity_s0":.72,"fidelity_s1":.69,"fidelity_s2":.9},2)
    _assert(not any(original["enabled"]) and second["enabled"][0] and second["age"][0]==1 and not second["enabled"][1] and not second["enabled"][2])
    retained=update_physics_curriculum(second,{"fidelity_s0":.60,"fidelity_s1":0.,"fidelity_s2":0.},2)
    stopped=update_physics_curriculum(retained,{"fidelity_s0":.54,"fidelity_s1":0.,"fidelity_s2":0.},2)
    _assert(retained["enabled"][0] and retained["age"][0]==2 and not stopped["enabled"][0] and stopped["age"][0]==0)
    # Test the allocation algorithm on stable reference totals, independently of the
    # mutable mode budgets.  Then validate each preset by contract so increasing a run
    # budget cannot make this unrelated physics-hysteresis test stale.
    reference_allocations={3:(1,1,1),12:(3,4,5),18:(4,6,8),30:(8,10,12),40:(10,13,17)}
    for total,expected in reference_allocations.items():
        _assert(progressive_supervised_epochs(total)==expected,
                f"progressive epoch allocator changed at total={total}")
    for mode,preset in PRESETS.items():
        allocation=progressive_supervised_epochs(preset.supervised_epochs)
        _assert(sum(allocation)==preset.supervised_epochs and min(allocation)>=1 and
                allocation[0]<=allocation[1]<=allocation[2],
                f"{mode} progressive allocation is invalid: total={preset.supervised_epochs}, allocation={allocation}")
    ready={**{f"fidelity_s{index}":CFG.ritz_gradient_start_fidelity+.01 for index in range(CFG.k_states)},
           "subspace_projector_p11":CFG.ritz_gradient_start_projector_loss-.01,
           "ritz_gradient_safe_fraction":CFG.ritz_gradient_min_safe_fraction+.01}
    _assert(ritz_gradient_validation_ready(ready)); ready["fidelity_s10"]=CFG.ritz_gradient_start_fidelity-.01
    _assert(not ritz_gradient_validation_ready(ready),"Ritz readiness ignored the worst state")
run_test("Per-state physics hysteresis and progressive epochs",test_state_ramps_and_physics_hysteresis)


def test_prefix_projector_invariance():
    K=CFG.k_states; N=2*K; w=torch.full((1,N),1/N); target=torch.zeros(1,K,N); target[0,torch.arange(K),torch.arange(K)]=math.sqrt(N)
    # Independent rotations inside [0:3], [3:6], and [6:11] preserve every nested
    # prefix projector while exercising arbitrary signs and basis choices within it.
    rotation=torch.block_diag(*(torch.linalg.qr(torch.randn(size,size))[0] for size in (3,3,5)))
    pred=torch.einsum("ij,bjn->bin",rotation,target); mask=torch.ones(1,K,dtype=torch.bool)
    loss,diagnostics=prefix_projector_loss(pred,target,w,mask,K); _assert(float(loss)<2e-6 and set(diagnostics)=={f"subspace_projector_p{p}" for p in (3,6,11)}|{f"subspace_fidelity_p{p}" for p in (3,6,11)})
    disjoint=torch.zeros_like(target); disjoint[0,torch.arange(K),K+torch.arange(K)]=math.sqrt(N)
    disjoint_loss,_=prefix_projector_loss(disjoint,target,w,mask,K); _assert(float(disjoint_loss)>.999)
    low_loss,low_diagnostics=prefix_projector_loss(pred,target,w,mask,3); altered=pred.clone(); altered[:,3:]=disjoint[:,3:]
    altered_loss,_=prefix_projector_loss(altered,target,w,mask,3)
    _assert(torch.allclose(low_loss,altered_loss) and set(low_diagnostics)=={"subspace_projector_p3","subspace_fidelity_p3"},"inactive states affected the p3 projector")
    perturbed=(.95*target+.2*disjoint).requires_grad_(True); gradient_loss,_=prefix_projector_loss(perturbed,target,w,mask,K); gradient_loss.backward()
    _assert(torch.isfinite(perturbed.grad).all() and float(perturbed.grad.abs().sum())>0,"projector gradient is invalid")
run_test("Nested projector rotation invariance and isolation",test_prefix_projector_invariance)


def test_selective_ritz_eigenvector_backward():
    projected=torch.stack([torch.diag(torch.tensor([0.,0.,2.,4.])),torch.diag(torch.tensor([0.,1.,2.,4.]))]).requires_grad_(True)
    epsilon,rotation,_,safe=stable_eigh_rotation(projected,detach_vectors=False)
    _assert(safe.tolist()==[False,True],f"unexpected Ritz safety mask {safe.tolist()}")
    unsafe_vector_grad=torch.autograd.grad(rotation[0].sum(),projected,retain_graph=True)[0]
    _assert(torch.count_nonzero(unsafe_vector_grad[0])==0,"degenerate sample retained an eigenvector-gradient path")
    (epsilon.sum()+rotation[1].sum()).backward(); _assert(torch.isfinite(projected.grad).all(),"mixed safe/degenerate Ritz backward produced non-finite gradients")
run_test("Selective Ritz gradients survive exact degeneracy",test_selective_ritz_eigenvector_backward)


def test_independent_module_clipping_and_accumulation():
    groups=curriculum_module_parameter_groups(model); target_ids={id(p) for p in model.solution_target_encoder.parameters()}
    _assert(not target_ids&{id(p) for parameters in groups.values() for p in parameters})
    large=nn.Parameter(torch.zeros(4)); small=nn.Parameter(torch.zeros(4)); inactive=nn.Parameter(torch.zeros(2))
    large.grad=torch.full_like(large,100.); small.grad=torch.full_like(small,.1); small_before=small.grad.clone()
    report=clip_module_gradients({"large":[large],"small":[small],"inactive":[inactive]},clip_rms=.5,hard_norm=1000)
    _assert(report["large"]["clipped"] and not report["small"]["clipped"] and not report["inactive"]["active"])
    _assert(torch.allclose(small.grad,small_before) and abs(float(large.grad.norm())-report["large"]["post"])<2e-6)
    _assert(report["clipped_module_events"]==1 and report["active_module_events"]==2 and report["any_clipped"])
    base=nn.Linear(3,2,bias=False); accumulated=copy.deepcopy(base); combined=copy.deepcopy(base)
    opt_a=torch.optim.SGD(accumulated.parameters(),lr=.1); opt_b=torch.optim.SGD(combined.parameters(),lr=.1)
    x1=torch.tensor([[1.,2.,3.]]); x2=torch.tensor([[2.,-1.,.5]])
    opt_a.zero_grad(); (accumulated(x1).square().mean()/2).backward(); (accumulated(x2).square().mean()/2).backward()
    clip_module_gradients({"linear":list(accumulated.parameters())},clip_rms=1e6); opt_a.step()
    opt_b.zero_grad(); ((combined(x1).square().mean()+combined(x2).square().mean())/2).backward()
    clip_module_gradients({"linear":list(combined.parameters())},clip_rms=1e6); opt_b.step()
    _assert(all(torch.allclose(a,b,atol=1e-7,rtol=1e-7) for a,b in zip(accumulated.parameters(),combined.parameters())),"accumulation changed the update")
    bad=nn.Parameter(torch.zeros(1)); bad.grad=torch.tensor([float("nan")])
    try: clip_module_gradients({"bad":[bad]})
    except RuntimeError: pass
    else: raise AssertionError("non-finite module gradient did not fail loudly")
run_test("Independent module clipping and accumulation",test_independent_module_clipping_and_accumulation)


def test_active_state_loss_isolation():
    b=to_device(collate_records([splits["train"][0]],include_targets=True)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad(): out=m.forward_training(b,detach_ritz_vectors=True)
    selected={"jepa":1.,"subspace_projector":1.,"wave":1.,"infidelity":1.,"density":1.,"h1":1.,
              "energy":1.,"gap":1.,"energy_ground_exact":1.,"energy_log_gap_exact":1.,
              "energy_spectrum_exact":1.,"topology_node_phase":1.,"topology_lobe_mass":1.,
              "topology_phase_cdf_supervision":1.,"topology_log_amplitude_supervision":1.}
    loss_a,terms_a=compute_losses(out,b,selected,loss_model=m,active_states=3,physics_state_weights=torch.zeros(CFG.k_states,device=DEVICE))
    altered_batch={key:(value.clone() if torch.is_tensor(value) else value) for key,value in b.items()}; altered_out=dict(out)
    altered_batch["psi"][:,3:]+=100*torch.randn_like(altered_batch["psi"][:,3:]); altered_batch["energy_dimensionless"][:,3:]+=1e5
    altered_batch["target_node_positions"][:,3:]+=1e5; altered_batch["target_node_mask"][:,3:]=True
    altered_batch["target_phase_cdf"][:,3:]+=1e5; altered_batch["target_phase_cdf_mask"][:,3:]=True
    altered_batch["target_log_amplitude"][:,3:]-=1e5; altered_batch["target_log_amplitude_mask"][:,3:]=True
    altered_out["z_target"]=out["z_target"].clone(); altered_out["z_target"][:,3:]+=1e5
    loss_b,terms_b=compute_losses(altered_out,altered_batch,selected,loss_model=m,active_states=3,physics_state_weights=torch.zeros(CFG.k_states,device=DEVICE))
    _assert(torch.allclose(loss_a,loss_b,atol=1e-6,rtol=1e-6),"inactive targets changed the active objective")
    for name in selected: _assert(torch.allclose(terms_a[name],terms_b[name],atol=1e-6,rtol=1e-6),f"inactive targets changed {name}")
run_test("Active-state losses exclude inactive targets",test_active_state_loss_isolation)


box_record=next(r for r in records if r["family"]=="infinite_box")
box_spec=PotentialSpec("infinite_box",box_record["parameters"],box_record["geometry"],box_record["boundary_condition"],box_record["domain_left"],box_record["domain_right"],None)
run_test("Infinite-box analytical energies",lambda:_assert(np.max(np.abs(box_record["energy"]-analytical_energies(box_spec))/np.maximum(1,np.abs(analytical_energies(box_spec))))<caps["analytic_rel"]))
def test_box_waves():
    exact=analytical_wavefunctions(box_spec,box_record["x"],(0,5,10)); fid=[sign_invariant_fidelity_np(box_record["psi"][[s]],exact[[i]],box_record["quadrature_weights"])[0] for i,s in enumerate((0,5,10))]; _assert(min(fid)>0.95,str(fid))
run_test("Infinite-box analytical wavefunctions",test_box_waves)
run_test("Harmonic-oscillator energies",lambda:_assert(solver_validation.loc[solver_validation.family=="harmonic","analytic_energy_rel_max"].max()<caps["analytic_rel"]))
def test_half_full():
    full=make_spec("harmonic",np.full(16,.5)); half=make_spec("half_harmonic",np.full(16,.5)); _assert(np.allclose(analytical_energies(half),analytical_energies(full)[2*np.arange(CFG.k_states)+1] if len(analytical_energies(full,22))>21 else analytical_energies(half)))
# Direct formula equivalence avoids truncating the full array at 11.
run_test("Half/full oscillator equivalence",lambda:_assert(np.allclose(analytical_energies(make_spec("half_harmonic",np.full(16,.5))),analytical_energies(make_spec("harmonic",np.full(16,.5)),22)[2*np.arange(CFG.k_states)+1])))
run_test("Radial analytical cases",lambda:_assert(solver_validation.loc[solver_validation.family.isin(["radial_coulomb","radial_oscillator"]),"analytic_energy_rel_max"].max()<caps["analytic_rel"]))
run_test("Reference normalization",lambda:_assert(max(r["reference_diagnostics"]["normalization_error"] for r in records)<1e-10))
run_test("Reference orthogonality",lambda:_assert(max(r["reference_diagnostics"]["gram_error"] for r in records)<2e-8))
run_test("Correct node counts",lambda:_assert(all(np.array_equal(r["node_count"],np.arange(CFG.k_states)) for r in records)))
run_test("Grid refinement",lambda:_assert(solver_validation.passes_gate.all()))
run_test("Domain enlargement",lambda:_assert(solver_validation.domain_rel_max.max()<caps["domain_rel"]))
run_test("Reference-to-model-grid projection",lambda:_assert(all(r["model_grid_diagnostics"]["passes_projection_gate"] for r in records)))
run_test("Model-grid state resolution",lambda:_assert(min(r["model_grid_diagnostics"]["projection_fidelity_min"] for r in records)>max(.75,1-5*caps["infidelity"])))
def test_sign_invariance():
    b=to_device(next(iter(loaders["train"]))); p=b["psi"].clone().requires_grad_(True)
    aligned1,f1=globally_align_torch(p,b["psi"],b["w"]); aligned2,f2=globally_align_torch(p,-b["psi"],b["w"])
    _assert(torch.allclose(f1,f2,atol=1e-6)); _assert(torch.allclose(((aligned1-b["psi"])**2).mean(),((aligned2+b["psi"])**2).mean(),atol=1e-6))
    # Any signed field paired with a state (residual, tangent, correction) must use
    # the same gauge before comparison; raw eigenvector signs are not observable.
    imposed=torch.where(torch.arange(CFG.k_states,device=DEVICE)%2==0,1.0,-1.0)[None,:,None]
    paired=.37*p.detach(); flipped_psi=p.detach()*imposed; flipped_paired=paired*imposed
    recovered,_=global_sign_alignment_torch(flipped_psi,b["psi"],b["w"])
    _assert(torch.allclose(flipped_paired*recovered[...,None],paired,atol=1e-7,rtol=1e-7),
            "paired signed fields did not follow the recovered eigenvector gauge")
run_test("Global-sign invariance",test_sign_invariance)
run_test("Exact rho = psi squared",lambda:_assert(all(np.array_equal(r["rho"],r["psi"]**2) for r in records)))
def test_physical_boundaries_and_tail_geometry():
    _assert(all(r["x"][0]==r["domain_left"] and r["x"][-1]==r["domain_right"] and np.max(np.abs(r["psi"][:,[0,-1]]))==0 for r in records))
    finite=next(r for r in records if r["geometry"]=="finite_interval"); truncated=next(r for r in records if r["geometry"]=="truncated_line")
    masks=physics_tail_mask(collate_records([finite,truncated],include_targets=False)); _assert(not bool(masks[0].any()) and bool(masks[1].any()),"tail loss mask was applied to a physical finite boundary")
run_test("Correct physical boundary locations",test_physical_boundaries_and_tail_geometry)
run_test("No group leakage",lambda:_assert(not(split_sets["train"]&split_sets["val"] or split_sets["train"]&split_sets["test"] or split_sets["val"]&split_sets["test"])))
def test_parent_leakage():
    parents={name:{r["augmentation_parent"] for r in rows} for name,rows in splits.items()}
    _assert(not(parents["train"]&parents["val"] or parents["train"]&parents["test"] or parents["val"]&parents["test"]))
    derivative_parents={r["augmentation_parent"] for category in ("constant_potential_shift","valid_spatial_reflection","variable_uniform_resolution","nonuniform_resolution","changed_domain_same_physical_operator") for r in ood_suites[category]}
    _assert(not derivative_parents&parents["train"] and not derivative_parents&parents["val"],"test-derived OOD augmentation parent leaked into fitting splits")
run_test("No augmentation-parent leakage",test_parent_leakage)
run_test("No exact duplicate leakage",lambda:_assert(len({r["potential_fingerprint"] for r in records})==len(records)))
def test_near_duplicate_signature_semantics():
    base=next(r for r in records if r["family"]=="harmonic"); shifted=copy.deepcopy(base); scaled=copy.deepcopy(base)
    shifted["V_raw"]=np.asarray(base["V_raw"])+3.25; scaled["V_raw"]=2*np.asarray(base["V_raw"])
    original_signature=normalized_signature(base); shifted_signature=normalized_signature(shifted); scaled_signature=normalized_signature(scaled)
    _assert(float(np.linalg.norm(original_signature-shifted_signature))<1e-10,"constant potential shifts must not change the centered signature")
    _assert(float(np.linalg.norm(original_signature-scaled_signature))>1e-2,"potential-strength changes must remain visible")
    reflected_signature=reflected_normalized_signature(original_signature)
    _assert(np.array_equal(reflected_signature[-2:],original_signature[-2:]),"reflection must not reverse signature metadata")
run_test("Near-duplicate signature semantics",test_near_duplicate_signature_semantics)
run_test("Near-duplicate checks",lambda:_assert(near_duplicate_min>NEAR_DUPLICATE_DISTANCE_MIN,json.dumps(near_duplicate_audit,sort_keys=True)))


def target_free_predict(m:TrueSchrodingerJEPA,operator_record:Mapping[str,Any])->dict[str,np.ndarray]:
    forbidden={"psi","rho","energy","valid_state_mask","node_count","family","parameters","bound_margin","solver_residual",
               "target_node_positions","target_node_mask","target_phase_cdf","target_phase_cdf_mask",
               "target_log_amplitude","target_log_amplitude_mask","state_mask","z_target","target_latent"}
    _assert(not(forbidden&set(operator_record)),f"forbidden inference keys: {forbidden&set(operator_record)}")
    batch=to_device(collate_records([operator_record],include_targets=False)); m.eval()
    with torch.no_grad(): out=m.forward_operator(batch)
    keys=("energy","psi","rho","energy_direct","psi_direct","rho_direct","energy_hybrid","psi_hybrid","rho_hybrid",
          "hybrid_topology_valid_by_state","hybrid_fallback_recommended")
    result={key:out[key][0].cpu().numpy() for key in keys}
    for key in ("amplitude","log_amplitude","phase_density","phase_cdf","theta","phase_increment","predicted_node_positions_t"):
        if key in out: result[key]=out[key][0].cpu().numpy()
    return result


def test_shift_covariance():
    base=operator_only_record(base_test); shifted=operator_only_record(shifted_record(base_test,2.75)); model.eval()
    p0=target_free_predict(model,base); p1=target_free_predict(model,shifted)
    _assert(np.allclose(p0["psi"],p1["psi"],atol=2e-5)); _assert(np.allclose(p0["rho"],p1["rho"],atol=2e-5)); _assert(np.allclose(p1["energy"],p0["energy"]+2.75,atol=2e-5))
run_test("Constant potential-shift covariance",test_shift_covariance)
def test_translation_and_dilation_covariance():
    base_record=next(r for r in splits["test"] if r["geometry"]=="truncated_line"); base=operator_only_record(base_record); length=base_record["domain_right"]-base_record["domain_left"]
    translated_target=translated_record(base_record,.13*length); scale=1.11; dilated_target=dilated_record(base_record,scale)
    translated=operator_only_record(translated_target); dilated=operator_only_record(dilated_target); model.eval()
    topology_targets=collate_records([base_record,translated_target,dilated_target],include_targets=True)
    _assert(torch.allclose(topology_targets["target_phase_cdf"][0],topology_targets["target_phase_cdf"][1],atol=3e-6,rtol=3e-6) and
            torch.allclose(topology_targets["target_phase_cdf"][0],topology_targets["target_phase_cdf"][2],atol=3e-6,rtol=3e-6),
            "target phase CDF lost translation/dilation covariance")
    _assert(torch.allclose(topology_targets["target_log_amplitude"][0],topology_targets["target_log_amplitude"][1],atol=3e-5,rtol=3e-5) and
            torch.allclose(topology_targets["target_log_amplitude"][0],topology_targets["target_log_amplitude"][2],atol=3e-5,rtol=3e-5) and
            torch.equal(topology_targets["target_log_amplitude_mask"][0],topology_targets["target_log_amplitude_mask"][1]) and
            torch.equal(topology_targets["target_log_amplitude_mask"][0],topology_targets["target_log_amplitude_mask"][2]),
            "target log amplitude lost translation/dilation covariance")
    p0=target_free_predict(model,base); pt=target_free_predict(model,translated); pdil=target_free_predict(model,dilated)
    translation_sign=np.where(np.sum(pt["psi"]*p0["psi"],axis=1)>=0,1.0,-1.0)
    _assert(np.allclose(pt["psi"]*translation_sign[:,None],p0["psi"],atol=3e-5) and np.allclose(p0["rho"],pt["rho"],atol=3e-5) and np.allclose(p0["energy"],pt["energy"],atol=3e-5),"translation covariance")
    dilation_sign=np.where(np.sum(pdil["psi"]*p0["psi"],axis=1)>=0,1.0,-1.0)
    _assert(np.allclose(pdil["psi"]*dilation_sign[:,None],p0["psi"]/math.sqrt(scale),atol=4e-5,rtol=4e-5),"wavefunction dilation covariance")
    _assert(np.allclose(pdil["energy"],p0["energy"]/(scale*scale),atol=4e-5,rtol=4e-5),"energy dilation covariance")
run_test("Translation and Schrödinger-dilation covariance",test_translation_and_dilation_covariance)
def test_exact_reflection_roundtrip():
    original=next(r for r in splits["train"] if r["geometry"]=="truncated_line"); roundtrip=reflected_record(reflected_record(original))
    _assert(np.allclose(roundtrip["x"],original["x"]) and np.allclose(roundtrip["V_raw"],original["V_raw"]))
    _assert(np.allclose(roundtrip["psi"],original["psi"],atol=1e-12) and np.allclose(weighted_gram(roundtrip["psi"],roundtrip["quadrature_weights"]),np.eye(CFG.k_states),atol=2e-8))
run_test("Exact reflection round trip",test_exact_reflection_roundtrip)
def test_target_free_modes():
    _assert(target_free_predict(model,operator_only_record(base_test))["psi"].shape==(CFG.k_states,len(base_test["x"])))
    inference_batch=collate_records([operator_only_record(base_test)],include_targets=False)
    _assert(not(set(FORBIDDEN_DIRECT_KEYS)&set(inference_batch)),"target-only tensors entered inference collation")
    missing=collate_records([operator_only_record(base_test)],include_targets=False,use_continuum_threshold=True)
    _assert(torch.count_nonzero(missing["features"][...,-2:])==0,"absent threshold channels must both be zero")
    assisted=to_device(collate_records([operator_only_record(base_test,include_continuum_threshold=True)],include_targets=False,use_continuum_threshold=True))
    assisted_model=TrueSchrodingerJEPA(in_features=assisted["features"].shape[-1]).to(DEVICE).eval()
    with torch.no_grad(): assisted_out=assisted_model.forward_operator(assisted)
    _assert(torch.isfinite(assisted_out["psi"]).all() and assisted_out["psi"].shape[-1]==len(base_test["x"]))
run_test("Target-free inference",test_target_free_modes)
run_test("Target encoder has no gradients",lambda:_assert(all(not p.requires_grad and p.grad is None for p in model.solution_target_encoder.parameters())))
run_test("Target encoder distinct from online",lambda:_assert(model.solution_target_encoder is not model.solution_online_encoder))
run_test("Target encoder distinct from potential",lambda:_assert(model.solution_target_encoder is not model.potential_context_encoder))
def test_ema():
    online=SolutionEncoder(CFG.preset.hidden); target=initialize_ema_target(online); optimizer=torch.optim.SGD(online.parameters(),lr=1e-2)
    optimizer_ids={id(p) for group in optimizer.param_groups for p in group["params"]}; target_ids={id(p) for p in target.parameters()}
    _assert(not optimizer_ids&target_ids and all(not p.requires_grad for p in target.parameters()),"EMA target was not excluded from optimization")
    online_before={name:p.detach().clone() for name,p in online.named_parameters()}; target_before={name:p.detach().clone() for name,p in target.named_parameters()}
    optimizer.zero_grad(set_to_none=True); sum(p.square().mean() for p in online.parameters()).backward(); optimizer.step()
    _assert(any(not torch.equal(p.detach(),online_before[name]) for name,p in online.named_parameters()),"optimizer did not update online encoder")
    _assert(all(torch.equal(p.detach(),target_before[name]) and p.grad is None for name,p in target.named_parameters()),"optimizer changed EMA target")
    tau=.9; ema_update(online,target,tau)
    for name,p in target.named_parameters():
        expected=tau*target_before[name]+(1-tau)*dict(online.named_parameters())[name].detach()
        _assert(torch.allclose(p,expected,atol=1e-7,rtol=1e-6),f"EMA mismatch for {name}")
run_test("EMA target parameters update after optimizer steps",test_ema)
def test_decoder_route():
    # Keep this routing/privacy test tiny to avoid GPU OOM from a full augmented train batch.
    gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    b=to_device(collate_records([splits["train"][0]],include_targets=True)); seen=[]; energy_seen=[]
    def capture(module,args,output): seen.append((args[0].data_ptr(),set(args[1])))
    def capture_energy(module,args,output): energy_seen.append((args[0].data_ptr(),args[1].data_ptr(),set(args[2])))
    hook=model.wavefunction_decoder.register_forward_hook(capture)
    energy_hook=model.energy_head.register_forward_hook(capture_energy)
    model.eval()
    try:
        with torch.no_grad(): out=model.forward_training(b,refinement_steps=0)
    finally:
        hook.remove(); energy_hook.remove()
    expected=out["z_nodes"].data_ptr() if model.decoder_type=="topology_phase" else out["z_pred"].data_ptr()
    _assert(seen==[(expected,set(OPERATOR_BATCH_KEYS))] and out["z_target"].data_ptr()!=seen[0][0],"decoder route was not nodewise and target-free")
    _assert(not(set(FORBIDDEN_DIRECT_KEYS)&seen[0][1]) and "operator_records" not in seen[0][1],"decoder received a target capability")
    _assert(len(energy_seen)==1 and energy_seen[0][0]==out["state_tokens"].data_ptr() and
            energy_seen[0][1]==out["global_context"].data_ptr() and energy_seen[0][2]==set(OPERATOR_BATCH_KEYS),
            "energy head did not receive only predictor latents and operator metadata")
    _assert(not(set(FORBIDDEN_DIRECT_KEYS)&energy_seen[0][2]),"exact energy or solution data reached the learned energy head")
run_test("No exact target latent reaches main decoder",test_decoder_route)


def test_topology_configuration_and_parameters():
    _assert(CFG.decoder_type=="topology_phase" and CFG.primary_output=="physics_selected")
    curriculum=build_curriculum_stage_definitions(); early=next(stage for stage in curriculum if stage["name"]=="E_subspace_3")
    required_statewise={"wave","infidelity","density","h1","energy_ground_exact","energy_log_gap_exact",
                        "energy_spectrum_exact","topology_node_phase","topology_lobe_mass",
                        "topology_phase_cdf_supervision","topology_log_amplitude_supervision"}
    _assert(required_statewise<=set(early["weights"]) and early["active_states"]==3 and
            all(float(early["weights"][name])>0 for name in required_statewise),
            "statewise topology supervision is not active from the three-state stage")
    _assert(math.isclose(CURRICULUM_MODULE_LR_MULTIPLIERS["wavefunction_decoder"],1.0,rel_tol=0,abs_tol=0) and
            math.isclose(CFG.decoder_learning_rate_multiplier,1.0,rel_tol=0,abs_tol=0),
            "topology decoder LR multiplier is not the requested 1.0")
    _assert(set(TINY_OVERFIT_POLICIES)==set(RUN_MODE_OPTIONS),"tiny-overfit policies do not cover every run mode")
    _assert(TINY_OVERFIT_POLICIES["DEVELOPMENT"].fidelity_min>=.90 and
            TINY_OVERFIT_POLICIES["QUICK"].max_steps<=TINY_OVERFIT_POLICIES["DEVELOPMENT"].max_steps<=
            TINY_OVERFIT_POLICIES["STANDARD"].max_steps<=TINY_OVERFIT_POLICIES["FULL"].max_steps,
            "tiny-overfit fidelity/budget policy is inconsistent across modes")
    _assert(CONFIG_DICT["tiny_overfit_policy"]==asdict(CFG.tiny_overfit_policy) and
            CONFIG_DICT["tiny_overfit_seed"]==TINY_OVERFIT_SEED,"tiny-overfit provenance is missing")
    _assert(math.isclose(trapezoid_integral_1d(np.asarray([0.,1.,0.]),np.asarray([0.,.5,1.])),.5,abs_tol=1e-15),
            "version-independent trapezoidal integration is incorrect")
    _assert(set(decoder_parameter_counts.decoder_type)==set(DECODER_OPTIONS) and
            not bool(decoder_parameter_counts.parameter_matched.any()),"decoder parameter accounting is incomplete")
    _assert(int(decoder_parameter_counts.total_parameters.nunique())>1,"parameter difference was hidden")
    invalid=({"decoder_type":"unknown"},{"phase_q_min":0.0},{"phase_q_min":float("nan")},
             {"phase_logit_clip":float("inf")},{"amplitude_log_clip":0.0},{"amplitude_log_clip":float("nan")},
             {"minimum_intervals_per_lobe":1},{"minimum_intervals_per_lobe":2.5},
             {"topology_min_lobe_mass":float("nan")},{"topology_node_loss_weight":-1.0},
             {"topology_phase_cdf_loss_weight":-1.0},{"topology_log_amplitude_loss_weight":float("nan")},
             {"topology_target_carrier_floor":0.0},{"topology_target_carrier_floor":1.0},
             {"topology_amplitude_target_clip_margin":0.0},{"topology_amplitude_target_clip_margin":6.0},
             {"decoder_learning_rate_multiplier":0.0},{"energy_ground_residual_clip":0.0},
             {"energy_log_gap_clip":float("nan")},{"energy_scale_floor":0.0},
             {"energy_reference_gap_floor":-1.0},{"energy_ground_loss_weight":-1.0},
             {"energy_rayleigh_consistency_weight":float("nan")})
    for kwargs in invalid:
        try: Config(**kwargs)
        except ValueError: pass
        else: raise AssertionError(f"invalid topology configuration was accepted: {kwargs}")
    malformed=operator_only_record(splits["train"][0]); malformed["x"]=np.asarray(malformed["x"]).copy(); malformed["x"][3]=malformed["x"][2]
    try: collate_records([malformed],include_targets=False)
    except ValueError: pass
    else: raise AssertionError("nonmonotone inference grid was silently accepted")
run_test("Topology decoder configuration and parameter accounting",test_topology_configuration_and_parameters)


ENERGY_COORDINATE_AUDIT=exact_energy_coordinate_audit(splits["train"])
def test_exact_energy_coordinate_coverage():
    audit=ENERGY_COORDINATE_AUDIT
    # This runs on ``DEVICE``: CPU QUICK checks the statistic, while a CUDA notebook
    # also proves that strict deterministic mode accepts the exact execution path.
    even_probe=torch.tensor([[9.,1.,5.,3.],[4.,4.,4.,4.]],device=DEVICE)
    odd_probe=torch.tensor([[9.,1.,5.,3.,7.]],device=DEVICE)
    prefix_probe=torch.tensor([[1.,2.,3.,4.,5.]],device=DEVICE)
    _assert(torch.equal(deterministic_middle_median(even_probe,dim=1),
                        torch.tensor([3.,4.],device=DEVICE)) and
            torch.equal(deterministic_middle_median(odd_probe,dim=1),
                        torch.tensor([5.],device=DEVICE)) and
            torch.equal(deterministic_prefix_sum(prefix_probe,dim=1),
                        torch.tensor([[1.,3.,6.,10.,15.]],device=DEVICE)),
            "deterministic value-only median or prefix sum is incorrect")
    _assert(audit["all_finite"] and audit["ground_clip_hit_fraction"]==0 and audit["log_gap_clip_hit_fraction"]==0,
            f"exact energy targets exceed the representable head coordinates: {audit}")
    _assert(audit["ground_residual_scaled"]["absolute_maximum"]<audit["ground_clip"] and
            audit["log_gap_ratio"]["absolute_maximum"]<audit["log_gap_clip"] and
            audit["exact_gap_minimum"]>0 and audit["reference_gap_minimum"]>0,
            f"invalid operator-scaled energy-coordinate audit: {audit}")
run_test("Exact spectra fit inside operator-scaled ground/log-gap coordinates",test_exact_energy_coordinate_coverage)
print("energy-coordinate audit",ENERGY_COORDINATE_AUDIT)


def test_topology_decoder_structural_invariants():
    shortest=min(splits["train"],key=lambda record:len(record["x"])); longest=max(splits["train"],key=lambda record:len(record["x"]))
    rows=[operator_only_record(longest),operator_only_record(shortest)]
    b=to_device(collate_records(rows,include_targets=False)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad(): out=m.forward_operator(b,refinement_steps=0)
    required=("psi_direct","rho_direct","amplitude","log_amplitude","phase_density_raw","phase_density","phase_cdf","theta","phase_increment")
    _assert(all(torch.isfinite(out[key]).all() for key in required),"topology decoder produced non-finite output")
    _assert(torch.equal(out["rho_direct"],out["psi_direct"].square()),"rho_direct is not identically psi_direct squared")
    norm=(b["w"][:,None]*out["psi_direct"].square()).sum(-1)
    _assert(torch.allclose(norm,torch.ones_like(norm),atol=3e-5,rtol=3e-5),"physical normalization failed")
    _assert(torch.count_nonzero(out["psi_direct"].masked_select(b["boundary_mask"][:,None]))==0,"physical boundary was not exactly zero")
    interior=(b["node_mask"]&~b["boundary_mask"])[:,None].expand_as(out["amplitude"])
    _assert(bool((out["amplitude"].masked_select(interior)>0).all()),"interior amplitude is not strictly positive")
    interval=(b["node_mask"][:,1:]&b["node_mask"][:,:-1])[:,None].expand_as(out["phase_increment"])
    cdf_difference=out["phase_cdf"][...,1:]-out["phase_cdf"][...,:-1]
    _assert(bool((cdf_difference.masked_select(interval)>0).all()),"phase CDF is not strictly increasing")
    dt=b["t"][:,1:]-b["t"][:,:-1]
    q_integral=(.5*(out["phase_density"][...,1:]+out["phase_density"][...,:-1])*dt[:,None]*interval).sum(-1)
    _assert(torch.allclose(q_integral,torch.ones_like(q_integral),atol=3e-6,rtol=3e-6),"normalized phase density does not integrate to one")
    last=(b["lengths"]-1)[:,None,None].expand(-1,CFG.k_states,1)
    c_last=torch.gather(out["phase_cdf"],-1,last).squeeze(-1); theta_last=torch.gather(out["theta"],-1,last).squeeze(-1)
    expected_theta=math.pi*torch.arange(1,CFG.k_states+1,device=DEVICE)[None]
    _assert(torch.count_nonzero(out["phase_cdf"][...,0])==0 and torch.allclose(c_last,torch.ones_like(c_last),atol=2e-7))
    _assert(torch.allclose(theta_last,expected_theta.expand_as(theta_last),atol=3e-6,rtol=3e-6))
    expected_nodes=torch.arange(CFG.k_states,device=DEVICE)[None].expand(len(rows),-1)
    _assert(torch.equal(out["predicted_node_mask"].sum(-1),expected_nodes),"phase quantile count is not the Sturm state index")
    for bi,n in enumerate(b["lengths"].tolist()):
        for state in range(CFG.k_states):
            c=out["phase_cdf"][bi,state,:n]
            for root_index in range(1,state+1):
                level=root_index/(state+1)
                crossings=((c[:-1]<level)&(c[1:]>=level)).sum()
                _assert(int(crossings)==1,f"state {state} did not have one unique phase crossing at j={root_index}")
            if float(out["maximum_phase_increment"][bi,state])<=math.pi/CFG.minimum_intervals_per_lobe+1e-6:
                _assert(persistent_nodes(out["psi_direct"][bi,state,:n].cpu().numpy())==state,
                        f"sampled topology failed despite resolution gate for state {state}")
        for key in ("psi_direct","rho_direct","amplitude","log_amplitude","phase_density_raw","phase_density","phase_cdf","theta"):
            _assert(torch.count_nonzero(out[key][bi,:,n:])==0,f"{key} leaked into padding")
        _assert(torch.count_nonzero(out["phase_increment"][bi,:,max(n-1,0):])==0,"padded intervals contributed phase")
    # Adversarially large phase-head logits must remain finite, strictly monotone, and
    # normalized; this exercises the bounded-logit and representable-increment path.
    extreme=copy.deepcopy(m.wavefunction_decoder)
    with torch.no_grad():
        extreme.phase_density_head.weight.mul_(1e7); extreme.phase_density_head.bias.fill_(1e7)
        stressed=extreme(out["z_nodes"],operator_batch_view(b))
    stressed_norm=(b["w"][:,None]*stressed["psi_direct"].square()).sum(-1)
    stressed_difference=stressed["phase_cdf"][...,1:]-stressed["phase_cdf"][...,:-1]
    _assert(torch.isfinite(stressed["psi_direct"]).all() and torch.allclose(stressed_norm,torch.ones_like(stressed_norm),atol=5e-5,rtol=5e-5),
            "extreme finite phase logits broke structural normalization")
    _assert(bool((stressed_difference.masked_select(interval)>0).all()),"extreme finite phase logits produced a CDF plateau")
run_test("Topology decoder enforces positive amplitude and exact Sturm phase",test_topology_decoder_structural_invariants)


def test_topology_reflection_algebra():
    reflected=reflected_record(base_test)
    target_pair=collate_records([base_test,reflected],include_targets=True)
    _assert(torch.allclose(target_pair["target_phase_cdf"][1],1-target_pair["target_phase_cdf"][0].flip(-1),atol=4e-6,rtol=4e-6) and
            torch.allclose(target_pair["target_log_amplitude"][1],target_pair["target_log_amplitude"][0].flip(-1),atol=4e-5,rtol=4e-5),
            "target topology fields do not obey reflection covariance")
    _assert(torch.equal(target_pair["target_phase_cdf_mask"][1],target_pair["target_phase_cdf_mask"][0].flip(-1)) and
            torch.equal(target_pair["target_log_amplitude_mask"][1],target_pair["target_log_amplitude_mask"][0].flip(-1)),
            "target topology masks do not obey reflection covariance")
    b=to_device(collate_records([operator_only_record(base_test)],include_targets=False))
    b_reflected=to_device(collate_records([operator_only_record(reflected)],include_targets=False)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad():
        out=m.forward_operator(b,refinement_steps=0)
        # The mirrored state-conditioned field is the exact decoder-level reflection
        # fixture; the second call uses the independently reflected operator/grid batch.
        reflected_out=m.wavefunction_decoder(out["z_nodes"].flip(-2),operator_batch_view(b_reflected))
    amplitude=out["amplitude"]; cdf=out["phase_cdf"]; reflected_amplitude=reflected_out["amplitude"]
    reflected_cdf=reflected_out["phase_cdf"]; order=torch.arange(1,CFG.k_states+1,device=DEVICE,dtype=cdf.dtype)[None,:,None]
    reflected_theta=reflected_out["theta"]; reflected_psi=reflected_out["psi_direct"]
    state_sign=torch.where(torch.arange(CFG.k_states,device=DEVICE)%2==0,1.0,-1.0)[None,:,None]
    _assert(torch.allclose(reflected_amplitude,amplitude.flip(-1),atol=3e-6,rtol=3e-6))
    _assert(torch.allclose(reflected_cdf,1-cdf.flip(-1),atol=4e-6,rtol=4e-6))
    # Two independent forward passes can differ by a few float32 ULPs in theta.
    _assert(torch.allclose(reflected_theta,order*math.pi-out["theta"].flip(-1),atol=1e-5,rtol=1e-5))
    _assert(torch.allclose(reflected_psi,state_sign*out["psi_direct"].flip(-1),atol=4e-5,rtol=4e-5),"reflected waveform has the wrong state sign")
run_test("Topology reflection transforms amplitude phase and waveform",test_topology_reflection_algebra)


def test_topology_loss_constructed_cases():
    b=to_device(collate_records([splits["train"][0]],include_targets=True)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad(): base=m.forward_operator(b,refinement_steps=0)
    target_keys={"target_phase_cdf","target_phase_cdf_mask","target_log_amplitude","target_log_amplitude_mask"}
    _assert(target_keys<=set(b) and not target_keys&set(collate_records([operator_only_record(splits["train"][0])],include_targets=False)),
            "topology field labels crossed the target-only collation boundary")
    n=int(b["lengths"][0]); valid_states=torch.where(b["state_mask"][0])[0]
    target_cdf=b["target_phase_cdf"]; target_log_amplitude=b["target_log_amplitude"]
    _assert(torch.isfinite(target_cdf).all() and torch.isfinite(target_log_amplitude).all(),"non-finite topology targets")
    _assert(torch.count_nonzero(target_cdf[0,valid_states,0])==0 and
            torch.allclose(target_cdf[0,valid_states,n-1],torch.ones(len(valid_states),device=DEVICE)),
            "target phase CDF endpoints are incorrect")
    _assert(bool((torch.diff(target_cdf[0,valid_states,:n],dim=-1)>=0).all()),"target phase CDF is not monotone")
    _assert(not bool(b["target_log_amplitude_mask"][...,0].any()) and
            not bool(b["target_log_amplitude_mask"][...,n-1].any()) and
            not bool((b["target_log_amplitude_mask"]&b["singularity_mask"][:,None]).any()),
            "unreliable boundary/singular amplitude samples were supervised")
    target_gauge=(b["w_dimensionless"][:,None]*target_log_amplitude).sum(-1)
    _assert(float(target_gauge[0,valid_states].abs().max())<2e-5,"target log-amplitude gauge does not match the decoder")
    order=torch.arange(1,CFG.k_states+1,device=DEVICE,dtype=target_cdf.dtype)[None,:,None]
    carrier=torch.sin(order*math.pi*target_cdf); phi=torch.exp(target_log_amplitude)*carrier
    phi=phi.masked_fill(b["boundary_mask"][:,None],0.0); phi=phi/torch.sqrt((b["w_dimensionless"][:,None]*phi.square()).sum(-1,keepdim=True).clamp_min(1e-30))
    reconstructed=phi/torch.sqrt((b["endpoints"][:,1]-b["endpoints"][:,0]))[:,None,None]
    oracle_overlap=(b["w"][:,None]*reconstructed*b["psi"]).sum(-1).square()
    _assert(float(oracle_overlap[0,valid_states].min())>.95,"production topology targets cannot represent the exact waveforms")
    exact=dict(base); exact["phase_cdf"]=target_cdf; exact["log_amplitude"]=target_log_amplitude; exact["rho_direct"]=b["psi"].square()
    exact["phase_increment"]=order*math.pi*torch.diff(target_cdf,dim=-1)
    mask=b["state_mask"]; terms=topology_supervision_terms(exact,b,mask)
    _assert(float(terms["topology_node_phase"])<2e-10 and float(terms["topology_lobe_mass"])<2e-10 and
            float(terms["topology_phase_cdf_supervision"])<1e-12 and
            float(terms["topology_log_amplitude_supervision"])<1e-12,
            f"constructed exact topology losses are not zero: {terms}")
    shifted=dict(exact); shifted["phase_cdf"]=target_cdf.pow(1.25); shifted["phase_increment"]=order*math.pi*torch.diff(shifted["phase_cdf"],dim=-1)
    shifted_terms=topology_supervision_terms(shifted,b,mask)
    _assert(shifted_terms["topology_node_phase"]>terms["topology_node_phase"]+1e-5 and
            shifted_terms["topology_phase_cdf_supervision"]>terms["topology_phase_cdf_supervision"]+1e-5,
            "shifted nodes did not increase phase losses")
    amplitude_shifted=dict(exact); amplitude_shifted["log_amplitude"]=target_log_amplitude+.25*torch.cos(math.pi*b["t"][:,None])
    amplitude_terms=topology_supervision_terms(amplitude_shifted,b,mask)
    _assert(amplitude_terms["topology_log_amplitude_supervision"]>terms["topology_log_amplitude_supervision"]+1e-4,
            "shifted log amplitude did not increase field supervision")
    concentrated=dict(exact); concentrated["phase_increment"]=torch.zeros_like(exact["phase_increment"])
    concentrated["phase_increment"][...,0]=2*math.pi/CFG.minimum_intervals_per_lobe
    concentrated_terms=topology_supervision_terms(concentrated,b,mask)
    _assert(float(concentrated_terms["topology_phase_resolution"])>0,"concentrated phase did not activate the resolution penalty")
    live_cdf=shifted["phase_cdf"].detach().requires_grad_(True); live=dict(shifted); live["phase_cdf"]=live_cdf
    topology_supervision_terms(live,b,mask)["topology_node_phase"].backward()
    _assert(live_cdf.grad is not None and torch.isfinite(live_cdf.grad).all() and float(live_cdf.grad.abs().sum())>0,"node-phase interpolation lost its gradient")
    live_profile_cdf=shifted["phase_cdf"].detach().requires_grad_(True); live_profile=dict(shifted); live_profile["phase_cdf"]=live_profile_cdf
    topology_supervision_terms(live_profile,b,mask)["topology_phase_cdf_supervision"].backward()
    _assert(live_profile_cdf.grad is not None and torch.isfinite(live_profile_cdf.grad).all() and float(live_profile_cdf.grad.abs().sum())>0,
            "phase-CDF profile supervision lost its gradient")
    live_log_amplitude=amplitude_shifted["log_amplitude"].detach().requires_grad_(True); live_amplitude=dict(amplitude_shifted); live_amplitude["log_amplitude"]=live_log_amplitude
    topology_supervision_terms(live_amplitude,b,mask)["topology_log_amplitude_supervision"].backward()
    _assert(live_log_amplitude.grad is not None and torch.isfinite(live_log_amplitude.grad).all() and float(live_log_amplitude.grad.abs().sum())>0,
            "log-amplitude profile supervision lost its gradient")
    live_increment=concentrated["phase_increment"].detach().requires_grad_(True); live_resolution=dict(concentrated); live_resolution["phase_increment"]=live_increment
    topology_supervision_terms(live_resolution,b,mask)["topology_phase_resolution"].backward()
    _assert(live_increment.grad is not None and torch.isfinite(live_increment.grad).all() and float(live_increment.grad.abs().sum())>0,
            "phase-resolution regularizer lost its gradient")
run_test("Topology node lobe and phase-resolution losses have constructed-case behavior",test_topology_loss_constructed_cases)


def test_decoder_serialization_privacy_and_fallback():
    operator=operator_only_record(splits["train"][0]); inference=to_device(collate_records([operator],include_targets=False))
    _assert(not(set(FORBIDDEN_DIRECT_KEYS)&set(inference)))
    for decoder in DECODER_OPTIONS:
        primary="physics_selected"
        candidate=TrueSchrodingerJEPA(decoder_type=decoder,primary_output=primary).to(DEVICE).eval()
        with torch.no_grad(): expected=candidate.forward_operator(inference,refinement_steps=0)
        buffer=io.BytesIO(); torch.save({"decoder_type":decoder,"primary_output":primary,"state":candidate.state_dict()},buffer); buffer.seek(0)
        payload=torch.load(buffer,map_location=DEVICE,weights_only=False); clone=TrueSchrodingerJEPA(decoder_type=payload["decoder_type"],primary_output=payload["primary_output"]).to(DEVICE).eval(); clone.load_state_dict(payload["state"],strict=True)
        with torch.no_grad(): actual=clone.forward_operator(inference,refinement_steps=0)
        _assert(torch.allclose(expected["psi_direct"],actual["psi_direct"],atol=1e-7,rtol=1e-7),f"{decoder} checkpoint round trip changed inference")
    target_batch=to_device(collate_records([splits["train"][0]],include_targets=True)); altered=dict(target_batch)
    altered["target_node_positions"]=torch.randn_like(target_batch["target_node_positions"])*1e6; altered["target_node_mask"]=~target_batch["target_node_mask"]
    altered["target_phase_cdf"]=torch.randn_like(target_batch["target_phase_cdf"])*1e6
    altered["target_phase_cdf_mask"]=~target_batch["target_phase_cdf_mask"]
    altered["target_log_amplitude"]=torch.randn_like(target_batch["target_log_amplitude"])*1e6
    altered["target_log_amplitude_mask"]=~target_batch["target_log_amplitude_mask"]
    altered["psi"]=torch.randn_like(target_batch["psi"])*1e6
    altered["energy"]=torch.randn_like(target_batch["energy"])*1e6
    altered["energy_dimensionless"]=torch.randn_like(target_batch["energy_dimensionless"])*1e6
    altered["state_mask"]=~target_batch["state_mask"]
    model.eval()
    with torch.no_grad(): before=model.forward_operator(target_batch,refinement_steps=0); after=model.forward_operator(altered,refinement_steps=0)
    _assert(torch.equal(before["psi_direct"],after["psi_direct"]) and torch.equal(before["energy_direct"],after["energy_direct"]) and
            torch.equal(before["psi_physics_selected"],after["psi_physics_selected"]) and
            torch.equal(before["physics_selected_route"],after["physics_selected_route"]),
            "target-only fields altered forward_operator or route selection")
    _assert(before["physics_selected_route"].shape==(1,) and before["physics_route_scores"].shape==(1,3),
            "physics route selection was not performed once per Hamiltonian")
    wrong=before["psi_direct"][:,0:1].expand_as(before["psi_direct"]).clone(); wrong_diag=detached_topology_diagnostics(wrong,operator_batch_view(target_batch))
    physics_accept=accept_refinement_step(torch.ones(1,CFG.k_states),torch.zeros(1,CFG.k_states),torch.zeros(1,CFG.k_states),torch.zeros(1,CFG.k_states))
    selected,_,fallback=resolve_topology_hybrid(before["psi_direct"],before["energy_direct_dimensionless"],wrong,before["energy_direct_dimensionless"],
                                                torch.ones(1,dtype=torch.bool,device=DEVICE),torch.ones(1,dtype=torch.bool,device=DEVICE),
                                                wrong_diag["valid"],torch.ones(1,dtype=torch.bool,device=DEVICE),True)
    _assert(bool(physics_accept.all()) and not bool(wrong_diag["valid"].all()),"wrong-node lower-residual control is malformed")
    _assert(bool(fallback.all()) and torch.equal(selected,before["psi_direct"]),"topology-invalid hybrid did not fall back to direct")
run_test("Both decoder types checkpoint and serialize target-free inference",test_decoder_serialization_privacy_and_fallback)


def test_global_latent_and_ritz():
    b=to_device(collate_records([splits["train"][0],splits["train"][1]],include_targets=True)); model.eval()
    with torch.no_grad(): out=model.forward_training(b)
    _assert(out["z_pred"].shape==(2,CFG.k_states,CFG.latent_dim) and out["z_target"].shape==out["z_pred"].shape,"global latent shape")
    _assert(bool(torch.all(torch.diff(out["energy_direct"],dim=1)>0)),"learned direct energies are not ordered")
    energy_diagnostics=("head_ground_energy_dimensionless","head_gap_dimensionless","head_ground_residual_scaled",
                        "head_log_gap_ratio","energy_reference_dimensionless","energy_reference_gap_dimensionless",
                        "energy_operator_scale_dimensionless","energy_ground_scale_dimensionless",
                        "weak_rayleigh_energy_direct_dimensionless")
    _assert(all(torch.isfinite(out[name]).all() for name in energy_diagnostics),"operator-scaled energy diagnostics are non-finite")
    _assert(bool((out["head_gap_dimensionless"]>0).all()) and bool((out["energy_reference_gap_dimensionless"]>0).all()) and
            bool((out["energy_operator_scale_dimensionless"]>0).all()) and bool((out["energy_ground_scale_dimensionless"]>0).all()),
            "operator-scaled head lost positive gaps or scales")
    _assert(float(torch.diff(out["energy_initial_ritz"],dim=1).min())>1e-7,"projected Ritz spectrum is numerically degenerate")
    gram=torch.einsum("bkn,bn,bjn->bkj",out["psi_initial_ritz"],b["w"],out["psi_initial_ritz"]); _assert(float((gram-torch.eye(CFG.k_states,device=DEVICE)).abs().max())<2e-3,"initial Ritz states not orthonormal")
    # At zero learned coordinates the fixed sine-Galerkin spectrum is already the
    # correct infinite-box scale; this catches missing kinetic or length factors.
    box_batch=to_device(collate_records([box_record],include_targets=True)); reference=dimensionless_sine_galerkin_energy_reference(operator_batch_view(box_batch))["energy"]
    box_relative=((reference-box_batch["energy_dimensionless"]).abs()/box_batch["energy_dimensionless"].abs().clamp_min(1)).amax()
    _assert(float(box_relative)<.05,f"sine-Galerkin box reference has incorrect scale: {float(box_relative):.3e}")
    # Saturated logits remain finite and strictly ordered for both signs.
    for sign in (-1.0,1.0):
        extreme=copy.deepcopy(model.energy_head).eval()
        with torch.no_grad():
            extreme.ground_out.bias.fill_(sign*1e6); extreme.gap_out.bias.fill_(sign*1e6)
            fields=extreme(out["state_tokens"],out["global_context"],operator_batch_view(b))
        _assert(torch.isfinite(fields["epsilon"]).all() and bool((torch.diff(fields["epsilon"],dim=1)>0).all()),
                f"extreme energy logits lost finite strict ordering for sign {sign:+.0f}")

    # Construct an exact supervised case and verify each new objective independently.
    exact_out=dict(out); target=b["energy_dimensionless"].float(); scale=out["energy_operator_scale_dimensionless"].detach().float()
    ground_scale=out["energy_ground_scale_dimensionless"].detach().float()
    exact_out["energy_dimensionless"]=target; exact_out["head_energy_dimensionless"]=target
    exact_out["head_ground_residual_scaled"]=((target[:,0]-out["energy_reference_dimensionless"][:,0])/ground_scale).clamp(
        -CFG.energy_ground_residual_clip,CFG.energy_ground_residual_clip)
    exact_out["head_log_gap_ratio"]=torch.log(torch.diff(target,dim=1).clamp_min(CFG.energy_reference_gap_floor*scale[:,None])/
                                               out["energy_reference_gap_dimensionless"].clamp_min(CFG.energy_reference_gap_floor*scale[:,None])).clamp(
        -CFG.energy_log_gap_clip,CFG.energy_log_gap_clip)
    exact_out["weak_rayleigh_energy_direct_dimensionless"]=target
    energy_names=("energy_ground_exact","energy_log_gap_exact","energy_spectrum_exact",
                  "rayleigh_energy_exact","energy_rayleigh_consistency")
    _,exact_terms=compute_losses(exact_out,b,{name:1.0 for name in energy_names},loss_model=model,
                                 physics_state_weights=torch.ones(CFG.k_states,device=DEVICE))
    _assert(max(float(exact_terms[name]) for name in energy_names)<2e-7,"constructed exact energy objectives are not zero")
    perturbations={
        "energy_ground_exact":("head_ground_residual_scaled",torch.full_like(exact_out["head_ground_residual_scaled"],.5)),
        "energy_log_gap_exact":("head_log_gap_ratio",torch.full_like(exact_out["head_log_gap_ratio"],.5)),
        "energy_spectrum_exact":("head_energy_dimensionless",.5*(ground_scale[:,None]+scale[:,None]*torch.arange(1,CFG.k_states+1,device=DEVICE).square()[None])),
        "rayleigh_energy_exact":("weak_rayleigh_energy_direct_dimensionless",.5*(ground_scale[:,None]+scale[:,None]*torch.arange(1,CFG.k_states+1,device=DEVICE).square()[None])),
    }
    for name,(key,delta) in perturbations.items():
        altered=dict(exact_out); altered[key]=exact_out[key]+delta
        _,altered_terms=compute_losses(altered,b,{name:1.0,"energy_rayleigh_consistency":1.0},loss_model=model,
                                       physics_state_weights=torch.ones(CFG.k_states,device=DEVICE))
        _assert(float(altered_terms[name])>1e-3,f"{name} did not detect a deliberate energy error")
        if key in ("head_energy_dimensionless","weak_rayleigh_energy_direct_dimensionless"):
            _assert(float(altered_terms["energy_rayleigh_consistency"])>1e-3,"head/Rayleigh inconsistency was not detected")
run_test("Operator-scaled ordered energies and exact/Rayleigh consistency",test_global_latent_and_ritz)


def test_refinement_configuration():
    development=Config(mode="DEVELOPMENT",seed=7); standard=Config(mode="STANDARD",seed=7)
    for candidate in (development,standard):
        train_steps=candidate.preset.refinement_train_steps; inference_steps=candidate.preset.refinement_inference_steps
        _assert(train_steps>=3 and inference_steps>train_steps,
                f"{candidate.mode} does not provide a multi-step train/inference refinement schedule: "
                f"train={train_steps}, inference={inference_steps}")
    _assert(0<development.refinement_damping<=1 and development.refinement_backtrack_steps>=1 and development.refinement_acceptance_tolerance>=0)
    for kwargs in ({"refinement_damping":0.0},{"refinement_backtrack_steps":0},{"refinement_acceptance_tolerance":-1.0}):
        try: Config(mode="DEVELOPMENT",seed=7,**kwargs)
        except ValueError: pass
        else: raise AssertionError(f"invalid recurrent configuration was accepted: {kwargs}")
run_test("Refinement configuration supports DEVELOPMENT and STANDARD",test_refinement_configuration)


def test_zero_depth_and_zero_modulation():
    b=to_device(collate_records([operator_only_record(splits["train"][0])],include_targets=False)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad(): out=m.forward_operator(b,refinement_steps=0)
    _assert(out["refinement_psi_trace"].shape[1]==1 and out["refinement_acceptance_trace"].shape[1]==0)
    _assert(torch.equal(out["refinement_psi_trace"][:,0],out["psi_initial"]) and
            torch.equal(out["refinement_energy_dimensionless_trace"][:,0],out["energy_dimensionless_initial"]),"zero-depth refinement changed the initial Ritz solution")
    residual,analytic,weak=dimensionless_weak_residual_correction(out["psi_initial"],out["energy_dimensionless_initial"],b)
    with torch.no_grad(): modulation=m.residual_correction(out["context"],out["state_tokens"],out["psi_initial"],analytic,weak,out["energy_dimensionless_initial"],b)
    _assert(torch.allclose(modulation,torch.ones_like(modulation),atol=0,rtol=0),"zero-initialized learned preconditioner is not the analytic identity")
    _assert(torch.isfinite(residual).all() and torch.equal(out["refinement_fallback_recommended"],~out["refinement_converged"]))
    classical=TrueSchrodingerJEPA(learned_refinement=False)
    _assert(not any(p.requires_grad for p in classical.residual_correction.parameters()),"classical refiner retains unused trainable parameters")
run_test("Zero-depth and zero-learned-modulation identity",test_zero_depth_and_zero_modulation)


def test_refinement_residual_contract():
    b=to_device(collate_records([operator_only_record(splits["train"][0])],include_targets=False)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad(): out=m.forward_operator(b,refinement_steps=1)
    for iteration in range(out["refinement_psi_trace"].shape[1]):
        state=out["refinement_psi_trace"][:,iteration]; epsilon=out["refinement_energy_dimensionless_trace"][:,iteration]
        residual,correction,weak=dimensionless_weak_residual_correction(state,epsilon,b)
        _assert(torch.allclose(residual,out["refinement_residual_trace"][:,iteration],atol=2e-6,rtol=2e-5),"trace residual differs from the Ritz weak operator")
        phi,_,_,_,_=dimensionless_weak_hamiltonian_action(state,b); galerkin=torch.einsum("bkn,bjn->bkj",phi,weak)
        _assert(float(galerkin.abs().max())<3e-3,f"Ritz residual violates Galerkin orthogonality: {float(galerkin.abs().max())}")
        projected=project_weighted_outside(correction,state,b["w"]); overlap=torch.einsum("bkn,bn,bjn->bkj",projected,b["w"],state)
        _assert(float(overlap.abs().max())<2e-5 and torch.count_nonzero(projected.masked_select(b["boundary_mask"][:,None]))==0)
    sign=torch.where(torch.arange(CFG.k_states,device=DEVICE)%2==0,1.0,-1.0)[None,:,None]
    residual_a,correction_a,weak_a=dimensionless_weak_residual_correction(out["psi_initial"],out["energy_dimensionless_initial"],b)
    residual_b,correction_b,weak_b=dimensionless_weak_residual_correction(out["psi_initial"]*sign,out["energy_dimensionless_initial"],b)
    _assert(torch.allclose(residual_a,residual_b,atol=2e-6,rtol=2e-5) and torch.allclose(correction_a*sign,correction_b,atol=2e-6,rtol=2e-5) and torch.allclose(weak_a*sign,weak_b,atol=2e-6,rtol=2e-5))
run_test("Residual feedback matches the Ritz weak operator",test_refinement_residual_contract)


def test_refinement_trace_constraints():
    radial=next(r for r in records if r["geometry"]=="radial_reduced"); small=training_resampled_record(base_test,.82,1.15)
    b=to_device(collate_records([operator_only_record(radial),operator_only_record(small)],include_targets=False)); m=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad(): out=m.forward_operator(b,refinement_steps=2)
    identity=torch.eye(CFG.k_states,device=DEVICE)[None]
    for iteration in range(out["refinement_psi_trace"].shape[1]):
        state=out["refinement_psi_trace"][:,iteration]; gram=torch.einsum("bkn,bn,bjn->bkj",state,b["w"],state)
        _assert(torch.isfinite(state).all() and float((gram-identity).abs().max())<3e-3)
        _assert(bool(torch.all(torch.diff(out["refinement_energy_dimensionless_trace"][:,iteration],dim=1)>=-1e-6)))
        for index,n in enumerate(b["lengths"].tolist()):
            _assert(torch.count_nonzero(state[index,:,n:])==0 and torch.count_nonzero(state[index,:,0])==0 and torch.count_nonzero(state[index,:,n-1])==0)
    _assert(torch.equal(out["psi_hybrid_candidate"],out["refinement_psi_trace"][:,-1]) and
            torch.equal(out["energy_hybrid_candidate_dimensionless"],out["refinement_energy_dimensionless_trace"][:,-1]))
    block=torch.sqrt(out["refinement_residual_trace"].square().mean(-1)); _assert(bool(torch.all(block[:,1:]<=block[:,:-1]*(1+CFG.refinement_acceptance_tolerance)+2e-7)))
    _assert(torch.count_nonzero(out["refinement_step_size_trace"][~out["refinement_acceptance_trace"]])==0)
run_test("Every committed refinement iterate preserves physical constraints",test_refinement_trace_constraints)


def test_refinement_acceptance_safeguard():
    current_residual=torch.ones(4,2); proposal_residual=torch.tensor([[.5,.5],[2.,2.],[.5,.5],[float("nan"),.5]])
    current_epsilon=torch.tensor([[0.,1.]]).expand(4,-1).clone(); proposal_epsilon=current_epsilon.clone(); proposal_epsilon[2]+=2
    accepted=accept_refinement_step(current_residual,proposal_residual,current_epsilon,proposal_epsilon)
    _assert(accepted.tolist()==[True,False,False,False],f"unexpected recurrent safeguard decisions {accepted.tolist()}")
run_test("Monotonic refinement safeguard rejects unsafe proposals",test_refinement_acceptance_safeguard)


def test_refinement_finite_trace_and_attempt_mask():
    old=torch.ones(2,3,requires_grad=True); best_block=torch.full((2,),float("inf"))
    invalid=torch.tensor([[float("nan"),.2,.2],[float("inf"),.2,.2]],requires_grad=True)
    best,best_block=select_finite_refinement_proposal(old,best_block,invalid,torch.ones(2,dtype=torch.bool))
    _assert(torch.equal(best,old) and torch.isinf(best_block).all(),"non-finite proposal entered the diagnostic trace")
    finite=torch.full((2,3),.4,requires_grad=True); best,best_block=select_finite_refinement_proposal(best,best_block,finite,torch.ones(2,dtype=torch.bool))
    _assert(torch.allclose(best,finite) and torch.isfinite(best_block).all(),"finite backtrack did not replace a rejected non-finite proposal")
    proposals=torch.tensor([[[1.2,1.2],[float("nan"),float("nan")]],[[.5,.5],[1e20,1e20]]],requires_grad=True)
    previous=torch.ones_like(proposals); state_mask=torch.ones(2,2,dtype=torch.bool)
    attempted=torch.tensor([[True,False],[True,False]]); accepted=torch.tensor([[True,False],[False,False]])
    contraction,fraction=refinement_contraction_statistics(proposals,previous,state_mask,attempted,accepted)
    reference,_=refinement_contraction_statistics(torch.tensor([[[1.2,1.2],[0.,0.]],[[.5,.5],[0.,0.]]]),previous,state_mask,attempted,accepted)
    _assert(torch.isfinite(contraction) and torch.allclose(contraction,reference) and torch.allclose(fraction,torch.tensor(.5)),"attempt-masked recurrent statistics are incorrect")
    contraction.backward(); _assert(torch.isfinite(proposals.grad).all(),"masked non-finite proposal produced a non-finite gradient")
run_test("Non-finite proposals and unattempted slots cannot poison refinement losses",test_refinement_finite_trace_and_attempt_mask)


def test_refinement_end_to_end_gradients():
    b=to_device(collate_records([splits["train"][0]],include_targets=True)); m=TrueSchrodingerJEPA().to(DEVICE).train()
    out=m.forward_training(b,detach_ritz_vectors=True,refinement_steps=3); aligned,_=globally_align_torch(out["psi"],b["psi"],b["w"])
    loss=torch.sum(b["w"][:,None]*(aligned-b["psi"]).square())+out["refinement_proposal_residual_trace"].square().mean()
    loss.backward(); modules={"context":m.potential_context_encoder,"predictor":m.predictor,"decoder":m.wavefunction_decoder,"refiner":m.residual_correction}
    for name,module in modules.items():
        gradients=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        _assert(gradients and all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients)>0,f"recurrent gradient did not reach {name}")
    _assert(all(p.grad is None for p in m.solution_target_encoder.parameters()),"recurrent training leaked gradients into EMA target")
run_test("Three-step refinement has finite end-to-end gradients",test_refinement_end_to_end_gradients)

run_test("Bidirectional graph edges",lambda:_assert(set(map(tuple,sample_batch["edge_index"].T.tolist()))=={(b,a) for a,b in sample_batch["edge_index"].T.tolist()}))
run_test("No cross-batch edges",lambda:_assert(torch.equal(sample_batch["edge_index"][0]//sample_batch["x"].shape[1],sample_batch["edge_index"][1]//sample_batch["x"].shape[1])))


def permuted_graph_batch(batch:Mapping[str,Any],perm:torch.Tensor)->dict[str,Any]:
    out={key:(value.clone() if torch.is_tensor(value) else value) for key,value in batch.items()}; N=batch["x"].shape[1]; inv=torch.empty_like(perm); inv[perm]=torch.arange(N,device=perm.device)
    for key in ("features","x","w","V","V_relative","node_mask","potential_valid_mask","singularity_mask","boundary_mask"):
        out[key]=batch[key][:,perm]
    src,dst=batch["edge_index"]; out["edge_index"]=torch.stack([inv[src%N],inv[dst%N]])
    return out


def test_permutation():
    one=to_device(collate_records([splits["train"][0]],include_targets=True)); perm=torch.randperm(one["x"].shape[1],device=DEVICE); p=permuted_graph_batch(one,perm); model.potential_context_encoder.eval()
    with torch.no_grad(): a=model.potential_context_encoder(operator_batch_view(one)); b=model.potential_context_encoder(operator_batch_view(p))
    _assert(torch.allclose(b,a[:,perm],atol=2e-5,rtol=2e-5),float((b-a[:,perm]).abs().max()))
run_test("Graph permutation equivariance",test_permutation)


def resampled_operator(record:Mapping[str,Any],n:int,nonuniform:bool=False)->dict[str,Any]:
    out=operator_only_record(record); t=np.linspace(0,1,n); t=t**1.4 if nonuniform else t; x=record["domain_left"]+(record["domain_right"]-record["domain_left"])*t
    out["x"]=x; out["quadrature_weights"]=trapezoid_weights(x); out["V_raw"]=scipy.interpolate.PchipInterpolator(record["x"],record["V_raw"])(x)
    out["potential_valid_mask"]=np.ones(n,dtype=bool); out["singularity_mask"]=np.zeros(n,dtype=bool)
    if record["singularity_mask"][0]: out["potential_valid_mask"][0]=False; out["singularity_mask"][0]=True; out["V_raw"][0]=out["V_raw"][1]
    if record["singularity_mask"][-1]: out["potential_valid_mask"][-1]=False; out["singularity_mask"][-1]=True; out["V_raw"][-1]=out["V_raw"][-2]
    dx=np.empty(n); dx[0]=x[1]-x[0]; dx[-1]=x[-1]-x[-2]; dx[1:-1]=.5*(x[2:]-x[:-2]); out["grid_spacing"]=dx; return out


def test_padding():
    small=resampled_operator(base_test,49); mixed=to_device(collate_records([operator_only_record(base_test),small],include_targets=False)); model.eval()
    with torch.no_grad():
        out=model.forward_operator(mixed); context=out["context"]
        Hpsi,interior=training_hamiltonian_action(out["psi"],mixed["x"],mixed["V"],mixed["kappa"],mixed["node_mask"])
    _assert(torch.isfinite(context).all() and torch.isfinite(out["psi"]).all() and torch.isfinite(out["energy"]).all() and torch.isfinite(Hpsi).all())
    _assert(torch.count_nonzero(context[1,49:])==0 and torch.count_nonzero(out["psi"][1,:,49:])==0 and torch.count_nonzero(out["rho"][1,:,49:])==0)
    _assert(torch.count_nonzero(out["psi"][:,:,0])==0 and all(torch.count_nonzero(out["psi"][i,:,int(mixed["lengths"][i])-1])==0 for i in range(2)))
run_test("Padded-node exclusion",test_padding)
def test_multiple_resolutions():
    rows=[resampled_operator(base_test,n,nonuniform=(i%2==1)) for i,n in enumerate((41,49,57))]
    b=to_device(collate_records(rows,include_targets=False)); model.eval()
    with torch.no_grad(): out=model.forward_operator(b)
    _assert(torch.isfinite(out["psi"]).all() and out["psi"].shape==(3,CFG.k_states,57))
    _assert(all(torch.count_nonzero(out["psi"][i,:,n:])==0 for i,n in enumerate((41,49,57))))
run_test("Multiple grid resolutions",test_multiple_resolutions)


def test_refinement_batch_composition_invariance():
    short=resampled_operator(base_test,49); long=operator_only_record(base_test)
    # Reflection preserves the peer resolution/edge count while changing its sampled
    # operator.  Comparing equal-shaped batches isolates genuine cross-sample leakage
    # from the shape-dependent roundoff of float32 CUDA kernels.
    alternate_long=operator_only_record(reflected_record(base_test))
    _assert(len(long["x"])==len(alternate_long["x"]) and not np.allclose(long["V_raw"],alternate_long["V_raw"]),
            "composition control requires distinct equal-resolution peer operators")
    single=to_device(collate_records([short],include_targets=False))
    mixed=to_device(collate_records([short,long],include_targets=False))
    mixed_alternate=to_device(collate_records([short,alternate_long],include_targets=False))
    # A fixed, forked fixture seed makes this test independent of earlier tests and
    # repeatable when its notebook cell is rerun, without perturbing later RNG state.
    cuda_devices=[torch.cuda.current_device()] if DEVICE.type=="cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(104729); candidate=TrueSchrodingerJEPA().to(DEVICE).eval()
    with torch.no_grad():
        candidate.residual_correction.network[-1].weight.fill_(.002); candidate.residual_correction.network[-1].bias.fill_(.01)
        def probe(batch:Mapping[str,torch.Tensor])->dict[str,Any]:
            initial=candidate.forward_operator(batch,refinement_steps=0)
            residual,correction,weak=dimensionless_weak_residual_correction(initial["psi_initial"],initial["energy_dimensionless_initial"],batch)
            modulation=candidate.residual_correction(initial["context"],initial["state_tokens"],initial["psi_initial"],correction,weak,initial["energy_dimensionless_initial"],batch)
            final=candidate.forward_operator(batch,refinement_steps=2)
            return {"initial":initial,"residual":residual,"correction":correction,"weak":weak,
                    "modulation":modulation,"effective_correction":correction*modulation,"final":final}
        result_single=probe(single); result_mixed=probe(mixed); result_alternate=probe(mixed_alternate)

    # Same B/N shapes use tight tolerances and exact branch decisions.  A change here is
    # a real companion-sample/edge/pooling leak, not a padding-length numerical effect.
    _assert_tensor_close("peer composition changed context",result_mixed["initial"]["context"][:1,:,:49],
                         result_alternate["initial"]["context"][:1,:,:49],atol=2e-6,rtol=2e-5)
    _assert_tensor_close("peer composition changed state tokens",result_mixed["initial"]["state_tokens"][:1],
                         result_alternate["initial"]["state_tokens"][:1],atol=2e-6,rtol=2e-5)
    _assert_tensor_close("peer composition changed residual",result_mixed["residual"][:1],
                         result_alternate["residual"][:1],atol=2e-6,rtol=2e-5)
    _assert_tensor_close("peer composition changed learned modulation",result_mixed["modulation"][:1,:,:49],
                         result_alternate["modulation"][:1,:,:49],atol=3e-5,rtol=3e-5)
    _assert(torch.equal(result_mixed["final"]["refinement_acceptance_trace"][:1],
                        result_alternate["final"]["refinement_acceptance_trace"][:1]),
            "equal-shaped peer composition changed refinement decisions")
    _assert_tensor_close("peer composition changed proposal residuals",result_mixed["final"]["refinement_proposal_residual_trace"][:1],
                         result_alternate["final"]["refinement_proposal_residual_trace"][:1],atol=3e-5,rtol=3e-5)
    aligned_alternate,_=globally_align_torch(result_alternate["final"]["psi_hybrid_candidate"][:1,:,:49],
                                               result_mixed["final"]["psi_hybrid_candidate"][:1,:,:49],mixed["w"][:1,:49])
    _assert_tensor_close("peer composition changed refined wavefunctions",result_mixed["final"]["psi_hybrid_candidate"][:1,:,:49],
                         aligned_alternate,atol=4e-5,rtol=4e-5)
    _assert_tensor_close("peer composition changed refined energies",result_mixed["final"]["energy_hybrid_candidate_dimensionless"][:1],
                         result_alternate["final"]["energy_hybrid_candidate_dimensionless"][:1],atol=4e-5,rtol=4e-5)

    # B=1,N=49 and B=2,N=max_resolution exercise different GEMM/reduction/linalg
    # shapes.  PyTorch does not promise bitwise equality for those mathematically
    # equivalent float32 computations, so use an explicit CUDA shape-roundoff budget.
    shape_atol=2e-4 if DEVICE.type=="cuda" else 4e-5; shape_rtol=shape_atol
    final_atol=4e-4 if DEVICE.type=="cuda" else 8e-5; final_rtol=final_atol
    free=single["node_mask"]&single["potential_valid_mask"]&~single["boundary_mask"]
    free_states=free[:,None].expand_as(result_single["effective_correction"])
    mixed_effective=result_mixed["effective_correction"][:1,:,:49]
    mixed_initial_psi=result_mixed["initial"]["psi_initial"][:1,:,:49]
    single_initial_psi=result_single["initial"]["psi_initial"]
    overlap_matrix=torch.einsum("bkn,bn,bjn->bkj",mixed_initial_psi,single["w"],single_initial_psi)
    dominant_match=overlap_matrix.abs().argmax(-1)
    expected_match=torch.arange(CFG.k_states,device=DEVICE)[None]
    _assert(torch.equal(dominant_match,expected_match),
            "padding permuted or rotated the ordered Ritz states beyond sign ambiguity")
    initial_sign,_=global_sign_alignment_torch(mixed_initial_psi,single_initial_psi,single["w"])
    aligned_mixed_initial=mixed_initial_psi*initial_sign[...,None]
    aligned_mixed_correction=result_mixed["correction"][:1,:,:49]*initial_sign[...,None]
    aligned_mixed_effective=mixed_effective*initial_sign[...,None]
    _assert(torch.count_nonzero(result_single["effective_correction"].masked_select(~free_states))==0 and
            torch.count_nonzero(result_mixed["effective_correction"][0,:,49:])==0,
            "effective refinement correction escaped the free-node mask")
    _assert_tensor_close("padding changed initial Ritz wavefunctions beyond sign gauge",single_initial_psi,aligned_mixed_initial,
                         atol=final_atol,rtol=final_rtol)
    _assert_tensor_close("padding changed relative residual",result_single["residual"],result_mixed["residual"][:1],
                         atol=shape_atol,rtol=shape_rtol)
    _assert_tensor_close("padding changed analytic correction beyond sign gauge",
                         result_single["correction"].masked_select(free_states),
                         aligned_mixed_correction.masked_select(free_states),atol=shape_atol,rtol=shape_rtol)
    _assert_tensor_close("padding changed sign-invariant learned modulation",
                         result_single["modulation"].masked_select(free_states),
                         result_mixed["modulation"][:1,:,:49].masked_select(free_states),atol=shape_atol,rtol=shape_rtol)
    _assert_tensor_close("padding changed effective learned correction",
                         result_single["effective_correction"].masked_select(free_states),
                         aligned_mixed_effective.masked_select(free_states),atol=shape_atol,rtol=shape_rtol)
    # Compare the terminal residual rather than trace shape: evaluation may stop a
    # whole batch once every member is inactive, so an unrelated peer can legitimately
    # change the number of trailing no-op trace entries.
    _assert_tensor_close("padding changed terminal refinement residual",
                         result_single["final"]["refinement_residual_trace"][:,-1],
                         result_mixed["final"]["refinement_residual_trace"][:1,-1],atol=shape_atol,rtol=shape_rtol)
    aligned_mixed,_=globally_align_torch(result_mixed["final"]["psi_hybrid_candidate"][:1,:,:49],result_single["final"]["psi_hybrid_candidate"],single["w"])
    _assert_tensor_close("padding changed refined wavefunctions",result_single["final"]["psi_hybrid_candidate"],aligned_mixed,
                         atol=final_atol,rtol=final_rtol)
    _assert_tensor_close("padding changed refined energies",result_single["final"]["energy_hybrid_candidate_dimensionless"],
                         result_mixed["final"]["energy_hybrid_candidate_dimensionless"][:1],atol=final_atol,rtol=final_rtol)
run_test("Refinement is stable to mixed-resolution padding and isolated across peers",test_refinement_batch_composition_invariance)

radial_example=next(r for r in records if r["geometry"]=="radial_reduced")
run_test("Nonuniform grids",lambda:_assert(target_free_predict(model,operator_only_record(radial_example))["psi"].shape[-1]==len(radial_example["x"])))
def test_distant_sensitivity():
    base=operator_only_record(base_test); pert=copy.deepcopy(base); pert["V_raw"]=np.asarray(pert["V_raw"]).copy(); pert["V_raw"][2*len(pert["V_raw"])//3:]+=1.0
    a=to_device(collate_records([base],include_targets=False)); b=to_device(collate_records([pert],include_targets=False)); model.eval()
    with torch.no_grad(): ca=model.potential_context_encoder(a); cb=model.potential_context_encoder(b)
    _assert(float((ca[:,:len(base["x"])//3]-cb[:,:len(base["x"])//3]).abs().max())>1e-8)
run_test("Distant-potential perturbation sensitivity",test_distant_sensitivity)


def trusted_basis_test(basis:str):
    x=torch.tensor([-.5,0.,.5],requires_grad=True); values=polynomial_basis_values(x,5,basis)
    if basis=="chebyshev": expected=2*x*x-1
    elif basis=="legendre": expected=.5*(3*x*x-1)
    else: expected=(x*x-1)/math.sqrt(2)
    _assert(torch.allclose(values[:,1],expected,atol=1e-6)); values.sum().backward(); _assert(torch.isfinite(x.grad).all())
for label,basis in (("Chebyshev trusted recurrence values","chebyshev"),("Legendre trusted recurrence values","legendre"),("Normalized-Hermite trusted recurrence values","hermite")):
    run_test(label,lambda b=basis:trusted_basis_test(b))
def all_basis_gradients():
    for basis in KAN_BASIS_OPTIONS:
        layer=DenseBasisKANLinear(4,3,5,basis); x=torch.randn(2,4,requires_grad=True); layer(x).square().mean().backward(); _assert(torch.isfinite(x.grad).all())
run_test("Finite gradients for all bases",all_basis_gradients)
def dense_serialization():
    layer=DenseBasisKANLinear(4,3,3,"legendre"); x=torch.randn(2,4); expected=layer(x); buffer=io.BytesIO(); torch.save(layer.state_dict(),buffer); buffer.seek(0); clone=DenseBasisKANLinear(4,3,3,"legendre"); clone.load_state_dict(torch.load(buffer,weights_only=True)); _assert(expected.shape==(2,3) and torch.allclose(expected,clone(x)))
run_test("Dense KAN shapes and serialization",dense_serialization)
def factorized_serialization():
    layer=FactorizedBasisKANLinear(4,3,5,2,"hermite"); layer.eval(); x=torch.randn(2,4); expected=layer(x); buffer=io.BytesIO(); torch.save(layer.state_dict(),buffer); buffer.seek(0); clone=FactorizedBasisKANLinear(4,3,5,2,"hermite"); clone.load_state_dict(torch.load(buffer,weights_only=True)); clone.eval(); _assert(layer.rank==2 and expected.shape==(2,3) and torch.allclose(expected,clone(x)))
run_test("Factorized KAN shapes, rank, and serialization",factorized_serialization)
def orthonormalization_gradient():
    raw=torch.randn(2,CFG.k_states,32,requires_grad=True); w=torch.ones(2,32)/32
    orth=weighted_cholesky_orthonormalize_torch(raw,w); loss=(torch.einsum("bkn,bn,bjn->bkj",orth,w,orth)-torch.eye(CFG.k_states)).square().mean()+1e-3*orth.square().mean()
    loss.backward(); _assert(torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0)
run_test("Differentiable weighted orthonormalization",orthonormalization_gradient)
def near_carrier_orthonormalization_gradient():
    modes=torch.arange(1,CFG.k_states+1,dtype=torch.float32); t=torch.linspace(0,1,129)
    carrier=torch.sqrt(torch.tensor(2.0))*torch.sin(math.pi*modes[:,None]*t[None]); carrier[:,[0,-1]]=0
    raw=(carrier[None]+1e-5*torch.randn(2,CFG.k_states,len(t))).requires_grad_(True)
    w=torch.ones(2,len(t))/128; w[:,[0,-1]]*=.5
    orth=weighted_cholesky_orthonormalize_torch(raw,w); (orth*torch.randn_like(orth)).sum().backward()
    _assert(torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0,"near-identity Gram produced an invalid gradient")
run_test("Stable near-carrier orthonormalization gradient",near_carrier_orthonormalization_gradient)


"""
def enabled_loss_gradients():
    loss_records=[next(r for r in records if r["geometry"]=="truncated_line"),next(r for r in records if r["geometry"] in ("half_line","radial_reduced"))]
    b=to_device(collate_records(loss_records,include_targets=True)); m=TrueSchrodingerJEPA().to(DEVICE)
    with torch.no_grad(): m.solution_reconstruction_decoder.head.bias.fill_(.05)
    out=m.forward_training(b,detach_ritz_vectors=True); _,terms=compute_losses(out,b,DEFAULT_LOSS_WEIGHTS,loss_model=m)
    enabled=("jepa","subspace_projector","initial_subspace_projector","wave","infidelity","density",
             "energy_ground_exact","energy_log_gap_exact","energy_spectrum_exact","rayleigh_energy_exact","energy_rayleigh_consistency",
             "energy","energy_head","gap","h1","reconstruction","orthogonality","residual","rayleigh","refinement_deep_residual","refinement_contraction","latent_variance","latent_covariance","tail","kan",
             "topology_node_phase","topology_lobe_mass","topology_phase_cdf_supervision",
             "topology_log_amplitude_supervision","topology_phase_smoothness","topology_amplitude_smoothness")
    for name in enabled:
        m.zero_grad(set_to_none=True); terms[name].backward(retain_graph=True); grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        _assert(grads and all(torch.isfinite(g).all() for g in grads),name)
        _assert(sum(float(g.abs().sum()) for g in grads)>0,name)
run_test("Finite nonzero gradients for every enabled loss",enabled_loss_gradients)
def waveform_gradient_route():
    b=to_device(collate_records([next(r for r in records if r["family"]=="double_well")],include_targets=True)); m=TrueSchrodingerJEPA().to(DEVICE)
    out=m.forward_training(b); _,terms=compute_losses(out,b,{"wave":1.0},loss_model=m); terms["wave"].backward()
    modules={"context":m.potential_context_encoder,"predictor":m.predictor,"decoder":m.wavefunction_decoder}
    for name,module in modules.items():
        grads=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        _assert(grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0,f"waveform gradient did not reach {name}")
    if m.decoder_type=="topology_phase":
        for name,head in (("amplitude",m.wavefunction_decoder.amplitude_head),("phase-density",m.wavefunction_decoder.phase_density_head)):
            gradients=[p.grad for p in head.parameters() if p.grad is not None]
            _assert(gradients and all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients)>0,
                    f"waveform gradient did not reach {name} head")
    else:
        _assert(m.wavefunction_decoder.residual_gate.grad is not None and bool(torch.isfinite(m.wavefunction_decoder.residual_gate.grad)) and float(m.wavefunction_decoder.residual_gate.grad.abs())>0,"waveform gradient did not reach residual gate")
run_test("Waveform gradient reaches every deployable route",waveform_gradient_route)
"""
print(pd.DataFrame(TEST_RESULTS)[["index","test","status","seconds"]].to_string(index=False))

# %% [notebook cell 59]
def continuous_topology_field_oracle(record:Mapping[str,Any],return_fields:bool=False)->dict[str,Any]:
    """Target-only feasibility audit for the bounded amplitude/monotone-phase fields.

    This is deliberately outside ``forward_operator`` and is never an inference input.
    The phase is the positive piecewise-linear CDF through the exact target roots.  The
    required log amplitude is shifted so its maximum reaches the positive clip before
    clipping its negligible tails; the decoder's later quadrature gauge is only a global
    scale and cancels on normalization.  Centering on bare grid measure before clipping
    would let physically irrelevant Gaussian tails dominate the available dynamic range
    and would create a falsely pessimistic oracle.
    """
    x=np.asarray(record["x"],dtype=np.float64); w=np.asarray(record["quadrature_weights"],dtype=np.float64)
    target=np.asarray(record["psi"],dtype=np.float64); length=float(x[-1]-x[0]); t=(x-x[0])/length; w_t=w/length
    if len(x)<3 or np.any(np.diff(t)<=0) or target.shape!=(CFG.k_states,len(x)):
        raise ValueError("invalid tiny-overfit oracle record")
    fidelities=[]; clipped_masses=[]; density_ratios=[]; maximum_phase_increments=[]
    phase_targets=[]; log_amplitude_targets=[]; probability_targets=[]
    for state in range(CFG.k_states):
        roots=target_node_positions_from_wave(x,target[state],state)
        roots_t=(roots-x[0])/length
        knots=np.r_[0.0,roots_t,1.0]; quantiles=np.arange(state+2,dtype=np.float64)/(state+1)
        cdf=np.interp(t,knots,quantiles); theta=(state+1)*math.pi*cdf; carrier=np.sin(theta)
        phi=canonical_global_sign(target[state])*math.sqrt(length)
        if float(np.sum(w_t*phi*carrier))<0: phi=-phi
        usable=(np.arange(len(t))>0)&(np.arange(len(t))<len(t)-1)&~np.asarray(record["singularity_mask"],dtype=bool)&(np.abs(carrier)>=CFG.topology_target_carrier_floor)
        if int(usable.sum())<2: raise AssertionError(f"topology oracle state {state} has insufficient usable samples")
        sampled_log=np.log(np.maximum(np.abs(phi[usable]/carrier[usable]),1e-30))
        required_log=np.interp(t,t[usable],sampled_log)
        probability=w_t*phi**2; probability=probability/probability.sum()
        clip=CFG.amplitude_log_clip-CFG.topology_amplitude_target_clip_margin
        shifted=required_log-required_log.max()+clip
        clipped=np.clip(shifted,-clip,clip)
        clipped_masses.append(float(probability[shifted < -clip].sum()))
        # Match the production decoder's amplitude gauge and physical normalization.
        decoded_log=clipped-float(np.sum(w_t*clipped)/np.sum(w_t))
        phase_targets.append(cdf); log_amplitude_targets.append(decoded_log); probability_targets.append(probability)
        candidate=np.exp(decoded_log)*carrier; candidate[[0,-1]]=0.0
        candidate=candidate/math.sqrt(max(float(np.sum(w_t*candidate**2)),1e-30))
        overlap=float(np.sum(w_t*candidate*phi)); fidelities.append(overlap**2)
        phase_density=np.diff(cdf)/np.diff(t)
        if np.any(phase_density<=0): raise AssertionError(f"topology oracle phase is not strict for state {state}")
        density_ratios.append(float(phase_density.max()/phase_density.min()))
        maximum_phase_increments.append(float(np.diff(theta).max()))
    raw_min=CFG.phase_q_min+float(np.logaddexp(0.0,-CFG.phase_logit_clip))
    raw_max=CFG.phase_q_min+float(np.logaddexp(0.0, CFG.phase_logit_clip))
    result={"mean_fidelity":float(np.mean(fidelities)),"minimum_state_fidelity":float(np.min(fidelities)),
            "per_state_fidelity":[float(value) for value in fidelities],
            "maximum_clipped_probability_mass":float(np.max(clipped_masses)),
            "maximum_phase_density_ratio":float(np.max(density_ratios)),
            "available_phase_density_ratio":float(raw_max/raw_min),
            "maximum_phase_increment":float(np.max(maximum_phase_increments))}
    if return_fields:
        result.update({"phase_cdf_target":np.stack(phase_targets),
                       "log_amplitude_target":np.stack(log_amplitude_targets),
                       "target_probability_weights":np.stack(probability_targets)})
    return result


def tiny_batch_overfit()->dict[str,Any]:
    """Deterministic direct-route capacity audit with an explicitly reported assist.

    The first phase retains the historical deployable waveform/topology objective as a
    comparable control. If its high-fidelity gate has not converged within the fixed
    budget, the production target-field implementation verifies that the *same context/
    predictor/decoder route* can realize the fields. Inference never receives these
    labels, and the assisted diagnostic phase is exported rather than hidden.
    """
    policy=CFG.tiny_overfit_policy
    # The first accepted Sobol harmonic is a stable prefix fixture for a given data seed;
    # unlike selecting an extreme item from a mode-dependent split, its identity does not
    # change merely because per-family sample count increases.
    tiny_record=next(record for record in records if record["family"]=="harmonic")
    oracle_payload=continuous_topology_field_oracle(tiny_record,return_fields=True)
    phase_target_np=oracle_payload.pop("phase_cdf_target")
    log_amplitude_target_np=oracle_payload.pop("log_amplitude_target")
    oracle_payload.pop("target_probability_weights")
    batch=to_device(collate_records([tiny_record],include_targets=True))
    _assert(np.allclose(phase_target_np,batch["target_phase_cdf"][0].cpu().numpy(),atol=2e-6,rtol=2e-6) and
            np.allclose(log_amplitude_target_np,batch["target_log_amplitude"][0].cpu().numpy(),atol=3e-5,rtol=3e-5),
            "tiny oracle and production topology targets diverged")
    direct_weights={"wave":1.0,"infidelity":1.0,"orthogonality":.01,
                    "topology_node_phase":CFG.topology_node_loss_weight,
                    "topology_lobe_mass":CFG.topology_lobe_mass_weight,
                    "topology_phase_resolution":CFG.topology_phase_resolution_weight,
                    "topology_phase_smoothness":CFG.topology_phase_smoothness_weight,
                    "topology_amplitude_smoothness":CFG.topology_amplitude_smoothness_weight}
    # The capacity warm-start isolates the two representational fields.  Production
    # regularizers remain in ``direct_weights`` and the ordinary curriculum; mixing
    # them into this short diagnostic warm-start creates a different optimization test.
    field_weights={"wave":.2,"infidelity":.2,
                   "topology_node_phase":CFG.topology_node_loss_weight,
                   "topology_lobe_mass":CFG.topology_lobe_mass_weight,
                   "topology_phase_cdf_supervision":4.0*CFG.topology_phase_cdf_loss_weight,
                   "topology_log_amplitude_supervision":2.0*CFG.topology_log_amplitude_loss_weight}

    def objective(tiny:TrueSchrodingerJEPA,weights:Mapping[str,float],assist:bool
                  )->tuple[torch.Tensor,dict[str,torch.Tensor],dict[str,torch.Tensor]]:
        out=tiny.forward_training(batch,detach_ritz_vectors=True,refinement_steps=0)
        loss,terms=compute_losses(out,batch,weights,loss_model=tiny)
        phase_loss=terms["topology_phase_cdf_supervision"]
        amplitude_loss=terms["topology_log_amplitude_supervision"]
        return loss,terms,{"phase_profile":phase_loss,"amplitude_profile":amplitude_loss,**out}

    def evaluate(tiny:TrueSchrodingerJEPA,weights:Mapping[str,float],assist:bool)->tuple[dict[str,Any],dict[str,torch.Tensor]]:
        tiny.eval()
        with torch.no_grad():
            loss,terms,extra=objective(tiny,weights,assist)
            native_loss,_,_=objective(tiny,direct_weights,False)
        per_state=[float(terms[f"fidelity_s{state}"]) for state in range(CFG.k_states)]
        metrics={"loss":float(loss),"native_loss":float(native_loss),"fidelity":float(terms["fidelity"]),
                 "minimum_state_fidelity":float(min(per_state)),"per_state_fidelity":per_state,
                 "wave":float(terms["wave"]),"infidelity":float(terms["infidelity"]),
                 "node_phase":float(terms["topology_node_phase"]),"lobe_mass":float(terms["topology_lobe_mass"]),
                 "phase_profile":float(extra["phase_profile"]),"amplitude_profile":float(extra["amplitude_profile"])}
        return metrics,extra

    history=[]
    def record_history(step:int,phase:str,metrics:Mapping[str,Any],learning_rate:float)->None:
        history.append({"step":step,"phase":phase,"learning_rate":learning_rate,
                        **{key:value for key,value in metrics.items() if key!="per_state_fidelity"},
                        **{f"fidelity_s{state}":value for state,value in enumerate(metrics["per_state_fidelity"])}})

    fork_devices=([DEVICE.index if DEVICE.index is not None else torch.cuda.current_device()]
                  if DEVICE.type=="cuda" else [])
    with torch.random.fork_rng(devices=fork_devices,enabled=True):
        torch.manual_seed(TINY_OVERFIT_SEED)
        if DEVICE.type=="cuda": torch.cuda.manual_seed_all(TINY_OVERFIT_SEED)
        tiny=TrueSchrodingerJEPA().to(DEVICE)
        direct_modules=(tiny.potential_context_encoder,tiny.predictor,tiny.wavefunction_decoder)
        params=[parameter for module in direct_modules for parameter in module.parameters() if parameter.requires_grad]
        if len({id(parameter) for parameter in params})!=len(params): raise AssertionError("tiny direct-route parameters overlap")

        initial_model_state=cpu_snapshot(tiny.state_dict())
        initial_metrics,_=evaluate(tiny,direct_weights,False); initial_loss=initial_metrics["native_loss"]
        record_history(0,"native",initial_metrics,policy.learning_rate)
        best_state=cpu_snapshot(tiny.state_dict()); best_metrics=dict(initial_metrics); best_step=0; best_phase="initial"

        def passes_gate(metrics:Mapping[str,Any])->bool:
            return (float(metrics["fidelity"])>=policy.fidelity_min and
                    float(metrics["native_loss"])/max(initial_loss,1e-30)<=policy.loss_ratio_max)

        def consider(step:int,phase:str,metrics:Mapping[str,Any])->None:
            nonlocal best_state,best_metrics,best_step,best_phase
            # Any checkpoint satisfying the complete acceptance predicate outranks an
            # attractive-looking checkpoint that fails either fidelity or loss ratio.
            candidate=(passes_gate(metrics),float(metrics["fidelity"]),-float(metrics["native_loss"]))
            incumbent=(passes_gate(best_metrics),float(best_metrics["fidelity"]),-float(best_metrics["native_loss"]))
            if candidate>incumbent:
                best_state=cpu_snapshot(tiny.state_dict()); best_metrics=dict(metrics); best_step=step; best_phase=phase

        optimizer=torch.optim.AdamW(params,lr=policy.learning_rate,weight_decay=CFG.weight_decay)
        for step in range(1,policy.native_steps+1):
            tiny.train(); tiny.solution_target_encoder.eval(); optimizer.zero_grad(set_to_none=True)
            loss,_,_=objective(tiny,direct_weights,False)
            if not bool(torch.isfinite(loss).detach()): raise FloatingPointError(f"non-finite native tiny loss at step {step}")
            loss.backward(); nn.utils.clip_grad_norm_(params,CFG.gradient_clip_norm,error_if_nonfinite=True); optimizer.step()
            if step%policy.eval_interval==0 or step==policy.native_steps:
                metrics,_=evaluate(tiny,direct_weights,False); consider(step,"native",metrics)
                record_history(step,"native",metrics,float(optimizer.param_groups[0]["lr"]))
        native_metrics,_=evaluate(tiny,direct_weights,False)
        native_loss_ratio=native_metrics["native_loss"]/max(initial_loss,1e-30)

        field_supervision_used=not passes_gate(native_metrics)
        executed_steps=policy.native_steps; stop_reason="native_gate_satisfied" if not field_supervision_used else "field_assist_required"
        if field_supervision_used:
            # This is a separate capacity trajectory, not a continuation from a basin
            # selected by the production-style objective.  Keeping the same captured
            # initialization makes the comparison deterministic and prevents one phase
            # from silently making the other optimization problem harder.
            tiny.load_state_dict(initial_model_state)
            reset_metrics,_=evaluate(tiny,field_weights,True)
            record_history(policy.native_steps,"field_reset",reset_metrics,policy.learning_rate)
            optimizer=torch.optim.AdamW(params,lr=policy.learning_rate,weight_decay=CFG.weight_decay)
            for local_step in range(1,policy.field_steps+1):
                step=policy.native_steps+local_step; tiny.train(); tiny.solution_target_encoder.eval(); optimizer.zero_grad(set_to_none=True)
                loss,_,_=objective(tiny,field_weights,True)
                if not bool(torch.isfinite(loss).detach()): raise FloatingPointError(f"non-finite field-assisted tiny loss at step {step}")
                loss.backward(); nn.utils.clip_grad_norm_(params,CFG.gradient_clip_norm,error_if_nonfinite=True); optimizer.step()
                if local_step%policy.eval_interval==0 or local_step==policy.field_steps:
                    metrics,_=evaluate(tiny,field_weights,True); consider(step,"field_assist",metrics)
                    record_history(step,"field_assist",metrics,float(optimizer.param_groups[0]["lr"]))
            executed_steps=policy.native_steps+policy.field_steps

            optimizer=torch.optim.AdamW(params,lr=1e-3,weight_decay=CFG.weight_decay)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=4,
                                                                  threshold=1e-4,min_lr=policy.minimum_learning_rate)
            stable=0
            for step in range(executed_steps+1,policy.max_steps+1):
                tiny.train(); tiny.solution_target_encoder.eval(); optimizer.zero_grad(set_to_none=True)
                loss,_,_=objective(tiny,direct_weights,False)
                if not bool(torch.isfinite(loss).detach()): raise FloatingPointError(f"non-finite polish tiny loss at step {step}")
                loss.backward(); nn.utils.clip_grad_norm_(params,CFG.gradient_clip_norm,error_if_nonfinite=True); optimizer.step()
                if step%policy.eval_interval==0 or step==policy.max_steps:
                    metrics,_=evaluate(tiny,direct_weights,False); consider(step,"waveform_polish",metrics)
                    record_history(step,"waveform_polish",metrics,float(optimizer.param_groups[0]["lr"]))
                    scheduler.step(metrics["loss"])
                    passed=passes_gate(metrics)
                    stable=stable+1 if passed else 0
                    if stable>=policy.stable_evaluations:
                        executed_steps=step; stop_reason="stable_high_fidelity"; break
                executed_steps=step
            else: stop_reason="maximum_budget_reached"

        tiny.load_state_dict(best_state); final_metrics,final_extra=evaluate(tiny,direct_weights,False)
        final_out={key:value for key,value in final_extra.items() if key not in ("phase_profile","amplitude_profile")}

    pd.DataFrame(history).to_csv(OUT/"tiny_overfit_history.csv",index=False)
    final_topology=detached_topology_diagnostics(final_out["psi_direct"],operator_batch_view(batch))
    expected_nodes=torch.arange(CFG.k_states,device=DEVICE)[None]
    resolution_gate=final_out["maximum_phase_increment"]<=math.pi/CFG.minimum_intervals_per_lobe+1e-6
    persistent=final_topology["persistent_node_count"]==expected_nodes
    conditional=persistent[resolution_gate]
    result={"fixture_group_id":tiny_record["group_id"],
            "fixture_parameters":{key:float(value) for key,value in tiny_record["parameters"].items()},
            "initial_loss":initial_loss,"final_loss":final_metrics["native_loss"],
            "loss_ratio":final_metrics["native_loss"]/max(initial_loss,1e-30),
            "initial_fidelity":initial_metrics["fidelity"],"native_fidelity":native_metrics["fidelity"],
            "native_minimum_state_fidelity":native_metrics["minimum_state_fidelity"],
            "native_loss_ratio":native_loss_ratio,"final_fidelity":final_metrics["fidelity"],
            "best_fidelity":final_metrics["fidelity"],"minimum_state_fidelity":final_metrics["minimum_state_fidelity"],
            "per_state_fidelity":final_metrics["per_state_fidelity"],"best_step":best_step,"best_phase":best_phase,
            "steps_executed":executed_steps,"maximum_steps":policy.max_steps,"stop_reason":stop_reason,
            "field_supervision_used":field_supervision_used,"topology_field_oracle":oracle_payload,
            "continuous_topology_compliance":float(torch.equal(final_out["predicted_node_mask"].sum(-1),expected_nodes)),
            "phase_resolution_gate_fraction":float(resolution_gate.float().mean()),
            "conditional_persistent_topology_compliance":float(conditional.float().mean()) if conditional.numel() else float("nan")}
    result["parameters_finite"]=all(bool(torch.isfinite(parameter).all()) for parameter in tiny.parameters())
    print("tiny overfit candidate",result)
    return result


def validate_tiny_batch_overfit(result:Mapping[str,Any])->None:
    policy=CFG.tiny_overfit_policy; oracle=result["topology_field_oracle"]
    _assert(result["parameters_finite"],"tiny fit produced non-finite parameters")
    _assert(oracle["minimum_state_fidelity"]>=max(.95,policy.fidelity_min+.04),
            f"topology fields cannot support the requested fidelity: {oracle}")
    _assert(oracle["maximum_phase_density_ratio"]<oracle["available_phase_density_ratio"],
            f"topology oracle exceeds the configured phase-density dynamic range: {oracle}")
    _assert(result["native_fidelity"]>=policy.native_fidelity_min,
            f"native tiny optimization did not establish end-to-end learning: {result['native_fidelity']}")
    _assert(result["native_loss_ratio"]<=policy.loss_ratio_max,
            f"native tiny loss did not fall enough: ratio={result['native_loss_ratio']}")
    _assert(result["loss_ratio"]<=policy.loss_ratio_max,
            f"restored tiny loss did not fall enough: ratio={result['loss_ratio']}")
    _assert(result["final_fidelity"]>=policy.fidelity_min,
            f"tiny fidelity stayed at {result['final_fidelity']} after {result['steps_executed']} steps")
    _assert(result["continuous_topology_compliance"]==1.0,"tiny fit lost continuous Sturm topology")
    _assert(result["phase_resolution_gate_fraction"]>0,"tiny fit has no numerically resolved states")
    _assert(result["conditional_persistent_topology_compliance"]==1.0,
            "tiny fit lost sampled topology despite passing the phase-resolution gate")


tiny_result={}
def test_tiny_batch_overfit():
    global tiny_result
    tiny_result=tiny_batch_overfit(); validate_tiny_batch_overfit(tiny_result)
run_test("Tiny-batch overfit",test_tiny_batch_overfit)
print("tiny overfit",tiny_result)
