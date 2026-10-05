"""Losses, curriculum, and gates for the five-principle controlled ablation.

Common numerical/optimization helpers are loaded from the frozen V08 stage copy.
Its production self-tests and physics-informed loss are intentionally not run here.
"""

_validation_source=(Path(__file__).with_name("s04_validation.py")).read_text(encoding="utf-8")
_common_prefix=_validation_source.split("def run_test(name:",1)[0]
if len(_common_prefix)==len(_validation_source):
    raise RuntimeError("V08 validation helper boundary changed")
exec(compile(_common_prefix,str(Path(__file__).with_name("s04_validation.py")),"exec"),globals())
del _validation_source,_common_prefix

def _assert(condition:bool,message:str="assertion failed")->None:
    if not bool(condition):
        raise AssertionError(message)

REMOVED_LOSSES={
    "orthogonality","gram_barrier","residual","rayleigh","rayleigh_energy_exact",
    "energy_rayleigh_consistency","energy_log_gap_exact","gap",
    "refinement_deep_residual","refinement_contraction",
    "topology_node_phase","topology_lobe_mass","topology_phase_cdf_supervision",
    "topology_log_amplitude_supervision","topology_phase_smoothness",
    "topology_phase_resolution","topology_amplitude_smoothness",
}

# These terms are retained from V08. The numerical residual, Gram error, node count,
# and energy ordering are measured later, but none may influence gradients or selection.
DEFAULT_LOSS_WEIGHTS={name:weight for name,weight in DEFAULT_LOSS_WEIGHTS.items()
                      if name not in REMOVED_LOSSES}
_base_curriculum=build_curriculum_stage_definitions

def prefix_projector_loss(pred:torch.Tensor,target:torch.Tensor,w:torch.Tensor,
                          state_mask:torch.Tensor,active_states:int|None,
                          prefixes:Sequence[int]=(3,6,11))->tuple[torch.Tensor,dict[str,torch.Tensor]]:
    """Compare spans without requiring the predicted state vectors to be orthogonal."""
    count=CFG.k_states if active_states is None else int(active_states)
    losses=[]; diagnostics={}
    for prefix in prefixes:
        if prefix>count: continue
        valid=state_mask[:,:prefix].all(1)
        proposed=pred[:,:prefix]
        reference=target[:,:prefix]
        gram=torch.einsum("bkn,bn,bjn->bkj",proposed,w,proposed)
        overlap=torch.einsum("bkn,bn,bjn->bkj",proposed,w,reference)
        eye=torch.eye(prefix,device=pred.device,dtype=pred.dtype)[None]
        inverse_overlap=torch.linalg.solve(gram+1e-3*eye,overlap)
        coverage=(overlap*inverse_overlap).sum((-2,-1)).clamp(0,prefix)/prefix
        loss=((1-coverage)*valid).sum()/valid.sum().clamp_min(1)
        diagnostics[f"subspace_projector_p{prefix}"]=loss
        diagnostics[f"subspace_fidelity_p{prefix}"]=(coverage*valid).sum()/valid.sum().clamp_min(1)
        losses.append(loss)
    if not losses: return pred.sum()*0,diagnostics
    coefficients=([1.] if len(losses)==1 else [.4,.6] if len(losses)==2 else [.2,.2,.6])
    return (torch.stack(losses)*torch.as_tensor(coefficients,device=pred.device,dtype=pred.dtype)).sum(),diagnostics

def build_curriculum_stage_definitions()->list[dict[str,Any]]:
    stages=_base_curriculum()
    for stage in stages:
        stage["weights"]={name:weight for name,weight in stage["weights"].items()
                          if name not in REMOVED_LOSSES}
        stage["allow_ritz_gradient"]=False
    return stages

def ritz_gradient_validation_ready(validation:Mapping[str,float])->bool:
    del validation
    return False

def compute_losses(out:Mapping[str,torch.Tensor],batch:Mapping[str,torch.Tensor],
                   weights:Mapping[str,float],loss_model:nn.Module|None=None,
                   active_states:int|None=None,
                   physics_state_weights:torch.Tensor|Sequence[float]|None=None
                   )->tuple[torch.Tensor,dict[str,torch.Tensor]]:
    forbidden={name:weight for name,weight in weights.items()
               if name in REMOVED_LOSSES and float(weight)!=0}
    if forbidden:
        raise ValueError(f"removed physics losses cannot be activated: {forbidden}")
    pred,target,w,full_mask=out["psi"],batch["psi"],batch["w"],batch["state_mask"]
    mask=curriculum_state_mask(full_mask,active_states)
    active_count=CFG.k_states if active_states is None else int(active_states)
    aligned,fidelity=globally_align_torch(pred,target,w)
    wave_per=(w[:,None]*(aligned-target).square()).sum(-1)
    density_per=(w[:,None]*(out["rho"]-target.square()).abs()).sum(-1)
    derivative_error,derivative_target=dimensionless_derivative_errors(aligned,target,batch)
    h1_per=(wave_per+derivative_error)/(1+derivative_target).clamp_min(1e-8)
    scale=out["energy_scale_dimensionless"].detach().clamp_min(1e-8)
    exact=batch["energy_dimensionless"]
    energy_per=F.smooth_l1_loss(out["energy_dimensionless"]/scale,exact/scale,reduction="none")
    z_pred=out["z_pred"]; z_target=out["z_target"].detach()
    overlap=(z_pred*z_target).sum(-1)
    aligned_z=z_pred*torch.where(overlap.detach()>=0,1.0,-1.0)[...,None]
    jepa_per=F.smooth_l1_loss(F.layer_norm(aligned_z,aligned_z.shape[-1:]),
                              F.layer_norm(z_target,z_target.shape[-1:]),reduction="none").mean(-1)
    recon_aligned,_=globally_align_torch(out["reconstruction"],target,w)
    reconstruction_per=(w[:,None]*(recon_aligned-target).square()).sum(-1)
    projector,projector_diagnostics=prefix_projector_loss(pred,target,w,full_mask,active_count)
    norm_per=((w[:,None]*pred.square()).sum(-1)-1).abs()
    boundary_per=(pred.square()*batch["boundary_mask"][:,None]).sum(-1)/batch["boundary_mask"].sum(-1)[:,None].clamp_min(1)
    tail_per=(w[:,None]*pred.square()*physics_tail_mask(batch)[:,None]).sum(-1)
    latent_pred=latent_regularization(z_pred,mask)
    latent_online=latent_regularization(out["z_online"],mask)
    latent_variance=.5*(latent_pred[0]+latent_online[0])
    latent_covariance=.5*(latent_pred[1]+latent_online[1])
    reflection_wave=pred.sum()*0; reflection_energy=pred.sum()*0
    pairs=batch.get("reflection_pairs")
    if pairs is not None and len(pairs)>0:
        original=pairs[:,0]; reflected=pairs[:,1]
        reversed_wave=reverse_valid_nodes(pred[original],batch["lengths"][original])
        pair_weight=w[reflected][:,None]
        pair_overlap=(pair_weight*pred[reflected]*reversed_wave).sum(-1)
        sign=torch.where(pair_overlap.detach()>=0,1.0,-1.0)
        pair_mask=mask[original]&mask[reflected]
        reflection_wave=masked_state_mean((pair_weight*(pred[reflected]-sign[...,None]*reversed_wave).square()).sum(-1),pair_mask)
        reflection_energy=masked_state_mean(F.smooth_l1_loss(out["energy_dimensionless"][reflected],
            out["energy_dimensionless"][original],reduction="none"),pair_mask)
    if physics_state_weights is None:
        ramps=torch.ones(CFG.k_states,device=pred.device,dtype=pred.dtype)
    else:
        ramps=torch.as_tensor(physics_state_weights,device=pred.device,dtype=pred.dtype)
    terms={
        "jepa":masked_state_mean(jepa_per,mask),"subspace_projector":projector,
        "initial_subspace_projector":projector,"wave":masked_state_mean(wave_per,mask),
        "infidelity":masked_state_mean(1-fidelity,mask),
        "density":masked_state_mean(density_per,mask),
        "energy_ground_exact":masked_state_mean(energy_per[:,:1],mask[:,:1]),
        "energy_spectrum_exact":masked_state_mean(energy_per,mask),
        "energy":masked_state_mean(energy_per,mask),
        "energy_head":masked_state_mean(energy_per,mask),
        "h1":masked_state_mean(h1_per,mask),
        "reconstruction":masked_state_mean(reconstruction_per,mask),
        "latent_variance":latent_variance,"latent_covariance":latent_covariance,
        "reflection_wave":reflection_wave,"reflection_energy":reflection_energy,
        "normalization":ramped_state_mean(norm_per,mask,ramps),
        "boundary":ramped_state_mean(boundary_per,mask,ramps),
        "tail":ramped_state_mean(tail_per,mask,ramps),
        "kan":kan_coefficient_penalty(model if loss_model is None else loss_model),
        "fidelity":masked_state_mean(fidelity,mask),
        "fidelity_all":masked_state_mean(fidelity,full_mask),
        **projector_diagnostics,
    }
    # Diagnostic-only: these detached metrics must never enter the objective.
    with torch.no_grad():
        gram=torch.einsum("bkn,bn,bjn->bkj",pred.detach(),w,pred.detach())
        identity=torch.eye(CFG.k_states,device=pred.device)[None]
        valid_pair=mask[:,:,None]&mask[:,None,:]
        orth=((gram-identity).square()*valid_pair).sum()/valid_pair.sum().clamp_min(1)
        Hpsi,interior=training_hamiltonian_action(pred.detach().float(),batch["x"].float(),
            batch["V"].float(),batch["kappa"].float(),batch["node_mask"])
        core=pred.detach().float()[:,:,1:-1]
        residual=Hpsi-out["energy"].detach().float()[:,:,None]*core
        wi=w[:,1:-1].float()*interior
        residual_per=(wi[:,None]*residual.square()).sum(-1)/(
            (wi[:,None]*Hpsi.square()).sum(-1)+out["energy"].detach().float().square()+1e-8)
        terms["orthogonality"]=orth.detach()
        terms["residual"]=masked_state_mean(residual_per,mask).detach()
        terms["gap"]=masked_state_mean(F.smooth_l1_loss(torch.diff(out["energy_dimensionless"].detach(),dim=1),
            torch.diff(exact,dim=1),reduction="none"),mask[:,1:]&mask[:,:-1]).detach()
    for state in range(CFG.k_states):
        valid=full_mask[:,state]
        terms[f"fidelity_s{state}"]=(fidelity[:,state]*valid).sum()/valid.sum().clamp_min(1)
    objective=sum(float(weight)*terms[name] for name,weight in weights.items()
                  if float(weight)!=0 and name in terms)
    return objective,terms


def _physicless_contract_tests()->dict[str,bool]:
    test_batch=to_device(sample_batch)
    model.eval()
    with torch.no_grad():
        operator=operator_batch_view(test_batch)
        assert "psi" not in operator and "energy" not in operator
        result=model.forward_operator(operator)
        assert result["psi"].shape==test_batch["psi"].shape
        assert torch.isfinite(result["psi"]).all() and torch.isfinite(result["energy"]).all()
        assert torch.count_nonzero(result["psi"].masked_select(operator["boundary_mask"][:,None]))==0
        norm=(operator["w"][:,None]*result["psi"].square()).sum(-1)
        assert float((norm-1).abs().max())<2e-4
        probe=UnorderedStateEnergyHead(CFG.preset.hidden,CFG.backbone=="gnn_kan").to(DEVICE)
        probe.ground_out.weight.zero_(); probe.ground_out.bias.zero_()
        probe.excited_out.weight.zero_(); probe.excited_out.bias.zero_()
        energies,_=probe(result["state_tokens"],result["global_context"],operator)
        assert float(torch.diff(energies,dim=1).abs().max())<1e-5
        decoder=UnconstrainedWavefunctionDecoder(CFG.preset.hidden,CFG.backbone=="gnn_kan").to(DEVICE)
        decoder.head.weight.zero_(); decoder.head.bias.fill_(1.0)
        constant_sign=decoder(result["z_nodes"],operator)
        assert bool((constant_sign[:,:,1:-1]>=0).all())
    stages=build_curriculum_stage_definitions()
    assert len(stages)==6 and all(not (REMOVED_LOSSES&set(stage["weights"])) for stage in stages)
    assert not (REMOVED_LOSSES&set(DEFAULT_LOSS_WEIGHTS))
    assert not list(model.residual_correction.parameters())
    result_train=model.forward_training(test_batch,refinement_steps=0)
    loss,terms=compute_losses(result_train,test_batch,DEFAULT_LOSS_WEIGHTS,loss_model=model)
    missing={name for stage in stages for name,weight in stage["weights"].items()
             if float(weight)!=0 and name not in terms}
    assert not missing,f"curriculum weights have no implemented term: {missing}"
    assert torch.isfinite(loss) and not terms["residual"].requires_grad and not terms["orthogonality"].requires_grad
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.wavefunction_decoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.energy_head.parameters())
    model.zero_grad(set_to_none=True)
    return {"target_free_inference":True,"finite_forward_and_backward":True,
            "dirichlet_and_normalization":True,"no_sturm_or_order_hard_constraint":True,
            "removed_losses_absent":True,"diagnostics_detached":True}


PHYSICLESS_VALIDATION=_physicless_contract_tests()
(OUT/"physicless_validation.json").write_text(json.dumps(PHYSICLESS_VALIDATION,indent=2))
print("PHYSICLESS MODEL CONTRACT: PASS",PHYSICLESS_VALIDATION)
