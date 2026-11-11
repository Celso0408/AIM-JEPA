"""State-conditioned latent diagnostics at the 6-to-11 curriculum boundary.

This stage is strictly observational: it loads immutable training snapshots, evaluates
the fixed validation set, writes diagnostics, and restores the validation-best model.
"""

from __future__ import annotations

# %% [notebook cell V12 latent diagnostic]
LATENT_DIAGNOSTIC_CHECKPOINTS = (
    ("end_6", OUT / "latent_checkpoint_end_6.pt"),
    ("first_11", OUT / "latent_checkpoint_first_11.pt"),
    ("literal_last_11", OUT / "latent_checkpoint_literal_last_11.pt"),
    ("selected_end_11", OUT / "latent_checkpoint_selected_end_11.pt"),
)


def _latent_effective_rank(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return float("nan")
    centered = values.astype(np.float64) - values.astype(np.float64).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular**2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    probability = power / total
    probability = probability[probability > 0.0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _latent_checkpoint_observations(label: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = load_torch_checkpoint(path)
    if payload is None or "model_state" not in payload:
        raise RuntimeError(f"missing V12 latent diagnostic checkpoint: {path}")
    model.load_state_dict(payload["model_state"])
    model.eval()
    sample_rows: list[dict[str, Any]] = []
    latent_groups: dict[tuple[str, int], list[np.ndarray]] = {}
    raw_pred, raw_target, raw_family, raw_group = [], [], [], []
    with torch.inference_mode():
        for raw in loaders["val"]:
            batch = to_device(raw)
            output = model.forward_training(batch, detach_ritz_vectors=True)
            target = output["z_target"].detach().float()
            prediction = output["z_pred"].detach().float()
            sign = torch.where((prediction * target).sum(-1, keepdim=True) >= 0.0, 1.0, -1.0)
            aligned = prediction * sign
            relative = torch.linalg.vector_norm(aligned-target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-8)
            cosine = F.cosine_similarity(aligned, target, dim=-1)
            _, direct_fidelity = globally_align_torch(output["psi_direct"], batch["psi"], batch["w"])
            _, reconstruction_fidelity = globally_align_torch(output["reconstruction"], batch["psi"], batch["w"])
            valid = batch["state_mask"].detach().cpu().numpy().astype(bool)
            aligned_np = aligned.cpu().numpy(); target_np = target.cpu().numpy()
            relative_np = relative.cpu().numpy(); cosine_np = cosine.cpu().numpy()
            direct_np = direct_fidelity.detach().cpu().numpy(); reconstruction_np = reconstruction_fidelity.detach().cpu().numpy()
            records = raw["operator_records"]
            for batch_index, record in enumerate(records):
                family = str(record["family"]); group_id = str(record["group_id"])
                raw_pred.append(aligned_np[batch_index]); raw_target.append(target_np[batch_index])
                raw_family.append(family); raw_group.append(group_id)
                for state in range(CFG.k_states):
                    if not valid[batch_index, state]:
                        continue
                    latent_groups.setdefault(("__all__", state), []).append(aligned_np[batch_index, state])
                    latent_groups.setdefault((family, state), []).append(aligned_np[batch_index, state])
                    sample_rows.append({"checkpoint":label,"family":family,"group_id":group_id,"state":state,
                        "latent_relative_l2":float(relative_np[batch_index,state]),
                        "latent_cosine":float(cosine_np[batch_index,state]),
                        "predicted_latent_norm":float(np.linalg.norm(aligned_np[batch_index,state])),
                        "target_latent_norm":float(np.linalg.norm(target_np[batch_index,state])),
                        "direct_fidelity":float(direct_np[batch_index,state]),
                        "target_reconstruction_fidelity":float(reconstruction_np[batch_index,state])})
    np.savez_compressed(OUT/f"latent_raw_{label}.npz", z_pred=np.asarray(raw_pred,dtype=np.float32),
                        z_target=np.asarray(raw_target,dtype=np.float32), family=np.asarray(raw_family),
                        group_id=np.asarray(raw_group))
    samples = pd.DataFrame(sample_rows)
    summaries=[]
    for (family,state), vectors in latent_groups.items():
        matrix=np.asarray(vectors,dtype=np.float64)
        subset=samples[samples.state.eq(state)] if family=="__all__" else samples[samples.state.eq(state)&samples.family.eq(family)]
        summaries.append({"checkpoint":label,"family":family,"state":state,"samples":len(matrix),
            "predicted_latent_variance":float(matrix.var(axis=0,ddof=1).mean()) if len(matrix)>1 else float("nan"),
            "predicted_latent_effective_rank":_latent_effective_rank(matrix),
            **{column:float(subset[column].mean()) for column in ("latent_relative_l2","latent_cosine","direct_fidelity","target_reconstruction_fidelity")}})
    return samples,pd.DataFrame(summaries)


_latent_restore = cpu_snapshot(model.state_dict())
_latent_samples=[]; _latent_summaries=[]
try:
    for _label,_path in LATENT_DIAGNOSTIC_CHECKPOINTS:
        _samples,_summary=_latent_checkpoint_observations(_label,_path)
        _latent_samples.append(_samples); _latent_summaries.append(_summary)
finally:
    model.load_state_dict(_latent_restore); model.solution_target_encoder.eval()

latent_sample_state_metrics=pd.concat(_latent_samples,ignore_index=True)
latent_family_state_metrics=pd.concat(_latent_summaries,ignore_index=True)
latent_checkpoint_summary=(latent_family_state_metrics[latent_family_state_metrics.family.eq("__all__")]
    .groupby("checkpoint",sort=False).agg(
        latent_relative_l2=("latent_relative_l2","mean"), latent_cosine=("latent_cosine","mean"),
        predicted_latent_variance=("predicted_latent_variance","mean"),
        predicted_latent_effective_rank=("predicted_latent_effective_rank","mean"),
        direct_fidelity=("direct_fidelity","mean"),
        target_reconstruction_fidelity=("target_reconstruction_fidelity","mean")).reset_index())
latent_sample_state_metrics.to_csv(OUT/"latent_sample_state_metrics.csv",index=False)
latent_family_state_metrics.to_csv(OUT/"latent_family_state_metrics.csv",index=False)
latent_checkpoint_summary.to_csv(OUT/"latent_checkpoint_summary.csv",index=False)

fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for checkpoint,frame in latent_family_state_metrics[latent_family_state_metrics.family.eq("__all__")].groupby("checkpoint",sort=False):
    ordered=frame.sort_values("state")
    axes[0,0].plot(ordered.state,ordered.latent_relative_l2,marker="o",label=checkpoint)
    axes[0,1].plot(ordered.state,ordered.latent_cosine,marker="o",label=checkpoint)
    axes[1,0].plot(ordered.state,ordered.predicted_latent_effective_rank,marker="o",label=checkpoint)
    axes[1,1].plot(ordered.state,ordered.direct_fidelity,marker="o",label=checkpoint)
axes[0,0].set_ylabel("Relative latent L2")
axes[0,1].set_ylabel("Latent cosine")
axes[1,0].set_ylabel("Effective rank")
axes[1,1].set_ylabel("Direct fidelity")
for axis in axes.flat:
    axis.set_xlabel("State n"); axis.set_xticks(range(CFG.k_states)); axis.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle("V12 state-conditioned latent diagnostic: 6→11 boundary")
fig.tight_layout(); fig.savefig(OUT/"latent_state_transition_diagnostic.png",dpi=160); plt.close(fig)
print("V12 LATENT DIAGNOSTIC",latent_checkpoint_summary.round(5).to_dict("records"))

