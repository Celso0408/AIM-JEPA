"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 23]
def _potential_signature(x:np.ndarray,V:np.ndarray,affine:bool=False,reflect_canonical:bool=False)->bytes:
    t=(x-x[0])/(x[-1]-x[0]); grid=np.linspace(0,1,128); values=scipy.interpolate.PchipInterpolator(t,V)(grid)
    if affine:
        values=values-np.median(values); values=values/(np.quantile(np.abs(values),0.75)+1e-12)
    metadata=np.asarray([x[0],x[-1]],dtype=np.float64)
    rounded=np.round(np.concatenate([metadata,values]),7).astype("<f8").tobytes()
    if reflect_canonical:
        reverse=np.round(np.concatenate([metadata,values[::-1]]),7).astype("<f8").tobytes(); rounded=min(rounded,reverse)
    return rounded


def _hash_bytes(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def atomic_pickle_save(payload:Any,path:Path)->None:
    """Persist large caches safely despite transient OneDrive file locks."""
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload,handle,protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(12):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==11: raise
            time.sleep(0.25*(attempt+1))


def outer_tail_probability(spec:PotentialSpec,x:np.ndarray,w:np.ndarray,psi:np.ndarray)->np.ndarray:
    length=x[-1]-x[0]
    if spec.geometry=="truncated_line":
        shell=(x<=x[0]+0.05*length)|(x>=x[-1]-0.05*length)
    elif spec.geometry in ("half_line","radial_reduced"):
        shell=x>=x[-1]-0.05*length
    else:
        return np.zeros(np.atleast_2d(psi).shape[0],dtype=np.float64)
    return np.sum(w[shell][None,:]*psi[:,shell]**2,axis=1)


def effective_reference_nodes(spec:PotentialSpec)->int:
    base=CFG.preset.reference_nodes
    if spec.family=="eckart": return 4*base-3
    if spec.family=="rose_morse": return 2*base-1
    if spec.geometry in ("half_line","radial_reduced","singular_interval") or spec.family in ("morse","poschl_h","asymptotically_constant",*HELD_OUT_FAMILIES):
        return 2*base-1
    return base


def effective_model_nodes(spec:PotentialSpec)->int:
    base=CFG.preset.model_nodes
    # QUICK remains small, but difficult singular/narrow/long-tail operators receive an
    # adaptive second model resolution rather than being retained under-resolved.
    if spec.family=="eckart": return 4*base-3
    if spec.family=="rose_morse": return 2*base-1
    if spec.geometry in ("half_line","radial_reduced","singular_interval") or spec.family in ("morse","poschl_h","asymptotically_constant",*HELD_OUT_FAMILIES):
        return 2*base-1
    return base


def create_complete_record(spec:PotentialSpec)->dict[str,Any]:
    current_caps=MODE_CAPS[CFG.mode]
    nref=effective_reference_nodes(spec); reference=solve_reference_fem(spec,nref); n_model=effective_model_nodes(spec); projection_attempts=[]
    max_model_nodes=nref
    while True:
        projected=project_reference_solution(spec,reference,n_model); projection_attempts.append({"model_nodes":n_model,**projected["diagnostics"]})
        if projected["diagnostics"]["passes_projection_gate"] or n_model>=max_model_nodes: break
        n_model=min(max_model_nodes,2*n_model-1)
    energy=reference["energy"]; all_energy=reference["all_energy"]; psi=projected["psi"]; w=projected["w"]
    margin=np.full(CFG.k_states,np.inf) if spec.continuum_threshold is None else spec.continuum_threshold-energy
    family_floor=float(FAMILY_FLOORS.get(spec.family,{"energy_floor":current_caps["energy_rel"]/5})["energy_floor"])
    multiplier=1.25
    uncertainty=family_floor*np.maximum(1.0,np.abs(energy))
    valid=np.isfinite(energy)&(margin>multiplier*uncertainty)
    valid=np.logical_and.accumulate(valid)
    bound_count=CFG.k_states+CFG.extra_eigenpairs if spec.continuum_threshold is None else int(np.sum(all_energy < spec.continuum_threshold))
    reference_residual=independent_residual(reference["x"],reference["w"],reference["V"],reference["psi"],energy)
    tails=outer_tail_probability(spec,projected["x"],w,psi)
    group_blob=json.dumps({"family":spec.family,"parameters":spec.parameters,"base_id":spec.base_id},sort_keys=True,default=float)
    group_id=hashlib.sha256(group_blob.encode()).hexdigest()[:24]
    operator_metadata=json.dumps({"geometry":spec.geometry,"bc":spec.boundary_condition,"kappa":CFG.kinetic_coefficient,
                                  "singular_left":spec.singular_left,"singular_right":spec.singular_right},sort_keys=True).encode()
    fp=_hash_bytes(_potential_signature(projected["x"],projected["V"])+operator_metadata)
    sampled_fp=_hash_bytes(np.round(np.stack([projected["x"],projected["V"]]),10).astype("<f8").tobytes()+operator_metadata)
    afp=_hash_bytes(_potential_signature(projected["x"],projected["V"],affine=True)+operator_metadata)
    rfp=_hash_bytes(_potential_signature(projected["x"],projected["V"],affine=True,reflect_canonical=True)+operator_metadata)
    refdiag={"mass_gram_error":reference["mass_gram_error"],"algebraic_residual":reference["algebraic_residual"].tolist(),
             "independent_residual":reference_residual.tolist(),"all_energy":all_energy.tolist(),
             "all_node_count":reference["all_node_count"].tolist(),"requested_eigenpairs":len(all_energy),
             "bound_eigenpairs_among_requested":bound_count,"solver_seconds":reference["solver_seconds"],
             "normalization_error":float(np.max(np.abs(np.sum(reference["w"]*reference["psi"]**2,axis=1)-1))),
             "gram_error":float(np.max(np.abs(weighted_gram(reference["psi"],reference["w"])-np.eye(CFG.k_states)))),
             "primary_algebraic_residual_max":float(np.max(reference["algebraic_residual"][:CFG.k_states])),
             "auxiliary_algebraic_residual_max":float(np.max(reference["algebraic_residual"][CFG.k_states:])) if len(reference["algebraic_residual"])>CFG.k_states else np.nan}
    modeldiag=dict(projected["diagnostics"]); modeldiag["projection_attempts"]=projection_attempts; modeldiag["selected_model_nodes"]=n_model
    rejection=[]
    if not valid.all(): rejection.append("incomplete_bound_spectrum")
    if bound_count<CFG.k_states: rejection.append("fewer_than_11_bound_pairs")
    if not np.array_equal(reference["all_node_count"][:CFG.k_states],np.arange(CFG.k_states)): rejection.append("reference_node_theorem")
    if refdiag["gram_error"]>2e-8: rejection.append("reference_orthogonality")
    if refdiag["primary_algebraic_residual_max"]>1e-6: rejection.append("reference_algebraic_residual")
    if float(np.max(reference_residual))>current_caps["residual"]: rejection.append("reference_independent_residual")
    if not modeldiag["passes_projection_gate"]: rejection.append("model_projection")
    if not np.array_equal(modeldiag["node_count"],np.arange(CFG.k_states)): rejection.append("model_node_theorem")
    tail_cap=5e-2
    if spec.geometry in ("truncated_line","half_line","radial_reduced") and float(tails.max())>tail_cap: rejection.append("outer_tail")
    if rejection:
        raise ValueError(";".join(rejection))
    x=projected["x"]
    local_dx=np.empty_like(x); local_dx[0]=x[1]-x[0]; local_dx[-1]=x[-1]-x[-2]; local_dx[1:-1]=0.5*(x[2:]-x[:-2])
    record={"x":x,"quadrature_weights":w,"V_raw":projected["V"],"potential_valid_mask":projected["potential_valid_mask"],
            "psi":psi,"rho":psi**2,"energy":energy,"valid_state_mask":valid,"family":spec.family,
            "parameters":spec.parameters,"geometry":spec.geometry,"boundary_condition":spec.boundary_condition,
            "domain_left":spec.domain_left,"domain_right":spec.domain_right,"grid_spacing":local_dx,
            "kinetic_coefficient":CFG.kinetic_coefficient,"continuum_threshold":spec.continuum_threshold,
            "bound_margin":margin,"solver_residual":np.asarray(modeldiag["independent_residual_per_state"]),
            "tail_probability":tails,"node_count":np.asarray(modeldiag["node_count"]),"group_id":group_id,
            "augmentation_parent":group_id,"potential_fingerprint":fp,"affine_fingerprint":afp,
            "reflection_fingerprint":rfp,"sampled_array_fingerprint":sampled_fp,"reference_grid_size":nref,"solver_version":SOLVER_VERSION,
            "reference_diagnostics":refdiag,"model_grid_diagnostics":modeldiag,
            "singularity_mask":projected["singularity_mask"],"generation_component":spec.generation_component}
    validate_record_schema(record)
    return record


def generate_complete_state_core()->tuple[list[dict[str,Any]],pd.DataFrame]:
    partial_path=CACHE_DIR/f"psi_true_jepa_dataset_{numerical_cache_key}.partial.pkl"
    accepted=[]; rejected=[]
    if REUSE_DEBUG_NUMERICAL_CACHE and partial_path.exists():
        try:
            with partial_path.open("rb") as handle: partial=pickle.load(handle)
            if partial.get("cache_key")==numerical_cache_key:
                accepted=list(partial.get("records",[])); rejected=list(partial.get("rejected",[]))
                print("resuming partial dataset cache",partial_path,{"accepted":len(accepted)})
        except (EOFError,pickle.UnpicklingError,OSError,AttributeError):
            accepted=[]; rejected=[]
    quick_core_families=(
        "infinite_box","harmonic","half_harmonic","morse","poschl_h","poschl_t",
        "radial_coulomb","radial_oscillator","kratzer","anharmonic_polynomial",
        "asymmetric_well","double_well","multiwell","single_barrier","multiple_barriers",
        "gaussian_well","gaussian_barrier","localized_defect","harmonic_compact",
        "fourier_random","chebyshev_random","gaussian_random_field","reflected_potential",
        "smooth_perturbed","composite_confining","procedural_composite",
        "asymptotically_constant",
    )
    generation_families=quick_core_families if CFG.mode=="QUICK" else TRAIN_FAMILIES
    if CFG.mode=="QUICK":
        legacy_path=PROJECT_DIR/"cache"/"psi_true_jepa_dataset_6bde2337166d871c0b0e.pkl"
        if legacy_path.exists():
            try:
                with legacy_path.open("rb") as handle: legacy=pickle.load(handle)
                by_group={record["group_id"]:record for record in accepted}
                for record in legacy.get("records",[]):
                    if record.get("family") in quick_core_families:
                        by_group.setdefault(record["group_id"],record)
                accepted=list(by_group.values())
                print("seeded QUICK dataset from compatible legacy cache",legacy_path,{"accepted":len(accepted)})
            except (EOFError,pickle.UnpicklingError,OSError,AttributeError,KeyError):
                pass
    for fi,family in enumerate(generation_families):
        accepted_before_family=len(accepted)
        needed=CFG.preset.per_family; exponent=int(math.ceil(math.log2(max(2,needed*CFG.max_generation_attempts))))
        if family=="rose_morse":
            # Rose-Morse needs a larger Sobol budget in QUICK to find enough valid
            # low-tail, well-projected examples.
            exponent=max(exponent,7)
        points=qmc.Sobol(16,scramble=True,seed=CFG.seed+101*fi).random_base2(exponent)
        for j,u in enumerate(points):
            if sum(r["family"]==family for r in accepted)>=needed: break
            base_id=f"sobol-{fi:02d}-{j:05d}"
            try:
                accepted.append(create_complete_record(make_spec(family,u,base_id)))
            except Exception as exc:
                rejected.append({"family":family,"base_id":base_id,"reason":str(exc)})
        count=sum(r["family"]==family for r in accepted)
        if count<needed:
            reasons=[row for row in rejected if row["family"]==family]
            raise RuntimeError(f"complete_state_core could accept only {count}/{needed} records for {family}: {reasons}")
        if REUSE_DEBUG_NUMERICAL_CACHE and len(accepted)>accepted_before_family:
            atomic_pickle_save({"cache_key":numerical_cache_key,"records":accepted,"rejected":rejected},partial_path)
    return accepted,pd.DataFrame(rejected,columns=["family","base_id","reason"])


DATASET_CACHE_VERSION="complete-state-core-v8-restored-27-families"
REUSE_DEBUG_NUMERICAL_CACHE=os.environ.get("PSI_JEPA_REUSE_NUMERICAL_CACHE","1")=="1"
cache_payload={"version":DATASET_CACHE_VERSION,"solver":SOLVER_VERSION,"seed":CFG.seed,
    "reference_nodes":CFG.preset.reference_nodes,"model_nodes":CFG.preset.model_nodes,
    "per_family":CFG.preset.per_family,"k":CFG.k_states,"extra":CFG.extra_eigenpairs,
    "families":TRAIN_FAMILIES,"held_out_families":HELD_OUT_FAMILIES,
    "mode":CFG.mode,"max_generation_attempts":CFG.max_generation_attempts,
    "source_sha256":SOURCE_SHA256}
numerical_cache_key=hashlib.sha256(json.dumps(cache_payload,sort_keys=True).encode()).hexdigest()[:20]
legacy_cache_payload={**cache_payload,"version":"complete-state-core-v3"}
legacy_cache_key=hashlib.sha256(json.dumps(legacy_cache_payload,sort_keys=True).encode()).hexdigest()[:20]
CACHE_DIR=Path(os.environ.get("PSI_JEPA_CACHE_DIR",PROJECT_DIR/"cache")).resolve(); CACHE_DIR.mkdir(parents=True,exist_ok=True)
dataset_cache_path=CACHE_DIR/f"psi_true_jepa_dataset_{numerical_cache_key}.pkl"
if REUSE_DEBUG_NUMERICAL_CACHE and dataset_cache_path.exists():
    try:
        with dataset_cache_path.open("rb") as handle: cached_dataset=pickle.load(handle)
        if cached_dataset.get("cache_key")!=numerical_cache_key: raise RuntimeError("incompatible dataset cache")
        records,rejection_ledger=cached_dataset["records"],cached_dataset["rejection_ledger"]
        print("loaded explicit debug dataset cache",dataset_cache_path)
    except (EOFError, pickle.UnpicklingError, OSError, RuntimeError, KeyError, AttributeError) as exc:
        print(f"ignoring unreadable dataset cache {dataset_cache_path}: {exc}")
        records,rejection_ledger=generate_complete_state_core()
        atomic_pickle_save({"cache_key":numerical_cache_key,"records":records,"rejection_ledger":rejection_ledger},dataset_cache_path)
else:
    records,rejection_ledger=generate_complete_state_core()
    if REUSE_DEBUG_NUMERICAL_CACHE:
        atomic_pickle_save({"cache_key":numerical_cache_key,"records":records,"rejection_ledger":rejection_ledger},dataset_cache_path)
rejection_ledger.to_csv(OUT/"rejections.csv",index=False)
assert all(np.asarray(r["valid_state_mask"]).all() for r in records)
print("complete_state_core",{"accepted":len(records),"rejected_attempts":len(rejection_ledger),
                             "states_per_hamiltonian":CFG.k_states})

# %% [notebook cell 25]
def deterministic_split(items:Sequence[dict[str,Any]])->dict[str,list[dict[str,Any]]]:
    result={"train":[],"val":[],"test":[]}
    for family_index,family in enumerate(TRAIN_FAMILIES):
        ordered=sorted([r for r in items if r["family"]==family],key=lambda r:hashlib.sha256((r["group_id"]+f"|{CFG.seed}").encode()).hexdigest())
        n=len(ordered); ntrain=max(1,int(round(.70*n))); remaining=ordered[ntrain:]
        if len(remaining)==1:
            destination="val" if family_index%2==0 else "test"; result[destination].extend(remaining)
        else:
            nval=len(remaining)//2+(family_index%2 and len(remaining)%2); result["val"].extend(remaining[:nval]); result["test"].extend(remaining[nval:])
        result["train"].extend(ordered[:ntrain])
    if not result["val"] or not result["test"]: raise RuntimeError("split did not produce validation and test sets")
    return result


splits=deterministic_split(records)
split_sets={name:{r["group_id"] for r in rows} for name,rows in splits.items()}
for a,b in (("train","val"),("train","test"),("val","test")):
    assert not split_sets[a]&split_sets[b]
for key in ("potential_fingerprint","affine_fingerprint","reflection_fingerprint"):
    sets={name:{r[key] for r in rows} for name,rows in splits.items()}
    for a,b in (("train","val"),("train","test"),("val","test")):
        assert not sets[a]&sets[b],f"{key} leakage"


NEAR_DUPLICATE_SHAPE_SIZE=128
NEAR_DUPLICATE_DISTANCE_MIN=1e-4


def normalized_signature(record:Mapping[str,Any])->np.ndarray:
    length=float(record["domain_right"]-record["domain_left"]); kappa=float(record["kinetic_coefficient"])
    if not (length>0 and kappa>0): raise ValueError("near-duplicate signature requires positive domain length and kinetic coefficient")
    t=(record["x"]-record["x"][0])/length
    values=scipy.interpolate.PchipInterpolator(t,record["V_raw"])(np.linspace(0,1,NEAR_DUPLICATE_SHAPE_SIZE))
    centered=values-np.median(values); norm=float(np.linalg.norm(centered))
    shape=np.zeros_like(centered) if norm<=1e-14 else centered/norm
    centered_rms=norm/math.sqrt(len(centered))
    # After x = left + length*t, the centered dimensionless operator contains
    # length**2*V/kappa; kappa/length**2 restores its absolute energy scale.
    metadata=np.asarray([math.log1p(centered_rms*length**2/kappa),math.log(kappa/length**2)])
    return np.concatenate([shape,metadata])


def reflected_normalized_signature(signature:np.ndarray)->np.ndarray:
    signature=np.asarray(signature,dtype=np.float64)
    if len(signature)!=NEAR_DUPLICATE_SHAPE_SIZE+2: raise ValueError("unexpected near-duplicate signature size")
    return np.concatenate([signature[:NEAR_DUPLICATE_SHAPE_SIZE][::-1],signature[NEAR_DUPLICATE_SHAPE_SIZE:]])


signature_by_group={r["group_id"]:normalized_signature(r) for r in records}
near_duplicate_min=float("inf"); near_duplicate_closest:dict[str,Any]|None=None
for a,b in (("train","val"),("train","test"),("val","test")):
    rows_a,rows_b=splits[a],splits[b]
    signatures_a=np.stack([signature_by_group[r["group_id"]] for r in rows_a])
    signatures_b=np.stack([signature_by_group[r["group_id"]] for r in rows_b])
    for orientation,candidates in (("direct",signatures_b),("reflected",np.stack([reflected_normalized_signature(v) for v in signatures_b]))):
        distances,indices=cKDTree(candidates).query(signatures_a,k=1)
        index_a=int(np.argmin(distances)); index_b=int(indices[index_a]); distance=float(distances[index_a])
        if distance<near_duplicate_min:
            near_duplicate_min=distance
            near_duplicate_closest={"distance":distance,"orientation":orientation,
                "split_a":a,"group_id_a":rows_a[index_a]["group_id"],"family_a":rows_a[index_a]["family"],
                "split_b":b,"group_id_b":rows_b[index_b]["group_id"],"family_b":rows_b[index_b]["family"]}
near_duplicate_audit={"signature":"centered-shape_dimensionless-strength_kinetic-scale-v1",
                      "threshold":NEAR_DUPLICATE_DISTANCE_MIN,"minimum_cross_split_distance":near_duplicate_min,
                      "closest_pair":near_duplicate_closest}
(OUT/"near_duplicate_audit.json").write_text(json.dumps(near_duplicate_audit,indent=2))
if not near_duplicate_min>NEAR_DUPLICATE_DISTANCE_MIN:
    raise RuntimeError("near-duplicate leakage audit failed: "+json.dumps(near_duplicate_audit,sort_keys=True))

def robust_potential_gauge(record:Mapping[str,Any])->float:
    """Quadrature-weighted median, excluding protected singular endpoint samples."""
    values=np.asarray(record["V_raw"],dtype=np.float64); weights=np.asarray(record["quadrature_weights"],dtype=np.float64)
    valid=np.asarray(record["potential_valid_mask"],dtype=bool); order=np.argsort(values[valid],kind="stable")
    sorted_values=values[valid][order]; sorted_weights=weights[valid][order]; cumulative=np.cumsum(sorted_weights)
    return float(sorted_values[min(int(np.searchsorted(cumulative,.5*cumulative[-1],side="left")),len(sorted_values)-1)])


def dimensionless_operator_arrays(record:Mapping[str,Any])->tuple[float,float,np.ndarray]:
    length=float(record["domain_right"]-record["domain_left"]); kappa=float(record["kinetic_coefficient"])
    if not (length>0 and kappa>0): raise ValueError("dimensionless operator requires positive L and kappa")
    gauge=robust_potential_gauge(record); energy_unit=kappa/(length*length)
    return gauge,energy_unit,(np.asarray(record["V_raw"],dtype=np.float64)-gauge)/energy_unit


def weighted_quantile(values:np.ndarray,weights:np.ndarray,quantile:float)->float:
    """Deterministic left-continuous weighted empirical quantile."""
    values=np.asarray(values,dtype=np.float64).reshape(-1); weights=np.asarray(weights,dtype=np.float64).reshape(-1)
    if len(values)!=len(weights) or len(values)==0: raise ValueError("weighted quantile requires equally sized nonempty arrays")
    if not 0<=quantile<=1 or not (np.isfinite(values).all() and np.isfinite(weights).all()): raise ValueError("invalid weighted quantile input")
    if np.any(weights<0) or not float(weights.sum())>0: raise ValueError("weighted quantile requires positive total nonnegative weight")
    order=np.argsort(values,kind="stable"); ordered_values=values[order]; ordered_weights=weights[order]
    threshold=quantile*float(ordered_weights.sum()); index=min(int(np.searchsorted(np.cumsum(ordered_weights),threshold,side="left")),len(values)-1)
    return float(ordered_values[index])


def fit_dimensionless_u_scale(rows:Sequence[Mapping[str,Any]],quantile:float=.75)->tuple[float,dict[str,Any]]:
    """Fit a train-only scale with equal family/Hamiltonian mass and quadrature node mass."""
    families=sorted({str(record["family"]) for record in rows})
    if not families: raise ValueError("cannot fit dimensionless scale without records")
    values=[]; masses=[]; family_rows=[]; hamiltonian_masses={}
    family_target_mass=1.0/len(families)
    for family in families:
        selected=[record for record in rows if record["family"]==family]
        family_values=[]; family_weights=[]
        for record in selected:
            valid=np.asarray(record["potential_valid_mask"],dtype=bool); quadrature=np.asarray(record["quadrature_weights"],dtype=np.float64)[valid]
            if not float(quadrature.sum())>0: raise ValueError(f"zero valid quadrature mass for {record['group_id']}")
            _,_,dimensionless=dimensionless_operator_arrays(record); node_values=np.abs(dimensionless[valid])
            node_masses=family_target_mass*quadrature/(len(selected)*quadrature.sum())
            values.append(node_values); masses.append(node_masses); family_values.append(node_values); family_weights.append(node_masses)
            hamiltonian_masses[str(record["group_id"])]=float(node_masses.sum())
        combined_values=np.concatenate(family_values); combined_weights=np.concatenate(family_weights)
        family_rows.append({"family":family,"hamiltonians":len(selected),"assigned_mass":float(combined_weights.sum()),
                            "weighted_abs_U_quantile":weighted_quantile(combined_values,combined_weights,quantile)})
    all_values=np.concatenate(values); all_masses=np.concatenate(masses); scale=max(weighted_quantile(all_values,all_masses,quantile),1e-8)
    audit={"estimator":"hierarchical_equal-family_equal-hamiltonian_quadrature-weighted-quantile-v1","fit_split":"train",
           "quantile":quantile,"absolute_dimensionless_U":True,"families":len(families),"hamiltonians":len(rows),
           "total_assigned_mass":float(all_masses.sum()),"family_statistics":family_rows,"hamiltonian_assigned_masses":hamiltonian_masses}
    return float(scale),audit


train_values=np.concatenate([r["V_raw"][r["potential_valid_mask"]] for r in splits["train"]])
TRAIN_V_MEDIAN=float(np.median(train_values)); TRAIN_V_SCALE=float(np.quantile(np.abs(train_values-TRAIN_V_MEDIAN),0.75)+1e-8)
TRAIN_U_SCALE,NORMALIZATION_AUDIT=fit_dimensionless_u_scale(splits["train"])
NORMALIZATION_CONVENTION="t=(x-left)/L; U=(V-weighted_median(V))*L^2/kappa; epsilon=(E-gauge)*L^2/kappa"
NORMALIZATION_CONTRACT={"dimensionless_U_scale":TRAIN_U_SCALE,"estimator":NORMALIZATION_AUDIT["estimator"],
                        "fit_split":NORMALIZATION_AUDIT["fit_split"],"quantile":NORMALIZATION_AUDIT["quantile"],
                        "hierarchy":"equal_family_equal_hamiltonian_quadrature_node_mass","convention":NORMALIZATION_CONVENTION}
NORMALIZATION_HASH=hashlib.sha256(json.dumps(NORMALIZATION_CONTRACT,sort_keys=True).encode()).hexdigest()
NORMALIZATION_METADATA={**NORMALIZATION_CONTRACT,"normalization_hash":NORMALIZATION_HASH,
                        "dimensionless_U_scale_audit":NORMALIZATION_AUDIT,
                        "legacy_diagnostics":{"V_median":TRAIN_V_MEDIAN,"V_scale":TRAIN_V_SCALE}}
split_manifest={name:[{"group_id":r["group_id"],"family":r["family"],"parent":r["augmentation_parent"]} for r in rows] for name,rows in splits.items()}
def record_label_checksum(record:Mapping[str,Any])->str:
    payload=b"".join(np.asarray(record[key],dtype="<f8").tobytes() for key in ("x","quadrature_weights","energy","psi"))
    return hashlib.sha256(payload).hexdigest()


dataset_manifest=[{"group_id":r["group_id"],"fingerprint":r["potential_fingerprint"],"sampled_fingerprint":r["sampled_array_fingerprint"],
                   "label_checksum":record_label_checksum(r),"family":r["family"],"reference_grid_size":r["reference_grid_size"],
                   "model_grid_size":len(r["x"]),"solver_version":r["solver_version"]} for r in records]
DATASET_MANIFEST_HASH=hashlib.sha256(json.dumps(dataset_manifest,sort_keys=True).encode()).hexdigest()
SPLIT_MANIFEST_HASH=hashlib.sha256(json.dumps(split_manifest,sort_keys=True).encode()).hexdigest()
(OUT/"dataset_manifest.json").write_text(json.dumps(dataset_manifest,indent=2))
(OUT/"split_manifest.json").write_text(json.dumps(split_manifest,indent=2))
(OUT/"normalization.json").write_text(json.dumps(NORMALIZATION_METADATA,indent=2))
print({"split_sizes":{k:len(v) for k,v in splits.items()},"near_duplicate_min":near_duplicate_min,
       "near_duplicate_closest":near_duplicate_closest,
       "dataset_manifest_hash":DATASET_MANIFEST_HASH[:16],"split_manifest_hash":SPLIT_MANIFEST_HASH[:16],
       "train_only_V_scale":TRAIN_V_SCALE,"train_only_dimensionless_U_scale":TRAIN_U_SCALE,
       "normalization_estimator":NORMALIZATION_AUDIT["estimator"],"normalization_hash":NORMALIZATION_HASH[:16]})


def refresh_record_fingerprints(record:dict[str,Any],variant:str)->dict[str,Any]:
    metadata=json.dumps({"geometry":record["geometry"],"bc":record["boundary_condition"],"kappa":record["kinetic_coefficient"],
                         "singular_left":bool(record["singularity_mask"][0]),"singular_right":bool(record["singularity_mask"][-1])},sort_keys=True).encode()
    record["potential_fingerprint"]=_hash_bytes(_potential_signature(record["x"],record["V_raw"])+metadata)
    record["affine_fingerprint"]=_hash_bytes(_potential_signature(record["x"],record["V_raw"],affine=True)+metadata)
    record["reflection_fingerprint"]=_hash_bytes(_potential_signature(record["x"],record["V_raw"],affine=True,reflect_canonical=True)+metadata)
    record["sampled_array_fingerprint"]=_hash_bytes(np.round(np.stack([record["x"],record["V_raw"]]),10).astype("<f8").tobytes()+variant.encode())
    record["augmentation_kind"]=variant; return record


def shifted_record(record:Mapping[str,Any],shift:float)->dict[str,Any]:
    out=copy.deepcopy(dict(record)); out["V_raw"]=np.asarray(out["V_raw"])+shift; out["energy"]=np.asarray(out["energy"])+shift
    if out["continuum_threshold"] is not None: out["continuum_threshold"]+=shift
    out["bound_margin"]=np.asarray(out["bound_margin"]); out["augmentation_parent"]=record["group_id"]
    out["augmentation_parameters"]={"constant_shift":float(shift)}
    return refresh_record_fingerprints(out,"constant_shift")


def reflected_record(record:Mapping[str,Any])->dict[str,Any]:
    out=copy.deepcopy(dict(record)); left=float(record["domain_left"]); right=float(record["domain_right"])
    out["x"]=left+right-np.asarray(record["x"])[::-1]; out["quadrature_weights"]=np.asarray(record["quadrature_weights"])[::-1].copy()
    for key in ("V_raw","potential_valid_mask","singularity_mask","grid_spacing"):
        out[key]=np.asarray(record[key])[::-1].copy()
    out["psi"]=np.stack([canonical_global_sign(q[::-1].copy()) for q in np.asarray(record["psi"])]); out["rho"]=out["psi"]**2
    out["augmentation_parent"]=record["group_id"]; out["augmentation_parameters"]={"reflection_center":.5*(left+right)}
    validate_record_schema(out); return refresh_record_fingerprints(out,"spatial_reflection")


def translated_record(record:Mapping[str,Any],translation:float)->dict[str,Any]:
    if record["geometry"] not in ("finite_interval","truncated_line"):
        return copy.deepcopy(dict(record))
    out=copy.deepcopy(dict(record)); out["x"]=np.asarray(out["x"])+translation
    out["domain_left"]=float(out["domain_left"]+translation); out["domain_right"]=float(out["domain_right"]+translation)
    out["augmentation_parent"]=record["group_id"]; out["augmentation_parameters"]={"translation":float(translation)}
    validate_record_schema(out); return refresh_record_fingerprints(out,"spatial_translation")


def dilated_record(record:Mapping[str,Any],scale:float)->dict[str,Any]:
    if not scale>0: raise ValueError("dilation scale must be positive")
    out=copy.deepcopy(dict(record)); center=.5*(float(record["domain_left"])+float(record["domain_right"]))
    out["x"]=center+scale*(np.asarray(record["x"])-center); out["domain_left"]=center+scale*(float(record["domain_left"])-center); out["domain_right"]=center+scale*(float(record["domain_right"])-center)
    out["quadrature_weights"]=scale*np.asarray(record["quadrature_weights"]); out["grid_spacing"]=scale*np.asarray(record["grid_spacing"])
    out["V_raw"]=np.asarray(record["V_raw"])/(scale*scale); out["energy"]=np.asarray(record["energy"])/(scale*scale)
    out["psi"]=np.asarray(record["psi"])/math.sqrt(scale); out["rho"]=out["psi"]**2
    if out["continuum_threshold"] is not None: out["continuum_threshold"]=float(out["continuum_threshold"]/(scale*scale))
    out["bound_margin"]=np.asarray(record["bound_margin"])/(scale*scale); out["augmentation_parent"]=record["group_id"]
    out["augmentation_parameters"]={"coordinate_dilation":float(scale)}
    validate_record_schema(out); return refresh_record_fingerprints(out,"coordinate_dilation")


def project_record_to_grid(record:Mapping[str,Any],x_new:np.ndarray,variant:str)->dict[str,Any]:
    """Quadrature L2-project an untouched labeled Hamiltonian onto a new audit grid."""
    out=copy.deepcopy(dict(record)); x_new=np.asarray(x_new,dtype=np.float64)
    psi,diag=l2_project_states(np.asarray(record["x"]),np.asarray(record["psi"]),x_new)
    w=trapezoid_weights(x_new); interpolator=scipy.interpolate.PchipInterpolator(record["x"],record["V_raw"])
    V=np.asarray(interpolator(x_new),dtype=np.float64)
    potential_valid=np.interp(x_new,record["x"],np.asarray(record["potential_valid_mask"],dtype=float))>.5
    singular=np.interp(x_new,record["x"],np.asarray(record["singularity_mask"],dtype=float))>.5
    if bool(record["singularity_mask"][0]): potential_valid[0]=False; singular[0]=True; V[0]=V[1]
    if bool(record["singularity_mask"][-1]): potential_valid[-1]=False; singular[-1]=True; V[-1]=V[-2]
    dx=np.empty_like(x_new); dx[0]=x_new[1]-x_new[0]; dx[-1]=x_new[-1]-x_new[-2]; dx[1:-1]=.5*(x_new[2:]-x_new[:-2])
    residual=independent_residual(x_new,w,V,psi,np.asarray(record["energy"]))
    spec=PotentialSpec(record["family"],record["parameters"],record["geometry"],record["boundary_condition"],
                       record["domain_left"],record["domain_right"],record["continuum_threshold"],
                       bool(record["singularity_mask"][0]),bool(record["singularity_mask"][-1]),"ood",str(record["group_id"]))
    node_count=np.asarray([persistent_nodes(row) for row in psi]); gram_error=float(np.max(np.abs(weighted_gram(psi,w)-np.eye(CFG.k_states))))
    if not np.array_equal(node_count,np.arange(CFG.k_states)): raise AssertionError(f"{variant} projection changed node counts")
    if not (diag["boundary_max"]<1e-12 and gram_error<2e-6): raise AssertionError(f"{variant} projection failed boundary/Gram audit")
    residual_cap=max(MODE_CAPS[CFG.mode]["residual"],3*float(np.max(record["solver_residual"])))
    if float(np.max(residual))>=residual_cap: raise AssertionError(f"{variant} independent residual {float(np.max(residual)):.3e} exceeds {residual_cap:.3e}")
    out.update({"x":x_new,"quadrature_weights":w,"V_raw":V,"potential_valid_mask":potential_valid,"singularity_mask":singular,
                "psi":psi,"rho":psi**2,"grid_spacing":dx,"solver_residual":residual,"tail_probability":geometry_tail_probability(spec,x_new,w,psi),
                "node_count":node_count,"augmentation_parent":record["group_id"],
                "sampled_array_fingerprint":_hash_bytes(np.round(np.stack([x_new,V]),10).astype("<f8").tobytes()+variant.encode()),
                "model_grid_diagnostics":{**dict(record["model_grid_diagnostics"]),**diag,"ood_grid_variant":variant,
                                          "independent_residual_per_state":residual.tolist(),"passes_projection_gate":True}})
    validate_record_schema(out); return refresh_record_fingerprints(out,variant)


def equivalence_cleaned_lofo_groups(family:str)->dict[str,Any]:
    held_out=[r for r in records if r["family"]==family]
    equivalent={(r["affine_fingerprint"],r["reflection_fingerprint"]) for r in held_out}
    fit=[r for r in records if r["family"]!=family and (r["affine_fingerprint"],r["reflection_fingerprint"]) not in equivalent]
    return {"family":family,"fit_group_ids":[r["group_id"] for r in fit],"held_out_group_ids":[r["group_id"] for r in held_out],
            "equivalence_cleaned":True,"status":"PENDING_MATCHED_RETRAINING"}


# OOD suites are created after split/statistics freeze and never enter early stopping.
ood_suites:dict[str,list[dict[str,Any]]]={}
for category,families in {"single_parameter_extrapolation":("harmonic","morse"),
                          "joint_parameter_extrapolation":("double_well","fourier_random")}.items():
    category_records=[]
    for f in families:
        errors=[]
        for amount in (1.08,1.05,1.02):
            u=np.full(16,.5); u[0]=amount
            if category.startswith("joint"): u[1]=amount
            try:
                category_records.append(create_complete_record(make_spec(f,u,f"ood-{category}-{f}-{amount}"))); break
            except Exception as exc: errors.append(f"{amount}:{exc}")
        else: raise RuntimeError(f"could not validate OOD {category}/{f}: {errors}")
    ood_suites[category]=category_records
def generate_held_out_family(family:str,count:int,seed_offset:int)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    exponent=int(math.ceil(math.log2(max(2,count*max(4,CFG.max_generation_attempts)))))
    points=qmc.Sobol(16,scramble=True,seed=CFG.seed+seed_offset).random_base2(exponent); accepted=[]; rejected=[]
    for index,u in enumerate(points):
        if len(accepted)>=count: break
        base_id=f"ood-sobol-{family}-{index:05d}"
        try: accepted.append(create_complete_record(make_spec(family,u,base_id)))
        except Exception as exc: rejected.append({"family":family,"base_id":base_id,"reason":str(exc)})
    if len(accepted)<count:
        raise RuntimeError(f"could accept only {len(accepted)}/{count} held-out records for {family}: {rejected[-8:]}")
    if len({r["group_id"] for r in accepted})!=count: raise AssertionError(f"duplicate held-out groups for {family}")
    return accepted,rejected


unseen_records=[]; reparameterized_records=[]; held_out_rejections=[]
for family_index,f in enumerate(UNSEEN_ANALYTICAL_FAMILIES):
    family_rows,failures=generate_held_out_family(f,CFG.unseen_per_family,7001+101*family_index)
    unseen_records.extend(family_rows); held_out_rejections.extend(failures)
for family_index,f in enumerate(REPARAMETERIZED_SEEN_FAMILIES):
    family_rows,failures=generate_held_out_family(f,max(1,min(CFG.unseen_per_family,16)),8101+101*family_index)
    reparameterized_records.extend(family_rows); held_out_rejections.extend(failures)
pd.DataFrame(held_out_rejections,columns=["family","base_id","reason"]).to_csv(OUT/"held_out_rejections.csv",index=False)
training_fingerprints={kind:{r[kind] for r in records} for kind in ("potential_fingerprint","affine_fingerprint","reflection_fingerprint")}
for kind,known in training_fingerprints.items():
    overlap={r[kind] for r in unseen_records}&known
    if overlap: raise AssertionError(f"genuine unseen {kind} overlaps training: {sorted(overlap)[:3]}")
ood_suites["unseen_analytical_functional_forms"]=unseen_records
ood_suites["seen_basis_reparameterization"] = reparameterized_records
reflection_candidates=[r for r in splits["test"] if r["geometry"] in ("finite_interval","truncated_line")]
base_test=max(reflection_candidates,key=lambda r:float(np.linalg.norm(r["V_raw"]-r["V_raw"][::-1])/(np.linalg.norm(r["V_raw"])+1e-30)))
ood_suites["constant_potential_shift"]=[shifted_record(base_test,2.75)]
ood_suites["valid_spatial_reflection"]=[reflected_record(base_test)]

# Cheap, strictly post-split QUICK challenge suites. Some are untouched test slices of
# families present in training and are labeled as such; they are not misrepresented as
# leave-family-out evidence.
harmonic_train=[r for r in splits["train"] if r["family"]=="harmonic"]
interp_omega=float(np.mean([r["parameters"]["omega"] for r in harmonic_train])); interp_x0=float(np.mean([r["parameters"]["x0"] for r in harmonic_train])); interp_radius=8.5/math.sqrt(interp_omega)
if not (min(r["parameters"]["omega"] for r in harmonic_train)<=interp_omega<=max(r["parameters"]["omega"] for r in harmonic_train) and min(r["parameters"]["x0"] for r in harmonic_train)<=interp_x0<=max(r["parameters"]["x0"] for r in harmonic_train)): raise AssertionError("interpolation challenge is not bracketed by training parameters")
interpolation_spec=PotentialSpec("harmonic",{"omega":interp_omega,"x0":interp_x0},"truncated_line","dirichlet",interp_x0-interp_radius,interp_x0+interp_radius,None,False,False,"ood","ood-harmonic-bracketed-midpoint")
ood_suites["within_range_unseen_parameter_interpolation"]=[create_complete_record(interpolation_spec)]
thresholded_test=[r for r in splits["test"] if r["continuum_threshold"] is not None]
near_continuum=min(thresholded_test,key=lambda r:float(np.min(r["bound_margin"])))
ood_suites["near_continuum_states"]=[near_continuum]
resolution_base=next(r for r in splits["test"] if r["family"]=="harmonic")
t_uniform=np.linspace(0,1,max(161,len(resolution_base["x"])+32)); x_uniform=resolution_base["domain_left"]+(resolution_base["domain_right"]-resolution_base["domain_left"])*t_uniform
t_nonuniform=np.linspace(0,1,max(145,len(resolution_base["x"])+16)); mapped=.5*(1+np.sinh(.7*(2*t_nonuniform-1))/np.sinh(.7)); x_nonuniform=resolution_base["domain_left"]+(resolution_base["domain_right"]-resolution_base["domain_left"])*mapped
ood_suites["variable_uniform_resolution"]=[project_record_to_grid(resolution_base,x_uniform,"uniform_resolution")]
ood_suites["nonuniform_resolution"]=[project_record_to_grid(resolution_base,x_nonuniform,"nonuniform_resolution")]
domain_spec=PotentialSpec(resolution_base["family"],resolution_base["parameters"],resolution_base["geometry"],resolution_base["boundary_condition"],
                          resolution_base["domain_left"],resolution_base["domain_right"],resolution_base["continuum_threshold"],
                          False,False,"ood",f"ood-domain-{resolution_base['group_id']}")
changed_domain_record=create_complete_record(enlarge_spec(domain_spec,1.15)); changed_domain_record["augmentation_parent"]=resolution_base["group_id"]; changed_domain_record["group_id"]=resolution_base["group_id"]
ood_suites["changed_domain_same_physical_operator"]=[changed_domain_record]
ood_suites["controlled_smooth_perturbations"]=[create_complete_record(make_spec("smooth_perturbed",np.full(16,.83),"ood-controlled-smooth-083"))]

def _test_slice(*families:str)->list[dict[str,Any]]:
    rows=[r for r in splits["test"] if r["family"] in families]
    if not rows: raise AssertionError(f"no untouched test Hamiltonian for {families}")
    return rows


ood_suites["random_smooth_potentials"]=_test_slice("fourier_random","gaussian_random_field")
ood_suites["anharmonic_wells"]=_test_slice("anharmonic_polynomial")
ood_suites["asymmetric_wells"]=[create_complete_record(make_spec("asymmetric_well",np.full(16,.83),"ood-asymmetric-083"))]
ood_suites["double_wells"]=_test_slice("double_well")
ood_suites["multiwell_potentials"]=[create_complete_record(make_spec("multiwell",np.full(16,.83),"ood-multiwell-083"))]
ood_suites["barriers"]=_test_slice("single_barrier")
ood_suites["multiple_barriers"]=[create_complete_record(make_spec("multiple_barriers",np.full(16,.83),"ood-multiple-barriers-083"))]
ood_suites["localized_defects"]=_test_slice("localized_defect")
LOFO_MANIFEST={family:equivalence_cleaned_lofo_groups(family) for family in TRAIN_FAMILIES}
(OUT/"equivalence_cleaned_lofo_manifest.json").write_text(json.dumps(LOFO_MANIFEST,indent=2))

OOD_EXPERIMENT_STATUS={
    "within_range_unseen_parameter_interpolation":{"status":"EXECUTED","category":"within_range_unseen_parameter_interpolation","note":"new harmonic parameter midpoint, explicitly bracketed by training omega and center values"},
    "single_parameter_extrapolation":{"status":"EXECUTED","category":"single_parameter_extrapolation"},
    "joint_parameter_extrapolation":{"status":"EXECUTED","category":"joint_parameter_extrapolation"},
    "near_continuum_states":{"status":"EXECUTED_DIAGNOSTIC","category":"near_continuum_states","minimum_margin":float(np.min(near_continuum["bound_margin"])),"note":"one untouched thresholded test Hamiltonian; production sweep pending"},
    "equivalence_cleaned_leave_one_family_out":{"status":"PENDING_MATCHED_RETRAINING","manifest":"equivalence_cleaned_lofo_manifest.json"},
    "unseen_analytical_functional_forms":{"status":"EXECUTED","category":"unseen_analytical_functional_forms","hamiltonians_per_family":CFG.unseen_per_family,
                                            "note":"genuinely distinct Scarf-II, Eckart, and cosh-well generators; never used for selection"},
    "seen_basis_reparameterization":{"status":"EXECUTED_DIAGNOSTIC","category":"seen_basis_reparameterization",
                                      "note":"Rosen-Morse II is the trained sech^2+tanh basis under a new parameterization; excluded from genuine unseen pooling"},
    "random_smooth_potentials":{"status":"EXECUTED_TEST_SLICE","category":"random_smooth_potentials","note":"untouched same-family ID-test slice, not leave-family-out evidence"},
    "anharmonic_wells":{"status":"EXECUTED_TEST_SLICE","category":"anharmonic_wells","note":"untouched same-family ID-test slice"},
    "asymmetric_wells":{"status":"EXECUTED_SAME_FAMILY_CHALLENGE","category":"asymmetric_wells","note":"new same-generator parameter point; not functional-form OOD"},
    "double_wells":{"status":"EXECUTED_TEST_SLICE","category":"double_wells","note":"untouched same-family ID-test slice"},
    "multiwell_potentials":{"status":"EXECUTED_SAME_FAMILY_CHALLENGE","category":"multiwell_potentials","note":"new same-generator multiwell parameter point; not unseen-form OOD"},
    "barriers":{"status":"EXECUTED_TEST_SLICE","category":"barriers","note":"untouched same-family ID-test slice"},
    "multiple_barriers":{"status":"EXECUTED_SAME_FAMILY_CHALLENGE","category":"multiple_barriers","note":"new same-generator multiple-barrier parameter point; not unseen-form OOD"},
    "localized_defects":{"status":"EXECUTED_TEST_SLICE","category":"localized_defects","note":"untouched same-family ID-test slice"},
    "variable_uniform_resolution":{"status":"EXECUTED","category":"variable_uniform_resolution"},
    "nonuniform_resolution":{"status":"EXECUTED","category":"nonuniform_resolution"},
    "changed_domain_same_physical_operator":{"status":"EXECUTED","category":"changed_domain_same_physical_operator","note":"same full-line harmonic potential with enlarged artificial truncation"},
    "controlled_smooth_perturbations":{"status":"EXECUTED_SAME_FAMILY_CHALLENGE","category":"controlled_smooth_perturbations","note":"new smooth-perturbed-family parameter point; not functional-form OOD"},
    "constant_potential_shift":{"status":"EXECUTED","category":"constant_potential_shift"},
    "valid_spatial_reflection":{"status":"EXECUTED","category":"valid_spatial_reflection"},
}

# %% [notebook cell 27]
data_audit=pd.DataFrame([{"group_id":r["group_id"],"family":r["family"],"component":r["generation_component"],
                          "binding_margin_min":float(np.min(r["bound_margin"])),"tail_max":float(np.max(r["tail_probability"])),
                          "label_residual_max":float(np.max(r["solver_residual"])),"model_gram_error":r["model_grid_diagnostics"]["gram_max_error"],
                          "projection_fidelity_min":r["model_grid_diagnostics"]["projection_fidelity_min"],
                          "solver_seconds":r["reference_diagnostics"]["solver_seconds"]} for r in records])
data_audit.to_csv(OUT/"data_audit.csv",index=False)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,record in zip(axes,[records[0],records[len(records)//2],records[-1]]):
    x=record["x"]; scale=np.ptp(record["V_raw"])+1e-12
    ax.plot(x,record["V_raw"],color="black",label="V")
    for state in (0,5,10): ax.plot(x,record["psi"][state]*0.15*scale+record["energy"][state],label=f"psi {state}")
    ax.set_title(record["family"]); ax.grid(alpha=.2)
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/"representative_reference_data.png",dpi=140); plt.close(fig)
print(data_audit.groupby("component")[["tail_max","label_residual_max","projection_fidelity_min","solver_seconds"]].agg(["mean","max"]).round(5))

# %% [notebook cell 29]
def operator_only_record(record:Mapping[str,Any],include_continuum_threshold:bool=False)->dict[str,Any]:
    allowed=("x","quadrature_weights","V_raw","potential_valid_mask","geometry","boundary_condition",
             "domain_left","domain_right","grid_spacing","kinetic_coefficient","singularity_mask")
    out={key:copy.deepcopy(record[key]) for key in allowed if key in record}
    if include_continuum_threshold:
        out["continuum_threshold"]=copy.deepcopy(record.get("continuum_threshold"))
    return out


def build_graph_edges(x:torch.Tensor,node_mask:torch.Tensor,potential_valid:torch.Tensor,
                      dilations:Sequence[int]=CFG.graph_dilations)->tuple[torch.Tensor,torch.Tensor]:
    B,N=x.shape; pairs=[]; attrs=[]
    for b in range(B):
        n=int(node_mask[b].sum()); length=float(x[b,n-1]-x[b,0]); local=x[b,1:n]-x[b,:n-1]
        for dilation in (0,*dilations):
            if dilation==0:
                iterator=((i,i) for i in range(n))
            else:
                iterator=((i,i+dilation) for i in range(max(0,n-dilation)))
            for i,j in iterator:
                directions=((i,j),) if i==j else ((i,j),(j,i))
                for src,dst in directions:
                    dx=float(x[b,dst]-x[b,src]); adx=abs(dx); hs=float(local[min(src,n-2)]) if n>1 else length
                    hd=float(local[min(dst,n-2)]) if n>1 else length
                    pairs.append((b*N+src,b*N+dst))
                    attrs.append((dx/max(length,1e-12),adx/max(length,1e-12),math.log(adx/max(length,1e-12)+1e-8),
                                  0.0 if dx==0 else math.copysign(1.0,dx),dilation/max(max(dilations),1),
                                  math.log((hs+1e-12)/(hd+1e-12)),float(potential_valid[b,src] and potential_valid[b,dst])))
    return torch.tensor(pairs,dtype=torch.long).T.contiguous(),torch.tensor(attrs,dtype=torch.float32)


def collate_records(rows:Sequence[Mapping[str,Any]],include_targets:bool=True,
                    use_continuum_threshold:bool=CFG.primary_use_continuum_threshold,
                    dilations:Sequence[int]|None=None)->dict[str,Any]:
    B=len(rows); lengths=torch.tensor([len(r["x"]) for r in rows]); N=int(lengths.max())
    x=torch.zeros(B,N); w=torch.zeros(B,N); V=torch.zeros(B,N); V_relative=torch.zeros(B,N); V_dimensionless=torch.zeros(B,N)
    V_offset=torch.zeros(B); energy_unit=torch.ones(B); spacing=torch.ones(B,N)
    node_mask=torch.zeros(B,N,dtype=torch.bool); pvalid=torch.zeros(B,N,dtype=torch.bool); singular=torch.zeros(B,N,dtype=torch.bool)
    geometry=torch.empty(B,dtype=torch.long); bc=torch.empty(B,dtype=torch.long); endpoints=torch.zeros(B,2); kappa=torch.zeros(B)
    thresholds=torch.zeros(B,2)
    for b,r in enumerate(rows):
        validate_record_schema(r,require_targets=False); n=len(r["x"]); node_mask[b,:n]=True
        x_np=np.asarray(r["x"],dtype=np.float64); w_np=np.asarray(r["quadrature_weights"],dtype=np.float64)
        V_np=np.asarray(r["V_raw"],dtype=np.float64); pvalid_np=np.asarray(r["potential_valid_mask"],dtype=bool)
        offset,unit,U_np=dimensionless_operator_arrays(r)
        x[b,:n]=torch.as_tensor(x_np,dtype=torch.float32); w[b,:n]=torch.as_tensor(w_np,dtype=torch.float32)
        V[b,:n]=torch.as_tensor(V_np,dtype=torch.float32); V_relative[b,:n]=torch.as_tensor(V_np-offset,dtype=torch.float32)
        V_dimensionless[b,:n]=torch.as_tensor(U_np,dtype=torch.float32); V_offset[b]=offset; energy_unit[b]=unit
        spacing[b,:n]=torch.as_tensor(np.asarray(r["grid_spacing"]),dtype=torch.float32)
        pvalid[b,:n]=torch.as_tensor(pvalid_np,dtype=torch.bool); singular[b,:n]=torch.as_tensor(np.asarray(r["singularity_mask"]),dtype=torch.bool)
        geometry[b]=GEOMETRIES.index(r["geometry"]); bc[b]=BOUNDARY_CONDITIONS.index(r["boundary_condition"])
        endpoints[b]=torch.tensor([r["domain_left"],r["domain_right"]]); kappa[b]=float(r["kinetic_coefficient"])
        threshold=r.get("continuum_threshold"); thresholds[b]=torch.tensor([float(threshold is not None),0.0 if threshold is None else float(threshold)])
    domain=(endpoints[:,1]-endpoints[:,0]).clamp_min(1e-8); t=(x-endpoints[:,0,None])/domain[:,None]; xn=2*t-1
    left_distance=t; right_distance=1-t
    boundary=torch.zeros_like(node_mask); boundary[:,0]=True
    for b,n in enumerate(lengths.tolist()): boundary[b,n-1]=True
    # Only dimensionless operator/coordinate information enters the deployable network.
    # Constant shifts, translations, and exact Schrödinger dilations are therefore
    # structural covariances instead of patterns the model must memorize.
    continuous=[torch.asinh(V_dimensionless/TRAIN_U_SCALE),xn,torch.log(w/domain[:,None]+1e-8),
                torch.log(spacing/domain[:,None]+1e-8),left_distance,right_distance,boundary.float(),singular.float(),pvalid.float()]
    for freq in range(1,CFG.positional_frequencies+1):
        continuous.extend([torch.sin(freq*math.pi*(xn+1)/2),torch.cos(freq*math.pi*(xn+1)/2)])
    if use_continuum_threshold:
        threshold_present=thresholds[:,0]
        threshold_dimensionless=torch.where(threshold_present.bool(),(thresholds[:,1]-V_offset)/energy_unit,torch.zeros_like(V_offset))
        continuous.extend([threshold_present[:,None].expand(B,N),threshold_dimensionless[:,None].expand(B,N)/TRAIN_U_SCALE])
    features=torch.stack(continuous,dim=-1)*node_mask[...,None]
    edge_index,edge_attr=build_graph_edges(x,node_mask,pvalid,CFG.graph_dilations if dilations is None else tuple(dilations))
    batch={"features":features,"x":x,"t":t*node_mask,"w":w,"w_dimensionless":w/domain[:,None],"V":V,"V_relative":V_relative,
           "V_dimensionless":V_dimensionless,"V_offset":V_offset,"V_gauge":V_offset,"energy_unit":energy_unit,"node_mask":node_mask,
           "potential_valid_mask":pvalid,"singularity_mask":singular,"boundary_mask":boundary,"geometry":geometry,"bc":bc,
           "endpoints":endpoints,"kappa":kappa,"lengths":lengths,"edge_index":edge_index,"edge_attr":edge_attr,
           "batch_index":torch.arange(B).repeat_interleave(N),"operator_records":rows}
    if include_targets:
        if not all({"psi","energy","valid_state_mask"}<=set(r) for r in rows): raise KeyError("target-bearing collation requested for operator-only records")
        psi=torch.zeros(B,CFG.k_states,N); energy=torch.zeros(B,CFG.k_states); state_mask=torch.zeros(B,CFG.k_states,dtype=torch.bool)
        # All topology labels are target-only.  They are intentionally created only in
        # target-bearing collation and are neither record features nor operator metadata.
        target_node_positions=torch.zeros(B,CFG.k_states,CFG.k_states-1)
        target_node_mask=torch.zeros(B,CFG.k_states,CFG.k_states-1,dtype=torch.bool)
        target_phase_cdf=torch.zeros(B,CFG.k_states,N)
        target_phase_cdf_mask=torch.zeros(B,CFG.k_states,N,dtype=torch.bool)
        target_log_amplitude=torch.zeros(B,CFG.k_states,N)
        target_log_amplitude_mask=torch.zeros(B,CFG.k_states,N,dtype=torch.bool)
        for b,r in enumerate(rows):
            n=len(r["x"]); psi_np=np.asarray(r["psi"],dtype=np.float64); valid_np=np.asarray(r["valid_state_mask"],dtype=bool)
            psi[b,:,:n]=torch.as_tensor(psi_np,dtype=torch.float32); energy[b]=torch.as_tensor(np.asarray(r["energy"]),dtype=torch.float32); state_mask[b]=torch.as_tensor(valid_np,dtype=torch.bool)
            topology=target_topology_supervision_from_wave(
                np.asarray(r["x"],dtype=np.float64),np.asarray(r["quadrature_weights"],dtype=np.float64),
                psi_np,valid_np,np.asarray(r["singularity_mask"],dtype=bool))
            target_node_positions[b]=torch.as_tensor(topology["target_node_positions"],dtype=torch.float32)
            target_node_mask[b]=torch.as_tensor(topology["target_node_mask"],dtype=torch.bool)
            target_phase_cdf[b,:,:n]=torch.as_tensor(topology["target_phase_cdf"],dtype=torch.float32)
            target_phase_cdf_mask[b,:,:n]=torch.as_tensor(topology["target_phase_cdf_mask"],dtype=torch.bool)
            target_log_amplitude[b,:,:n]=torch.as_tensor(topology["target_log_amplitude"],dtype=torch.float32)
            target_log_amplitude_mask[b,:,:n]=torch.as_tensor(topology["target_log_amplitude_mask"],dtype=torch.bool)
        batch.update({"psi":psi,"energy":energy,"energy_relative":energy-V_offset[:,None],
                      "energy_dimensionless":(energy-V_offset[:,None])/energy_unit[:,None],"state_mask":state_mask,
                      "target_node_positions":target_node_positions,"target_node_mask":target_node_mask,
                      "target_phase_cdf":target_phase_cdf,"target_phase_cdf_mask":target_phase_cdf_mask,
                      "target_log_amplitude":target_log_amplitude,"target_log_amplitude_mask":target_log_amplitude_mask})
    return batch


def training_resampled_record(record:Mapping[str,Any],ratio:float,power:float)->dict[str,Any]:
    n=max(2*CFG.k_states+3,int(round(len(record["x"])*ratio))); t=np.linspace(0,1,n)**power
    x_new=float(record["domain_left"])+(float(record["domain_right"])-float(record["domain_left"]))*t
    return project_record_to_grid(record,x_new,"train_resampling")


class HamiltonianDataset(Dataset):
    """Stateless, epoch-keyed train transforms; validation/test records stay immutable."""
    def __init__(self,rows:Sequence[Mapping[str,Any]],augment:bool=False):
        self.rows=list(rows); self.augment=augment; self.epoch=0; self.resample_cache:dict[tuple[str,int],dict[str,Any]]={}; self.resample_failures=[]
    def set_epoch(self,epoch:int)->None: self.epoch=int(epoch)
    def __len__(self)->int: return len(self.rows)
    def __getitem__(self,index:int)->Mapping[str,Any]:
        base=self.rows[index]
        if not self.augment: return base
        digest=hashlib.sha256(f"{CFG.seed}|{self.epoch}|{base['group_id']}".encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],"little")); out=base
        if rng.random()<CFG.dilation_probability:
            out=dilated_record(out,float(np.exp(rng.uniform(math.log(.85),math.log(1.18)))))
        if out["geometry"] in ("finite_interval","truncated_line") and rng.random()<CFG.translation_probability:
            out=translated_record(out,float(rng.uniform(-.15,.15)*(out["domain_right"]-out["domain_left"])))
        if out["geometry"] in ("finite_interval","truncated_line") and rng.random()<CFG.reflection_probability:
            out=reflected_record(out)
        if rng.random()<CFG.shift_probability:
            gauge,unit,_=dimensionless_operator_arrays(out); del gauge
            amplitude=.5*max(float(np.ptp(out["energy"])),unit); out=shifted_record(out,float(rng.uniform(-amplitude,amplitude)))
        if rng.random()<CFG.resampling_probability:
            key=(base["group_id"],self.epoch)
            if key not in self.resample_cache:
                try: self.resample_cache[key]=training_resampled_record(out,float(rng.uniform(.82,1.16)),float(rng.uniform(.85,1.20)))
                except Exception as exc:
                    self.resample_failures.append({"epoch":self.epoch,"group_id":base["group_id"],"reason":str(exc)}); self.resample_cache[key]=out
            out=self.resample_cache[key]
        out=copy.deepcopy(dict(out)); out["_request_reflection_pair"]=bool(out["geometry"] in ("finite_interval","truncated_line") and rng.random()<CFG.reflection_pair_probability)
        return out


def collate_training_records(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    # Reflection peers must obey the physical microbatch memory budget.
    expanded=list(rows); pairs=[]
    for original_index,record in enumerate(rows):
        if (bool(record.get("_request_reflection_pair",False)) and
                len(expanded)<TRAINING_MICROBATCH_SIZE):
            reflected_index=len(expanded); expanded.append(reflected_record(record)); pairs.append((original_index,reflected_index))
    if len(expanded)>TRAINING_MICROBATCH_SIZE:
        raise RuntimeError("physical training batch exceeded TRAINING_MICROBATCH_SIZE")
    batch=collate_records(expanded,include_targets=True)
    batch["reflection_pairs"]=torch.tensor(pairs,dtype=torch.long).reshape(-1,2)
    batch["base_batch_size"]=len(rows)
    return batch


def make_loader(rows:Sequence[Mapping[str,Any]],shuffle:bool,include_targets:bool=True)->DataLoader:
    generator=torch.Generator().manual_seed(CFG.seed)
    dataset=HamiltonianDataset(rows,augment=shuffle and include_targets)
    collate_fn=collate_training_records if shuffle and include_targets else lambda batch:collate_records(batch,include_targets=include_targets)
    batch_size=TRAINING_MICROBATCH_SIZE if shuffle and include_targets else EVALUATION_BATCH_SIZE
    return DataLoader(dataset,batch_size=min(batch_size,max(1,len(rows))),shuffle=shuffle,
                      collate_fn=collate_fn,generator=generator,
                      num_workers=0,pin_memory=DEVICE.type=="cuda")


loaders={name:make_loader(rows,shuffle=name=="train") for name,rows in splits.items()}
sample_batch=next(iter(loaders["train"]))
print({"node_features":sample_batch["features"].shape[-1],"edge_features":sample_batch["edge_attr"].shape[-1],
       "batch_shape":tuple(sample_batch["features"].shape),"directed_edges":sample_batch["edge_index"].shape[1]})
