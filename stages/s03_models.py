"""Generated from the original notebook; execute through main.py."""

# %% [notebook cell 31]
class BasisTransform(nn.Module):
    def __init__(self,width:int,basis:str,momentum:float=0.05):
        super().__init__(); self.basis=basis; self.momentum=momentum
        self.register_buffer("running_mean",torch.zeros(width)); self.register_buffer("running_var",torch.ones(width)); self.register_buffer("updates",torch.tensor(0,dtype=torch.long))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        if self.basis in ("chebyshev","legendre"): return torch.tanh(x)
        if self.basis!="hermite": raise ValueError(self.basis)
        if self.training and x.numel()>0:
            dims=tuple(range(x.ndim-1)); mean=x.detach().float().mean(dims); var=x.detach().float().var(dims,unbiased=False).clamp_min(1e-6)
            self.running_mean.mul_(1-self.momentum).add_(mean,alpha=self.momentum)
            self.running_var.mul_(1-self.momentum).add_(var,alpha=self.momentum); self.updates.add_(1)
        return ((x-self.running_mean)/torch.sqrt(self.running_var+1e-6)).clamp(-3,3)


def polynomial_basis_values(z:torch.Tensor,degree:int,basis:str)->torch.Tensor:
    if degree<1: raise ValueError("degree must be positive")
    values=[torch.ones_like(z),z]
    for order in range(2,degree+1):
        if basis=="chebyshev": nxt=2*z*values[-1]-values[-2]
        elif basis=="legendre": nxt=((2*order-1)*z*values[-1]-(order-1)*values[-2])/order
        elif basis=="hermite": nxt=z*values[-1]/math.sqrt(order)-math.sqrt((order-1)/order)*values[-2]
        else: raise ValueError(basis)
        values.append(nxt)
    return torch.stack(values[1:],dim=-1)

# %% [notebook cell 33]
class DenseBasisKANLinear(nn.Module):
    def __init__(self,in_features:int,out_features:int,degree:int=5,basis:str="chebyshev"):
        super().__init__(); self.in_features=in_features; self.out_features=out_features; self.degree=degree; self.basis=basis
        self.transform=BasisTransform(in_features,basis); self.base=nn.Linear(in_features,out_features)
        self.coefficients=nn.Parameter(torch.empty(out_features,in_features,degree)); nn.init.normal_(self.coefficients,std=0.005)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        basis=polynomial_basis_values(self.transform(x),self.degree,self.basis)
        return self.base(F.silu(x))+torch.einsum("...id,oid->...o",basis,self.coefficients)


class FactorizedBasisKANLinear(nn.Module):
    """Rank-r approximation c[o,i,d] = sum_r output_factor[o,r] basis_factor[r,i,d]."""
    def __init__(self,in_features:int,out_features:int,degree:int=5,rank:int=4,basis:str="chebyshev"):
        super().__init__(); self.in_features=in_features; self.out_features=out_features; self.degree=degree; self.rank=rank; self.basis=basis
        self.transform=BasisTransform(in_features,basis); self.base=nn.Linear(in_features,out_features)
        self.output_factor=nn.Parameter(torch.empty(out_features,rank)); self.basis_factor=nn.Parameter(torch.empty(rank,in_features,degree))
        nn.init.normal_(self.output_factor,std=0.01); nn.init.normal_(self.basis_factor,std=0.01)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        basis=polynomial_basis_values(self.transform(x),self.degree,self.basis)
        compressed=torch.einsum("...id,rid->...r",basis,self.basis_factor)
        return self.base(F.silu(x))+compressed@self.output_factor.T
    def dense_coefficients(self)->torch.Tensor: return torch.einsum("or,rid->oid",self.output_factor,self.basis_factor)


def nonlinear_linear(in_features:int,out_features:int,use_kan:bool,placement:str)->nn.Module:
    if use_kan and placement in CFG.kan_placements:
        return FactorizedBasisKANLinear(in_features,out_features,CFG.kan_degree,CFG.kan_rank,CFG.kan_basis)
    return nn.Linear(in_features,out_features)


class FeedForwardBlock(nn.Module):
    def __init__(self,in_features:int,out_features:int,use_kan:bool=False,placement:str=""):
        super().__init__(); self.first=nonlinear_linear(in_features,out_features,use_kan,placement); self.norm=nn.LayerNorm(out_features); self.second=nonlinear_linear(out_features,out_features,use_kan,placement)
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.second(F.silu(self.norm(self.first(x))))

# %% [notebook cell 35]
class DirectContextEncoder(nn.Module):
    def __init__(self,in_features:int,hidden:int):
        super().__init__(); self.stem=nn.Linear(in_features,hidden); self.geometry=nn.Embedding(len(GEOMETRIES),hidden); self.bc=nn.Embedding(len(BOUNDARY_CONDITIONS),hidden)
        heads=4 if hidden%4==0 else 2; layer=nn.TransformerEncoderLayer(hidden,heads,4*hidden,CFG.dropout,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,max(1,CFG.graph_layers//2)); self.norm=nn.LayerNorm(hidden)
    def forward(self,batch:Mapping[str,torch.Tensor])->torch.Tensor:
        h=self.stem(batch["features"])+self.geometry(batch["geometry"])[:,None]+self.bc(batch["bc"])[:,None]
        h=self.transformer(h,src_key_padding_mask=~batch["node_mask"]); return self.norm(h)*batch["node_mask"][...,None]

# %% [notebook cell 37]
def weighted_global_pool(h:torch.Tensor,w:torch.Tensor,node_mask:torch.Tensor)->torch.Tensor:
    weight=w*node_mask; return (h*weight[...,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(1e-8)


class OperatorGraphBlock(nn.Module):
    def __init__(self,hidden:int,edge_features:int,use_kan:bool,use_global_context:bool=True):
        super().__init__(); self.use_global_context=use_global_context; self.message=FeedForwardBlock(3*hidden+edge_features,hidden,use_kan,"message"); self.update=FeedForwardBlock(3*hidden,hidden,use_kan,"update")
        self.pre=nn.LayerNorm(hidden); self.gate=nn.Parameter(torch.tensor(-1.5))
    def forward(self,h:torch.Tensor,w:torch.Tensor,node_mask:torch.Tensor,edge_index:torch.Tensor,edge_attr:torch.Tensor)->torch.Tensor:
        B,N,H=h.shape; flat=self.pre(h).reshape(B*N,H); src,dst=edge_index
        global_context=weighted_global_pool(flat.reshape(B,N,H),w,node_mask)
        if not self.use_global_context: global_context=torch.zeros_like(global_context)
        graph_of_dst=torch.div(dst,N,rounding_mode="floor")
        message=self.message(torch.cat([flat[dst],flat[src]-flat[dst],global_context[graph_of_dst],edge_attr],dim=-1))
        # Sort destinations and use a length-described segment reduction. Unlike CUDA
        # atomic index_add_, this is compatible with strict deterministic algorithms.
        order=torch.argsort(dst,stable=True); sorted_dst=dst[order]; unique_dst,counts=torch.unique_consecutive(sorted_dst,return_counts=True)
        degree=torch.zeros(B*N,dtype=torch.long,device=h.device); degree[unique_dst]=counts
        aggregate=torch.segment_reduce(message[order],"sum",lengths=degree); aggregate=aggregate/degree.clamp_min(1).to(h.dtype)[:,None]
        global_nodes=global_context[:,None,:].expand(B,N,H).reshape(B*N,H)
        delta=self.update(torch.cat([flat,aggregate,global_nodes],dim=-1)).reshape(B,N,H)
        return (h+torch.sigmoid(self.gate)*delta)*node_mask[...,None]


class GraphContextEncoder(nn.Module):
    def __init__(self,in_features:int,hidden:int,use_kan:bool,use_global_context:bool=True):
        super().__init__(); self.stem=nn.Linear(in_features,hidden); self.geometry=nn.Embedding(len(GEOMETRIES),hidden); self.bc=nn.Embedding(len(BOUNDARY_CONDITIONS),hidden)
        self.blocks=nn.ModuleList([OperatorGraphBlock(hidden,7,use_kan,use_global_context) for _ in range(CFG.graph_layers)]); self.norm=nn.LayerNorm(hidden)
    def forward(self,batch:Mapping[str,torch.Tensor])->torch.Tensor:
        h=self.stem(batch["features"])+self.geometry(batch["geometry"])[:,None]+self.bc(batch["bc"])[:,None]; h*=batch["node_mask"][...,None]
        for block in self.blocks: h=block(h,batch["w"],batch["node_mask"],batch["edge_index"],batch["edge_attr"])
        return self.norm(h)*batch["node_mask"][...,None]


def repeat_graph_for_states(batch:Mapping[str,torch.Tensor],states:int)->dict[str,torch.Tensor]:
    B,N=batch["x"].shape; src,dst=batch["edge_index"]; state=torch.arange(states,device=src.device)[:,None]
    graph=torch.div(src,N,rounding_mode="floor")[None]; local_src=(src%N)[None]; local_dst=(dst%N)[None]
    repeated_src=((graph*states+state)*N+local_src).reshape(-1); repeated_dst=((graph*states+state)*N+local_dst).reshape(-1)
    edge_index=torch.stack([repeated_src,repeated_dst])
    return {"w":batch["w"][:,None].expand(B,states,N).reshape(B*states,N),
            "node_mask":batch["node_mask"][:,None].expand(B,states,N).reshape(B*states,N),
            "edge_index":edge_index,"edge_attr":batch["edge_attr"][None].expand(states,-1,-1).reshape(-1,batch["edge_attr"].shape[-1]),
            "geometry":batch["geometry"][:,None].expand(B,states).reshape(-1),"bc":batch["bc"][:,None].expand(B,states).reshape(-1)}

# %% [notebook cell 39]
def geometry_canonical_coordinate(batch:Mapping[str,torch.Tensor])->torch.Tensor:
    """Invert the solver's fixed mesh maps to a dimensionless computational coordinate."""
    t=batch["t"].clamp(0,1); geometry=batch["geometry"][:,None]; coordinate=t
    truncated=geometry==GEOMETRIES.index("truncated_line"); a=1.1
    truncated_coordinate=.5*(1+torch.asinh((2*t-1)*math.sinh(a))/a)
    half=geometry==GEOMETRIES.index("half_line"); half_coordinate=torch.sqrt(t.clamp_min(0))
    radial=geometry==GEOMETRIES.index("radial_reduced"); radial_coordinate=t.clamp_min(0).pow(1/2.2)
    singular=geometry==GEOMETRIES.index("singular_interval"); singular_coordinate=torch.acos((1-2*t).clamp(-1,1))/math.pi
    coordinate=torch.where(truncated,truncated_coordinate,coordinate)
    coordinate=torch.where(half,half_coordinate,coordinate); coordinate=torch.where(radial,radial_coordinate,coordinate)
    coordinate=torch.where(singular,singular_coordinate,coordinate)
    return coordinate*batch["node_mask"]


def normalized_sine_basis(batch:Mapping[str,torch.Tensor],modes:int)->torch.Tensor:
    mode=torch.arange(1,modes+1,device=batch["x"].device,dtype=batch["x"].dtype)
    coordinate=geometry_canonical_coordinate(batch)
    return math.sqrt(2.0)*torch.sin(math.pi*coordinate[...,None]*mode)*batch["node_mask"][...,None]


def wave_to_spectral(batch:Mapping[str,torch.Tensor],psi:torch.Tensor,modes:int=CFG.spectral_modes)->torch.Tensor:
    length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8)
    phi=psi*torch.sqrt(length)[:,None,None]; basis=normalized_sine_basis(batch,modes)
    # Model grids can be nonuniform, so continuum sine orthogonality is not exact under
    # their stored quadrature.  Solve the discrete weighted projection instead of
    # treating raw inner products as expansion coefficients.
    with torch.autocast(device_type=psi.device.type,enabled=False):
        basis32=basis.float(); weights=batch["w_dimensionless"].float()
        gram=torch.einsum("bnm,bn,bnl->bml",basis32,weights,basis32)
        rhs=torch.einsum("bkn,bn,bnm->bkm",phi.float(),weights,basis32)
        identity=torch.eye(modes,device=psi.device,dtype=torch.float32)[None]
        # A small Tikhonov term is required on strongly mapped grids where the highest
        # sampled sine modes become numerically collinear in float32.
        factor,info=torch.linalg.cholesky_ex(gram+1e-4*identity)
        if bool(torch.any(info)):
            failed=torch.nonzero(info).flatten().tolist(); raise RuntimeError(f"spectral projection solve failed for batches {failed}")
        coefficients=torch.cholesky_solve(rhs.transpose(1,2),factor).transpose(1,2)
    return coefficients.to(psi.dtype)


def spectral_to_wave(coefficients:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8); basis=normalized_sine_basis(batch,coefficients.shape[-1])
    raw=torch.einsum("bkm,bnm->bkn",coefficients,basis)/torch.sqrt(length)[:,None,None]
    raw=raw*batch["node_mask"][:,None]; raw=raw.masked_fill(batch["boundary_mask"][:,None],0.0)
    norm=torch.sqrt(torch.sum(batch["w"][:,None]*raw.square(),dim=-1,keepdim=True).clamp_min(1e-10))
    return raw/norm*batch["node_mask"][:,None]


class SolutionEncoder(nn.Module):
    """A genuine state-global spectral bottleneck with no operator-metadata shortcut."""
    def __init__(self,hidden:int,use_kan:bool=True):
        super().__init__(); del hidden,use_kan
        self.residual=nn.Sequential(nn.LayerNorm(CFG.latent_dim),nn.Linear(CFG.latent_dim,2*CFG.latent_dim),nn.SiLU(),nn.Linear(2*CFG.latent_dim,CFG.latent_dim))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
    def forward(self,batch:Mapping[str,torch.Tensor],psi:torch.Tensor)->torch.Tensor:
        coefficients=wave_to_spectral(batch,psi,CFG.latent_dim)
        return coefficients+0.05*self.residual(coefficients)


class SolutionReconstructionDecoder(nn.Module):
    def __init__(self,hidden:int):
        super().__init__(); del hidden; self.head=nn.Linear(CFG.latent_dim,CFG.spectral_modes)
        with torch.no_grad(): self.head.weight.copy_(torch.eye(CFG.spectral_modes,CFG.latent_dim)); self.head.bias.zero_()
    def forward(self,z:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
        raw=spectral_to_wave(self.head(z),batch)
        return weighted_cholesky_orthonormalize_torch(raw,batch["w"])

# %% [notebook cell 41]
solution_online_encoder=SolutionEncoder(CFG.preset.hidden,use_kan=CFG.backbone=="gnn_kan").to(DEVICE)
solution_reconstruction_decoder=SolutionReconstructionDecoder(CFG.preset.hidden).to(DEVICE)

# %% [notebook cell 43]
def initialize_ema_target(online:nn.Module)->nn.Module:
    target=copy.deepcopy(online).eval()
    for parameter in target.parameters(): parameter.requires_grad_(False)
    return target


@torch.no_grad()
def ema_update(online:nn.Module,target:nn.Module,tau:float)->None:
    online_params=dict(online.named_parameters()); target_params=dict(target.named_parameters())
    if online_params.keys()!=target_params.keys(): raise RuntimeError("EMA parameter structure mismatch")
    for name,tp in target_params.items(): tp.mul_(tau).add_(online_params[name].detach(),alpha=1-tau)
    online_buffers=dict(online.named_buffers()); target_buffers=dict(target.named_buffers())
    for name,tb in target_buffers.items():
        ob=online_buffers[name]
        if torch.is_floating_point(tb): tb.mul_(tau).add_(ob.detach(),alpha=1-tau)
        else: tb.copy_(ob)


def ema_tau(step:int,total_steps:int)->float:
    progress=min(max(step/max(total_steps,1),0.0),1.0)
    return CFG.ema_tau_end-(CFG.ema_tau_end-CFG.ema_tau_start)*0.5*(1+math.cos(math.pi*progress))


solution_target_encoder=initialize_ema_target(solution_online_encoder).to(DEVICE)
assert solution_target_encoder is not solution_online_encoder

# %% [notebook cell 45]
def make_context_encoder(backbone:str,in_features:int,hidden:int,use_global_context:bool=True)->nn.Module:
    if backbone=="direct_baseline": return DirectContextEncoder(in_features,hidden)
    if backbone=="gnn_mlp": return GraphContextEncoder(in_features,hidden,use_kan=False,use_global_context=use_global_context)
    if backbone=="gnn_kan": return GraphContextEncoder(in_features,hidden,use_kan=True,use_global_context=use_global_context)
    raise ValueError(backbone)


potential_context_encoder=make_context_encoder(CFG.backbone,sample_batch["features"].shape[-1],CFG.preset.hidden).to(DEVICE)

# %% [notebook cell 47]
class StateConditionedPredictor(nn.Module):
    def __init__(self,hidden:int,use_kan:bool,use_global_context:bool=True):
        super().__init__(); self.use_global_context=use_global_context; self.state_queries=nn.Embedding(CFG.k_states,hidden); self.input=FeedForwardBlock(3*hidden,hidden,use_kan,"predictor")
        self.graph=OperatorGraphBlock(hidden,7,use_kan,use_global_context); heads=4 if hidden%4==0 else 2
        self.state_attention=nn.MultiheadAttention(hidden,heads,batch_first=True); self.token_update=FeedForwardBlock(2*hidden,hidden,use_kan,"predictor"); self.norm=nn.LayerNorm(hidden)
        self.latent_projection=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,CFG.latent_dim))
    def forward(self,context:torch.Tensor,batch:Mapping[str,torch.Tensor])->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        B,N,H=context.shape; K=CFG.k_states; global_context=weighted_global_pool(context,batch["w"],batch["node_mask"])
        if not self.use_global_context: global_context=torch.zeros_like(global_context)
        queries=self.state_queries.weight[None,:,None,:].expand(B,K,N,H); ctx=context[:,None].expand(B,K,N,H); glob=global_context[:,None,None,:].expand(B,K,N,H)
        z=self.input(torch.cat([ctx,queries,glob],dim=-1)).reshape(B*K,N,H); repeated=repeat_graph_for_states(batch,K)
        z=self.graph(z,repeated["w"],repeated["node_mask"],repeated["edge_index"],repeated["edge_attr"]).reshape(B,K,N,H)
        weights=batch["w"]*batch["node_mask"]; tokens=(z*weights[:,None,:,None]).sum(2)/weights.sum(1)[:,None,None].clamp_min(1e-8)
        mixed,_=self.state_attention(tokens,tokens,tokens,need_weights=False); update=self.token_update(torch.cat([tokens,mixed],dim=-1))
        z=self.norm(z+update[:,:,None,:])*batch["node_mask"][:,None,:,None]
        tokens=(z*weights[:,None,:,None]).sum(2)/weights.sum(1)[:,None,None].clamp_min(1e-8)
        return z,self.latent_projection(tokens),tokens,global_context

# %% [notebook cell 49]
def weighted_cholesky_orthonormalize_torch(psi:torch.Tensor,w:torch.Tensor,jitter:float=CFG.orth_eigen_floor)->torch.Tensor:
    """Smooth weighted row orthonormalization without degenerate Gram eigenvectors."""
    gram=torch.einsum("bkn,bn,bjn->bkj",psi,w,psi).float()
    identity=torch.eye(gram.shape[-1],device=gram.device,dtype=gram.dtype).expand_as(gram)
    scale=torch.diagonal(gram,dim1=-2,dim2=-1).mean(-1).clamp_min(1.0)
    factor,info=torch.linalg.cholesky_ex(gram+jitter*scale[:,None,None]*identity)
    if bool(torch.any(info)):
        failed=torch.nonzero(info).flatten().tolist()
        raise RuntimeError(f"weighted Cholesky orthonormalization failed for batches {failed}")
    return torch.linalg.solve_triangular(factor,psi.float(),upper=False).to(psi.dtype)


def safe_weighted_cholesky_orthonormalize_torch(psi:torch.Tensor,w:torch.Tensor,jitter:float=CFG.orth_eigen_floor)->tuple[torch.Tensor,torch.Tensor]:
    """Per-Hamiltonian Cholesky with a recoverable direct-output fallback."""
    gram=torch.einsum("bkn,bn,bjn->bkj",psi,w,psi).float(); K=gram.shape[-1]
    identity=torch.eye(K,device=gram.device,dtype=gram.dtype)[None].expand_as(gram)
    scale=torch.diagonal(gram,dim1=-2,dim2=-1).mean(-1).clamp_min(1.0)
    factor,info=torch.linalg.cholesky_ex(gram+jitter*scale[:,None,None]*identity); success=info==0
    safe_factor=torch.where(success[:,None,None],factor,identity)
    solved=torch.linalg.solve_triangular(safe_factor,psi.float(),upper=False).to(psi.dtype)
    finite=torch.isfinite(solved).flatten(1).all(-1); success=success&finite
    return torch.where(success[:,None,None],solved,psi),success


class SignedWavefunctionDecoder(nn.Module):
    def __init__(self,hidden:int,use_kan:bool):
        super().__init__(); self.pre=FeedForwardBlock(CFG.latent_dim,hidden,use_kan,"decoder"); self.head=nn.Linear(hidden,CFG.spectral_modes); self.residual_gate=nn.Parameter(torch.tensor(-0.5))
        nn.init.normal_(self.head.weight,std=0.002); nn.init.zeros_(self.head.bias)
    def forward(self,z_pred:torch.Tensor,batch:Mapping[str,torch.Tensor],orthonormalize:bool)->tuple[torch.Tensor,torch.Tensor]:
        B,K,_=z_pred.shape; carrier=torch.zeros(B,K,CFG.spectral_modes,device=z_pred.device,dtype=z_pred.dtype)
        diagonal=torch.arange(min(K,CFG.spectral_modes),device=z_pred.device); carrier[:,diagonal,diagonal]=1.0
        coefficients=carrier+torch.sigmoid(self.residual_gate)*self.head(self.pre(z_pred))
        pre=spectral_to_wave(coefficients,batch)
        post=weighted_cholesky_orthonormalize_torch(pre,batch["w"]) if orthonormalize else pre
        post=post*batch["node_mask"][:,None]
        if CFG.boundary_mode!="soft": post=post.masked_fill(batch["boundary_mask"][:,None],0.0)
        return pre,post


# Explicit name used by ablations and checkpoint metadata; the historical class name is
# retained so pre-v6 code paths remain executable.
SpectralSineResidualDecoder=SignedWavefunctionDecoder


def deterministic_prefix_sum(values:torch.Tensor,dim:int=-1)->torch.Tensor:
    """Inclusive floating-point prefix sum accepted by strict CUDA determinism.

    CUDA ``torch.cumsum`` has no deterministic floating/complex implementation.
    This out-of-place Hillis--Steele scan uses only fixed-offset additions, costs
    O(log N) tensor additions, supports autograd, and avoids an O(N^2) triangular
    matrix that would be impractical for the notebook's largest grids.
    """
    canonical_dim=dim if dim>=0 else values.ndim+dim
    if canonical_dim<0 or canonical_dim>=values.ndim:
        raise IndexError(f"prefix-sum dimension {dim} is invalid for rank-{values.ndim} input")
    count=values.shape[canonical_dim]
    if count==0:
        return values.clone()
    result=values.movedim(canonical_dim,-1)
    offset=1
    while offset<count:
        shifted=torch.cat([torch.zeros_like(result[...,:offset]),result[...,:-offset]],dim=-1)
        result=result+shifted
        offset*=2
    return result.movedim(-1,canonical_dim)


class TopologyPhaseDecoder(nn.Module):
    """Positive-amplitude, strictly monotone-phase Sturm decoder.

    The learned fields are nodewise functions of ``z_nodes``.  Cumulative integration,
    gauge removal, and normalization run in float32 even under mixed precision.  No
    target tensor, family label, or generator parameter is accepted by this module.
    """
    def __init__(self,hidden:int,use_kan:bool,phase_q_min:float=CFG.phase_q_min,
                 phase_logit_clip:float=CFG.phase_logit_clip,
                 amplitude_log_clip:float=CFG.amplitude_log_clip):
        super().__init__(); self.phase_q_min=float(phase_q_min); self.phase_logit_clip=float(phase_logit_clip); self.amplitude_log_clip=float(amplitude_log_clip)
        self.field=FeedForwardBlock(hidden,hidden,use_kan,"decoder")
        self.amplitude_head=nn.Linear(hidden,1); self.phase_density_head=nn.Linear(hidden,1)
        nn.init.normal_(self.amplitude_head.weight,std=0.002); nn.init.zeros_(self.amplitude_head.bias)
        nn.init.normal_(self.phase_density_head.weight,std=0.002); nn.init.zeros_(self.phase_density_head.bias)

    def forward(self,z_nodes:torch.Tensor,batch:Mapping[str,torch.Tensor])->dict[str,torch.Tensor]:
        learned=self.field(z_nodes)
        u=self.amplitude_head(learned).squeeze(-1)
        g=self.phase_density_head(learned).squeeze(-1)
        device_type=z_nodes.device.type if z_nodes.device.type in ("cpu","cuda") else "cpu"
        with torch.autocast(device_type=device_type,enabled=False):
            u=u.float(); g=g.float(); t=batch["t"].float(); w_t=batch["w_dimensionless"].float()
            valid=batch["node_mask"]; interval_valid=valid[:,:-1]&valid[:,1:]
            dt=t[:,1:]-t[:,:-1]
            # Bound the learned logit by construction.  Together with q_min this keeps
            # the density ratio finite for every finite network output while preserving
            # smooth, non-saturating gradients around the learned operating range.
            g_bounded=self.phase_logit_clip*torch.tanh(g/self.phase_logit_clip)
            q_raw=(self.phase_q_min+F.softplus(g_bounded))*valid[:,None]
            raw_increment=0.5*(q_raw[...,1:]+q_raw[...,:-1])*dt[:,None]*interval_valid[:,None]
            raw_integral=raw_increment.sum(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
            q=q_raw/raw_integral
            integrated_mass=0.5*(q[...,1:]+q[...,:-1])*dt[:,None]*interval_valid[:,None]
            integrated_mass=integrated_mass/integrated_mass.sum(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
            # A tiny representability floor is a numerical enforcement, not a learned
            # topology prior.  It prevents positive trapezoidal masses from rounding to
            # zero when accumulated near C=1 in float32.  The total correction is below
            # 0.1% even for grids with one thousand intervals.
            cdf_increment_floor=8*torch.finfo(torch.float32).eps
            interval_count=interval_valid.sum(-1).to(integrated_mass.dtype)[:,None,None]
            floor_budget=(cdf_increment_floor*interval_count).clamp_max(0.01)
            phase_increment_cdf=(integrated_mass*(1-floor_budget)+
                                 cdf_increment_floor*interval_valid[:,None].to(integrated_mass.dtype))
            phase_increment_cdf=phase_increment_cdf/phase_increment_cdf.sum(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
            cdf=torch.cat([torch.zeros_like(q[...,:1]),deterministic_prefix_sum(phase_increment_cdf,dim=-1)],dim=-1)
            last=(batch["lengths"]-1).to(cdf.device); gather_last=last[:,None,None].expand(-1,cdf.shape[1],1)
            cdf_last=torch.gather(cdf,-1,gather_last).clamp_min(torch.finfo(torch.float32).eps)
            cdf=(cdf/cdf_last)*valid[:,None]
            cdf=cdf.scatter(-1,gather_last,torch.ones_like(cdf_last)); cdf[...,0]=0.0
            # Recompute from the endpoint-normalized CDF so diagnostics exactly match
            # the phase actually used to synthesize the waveform.
            phase_increment_cdf=(cdf[...,1:]-cdf[...,:-1])*interval_valid[:,None]
            states=torch.arange(1,CFG.k_states+1,device=cdf.device,dtype=cdf.dtype)[None,:,None]
            theta=states*math.pi*cdf

            log_amplitude=self.amplitude_log_clip*torch.tanh(u/self.amplitude_log_clip)
            gauge=(log_amplitude*w_t[:,None]).sum(-1,keepdim=True)/w_t.sum(-1,keepdim=True)[:,None].clamp_min(1e-12)
            log_amplitude=(log_amplitude-gauge)*valid[:,None]
            amplitude=torch.exp(log_amplitude)*valid[:,None]
            phi_raw=amplitude*torch.sin(theta)
            phi_raw=phi_raw.masked_fill(batch["boundary_mask"][:,None],0.0)*valid[:,None]
            norm_square=(phi_raw.square()*w_t[:,None]).sum(-1,keepdim=True)
            norm_t=torch.sqrt(norm_square.clamp_min(torch.finfo(norm_square.dtype).tiny))
            phi=phi_raw/norm_t
            length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).float().clamp_min(1e-12)
            psi=phi/torch.sqrt(length)[:,None,None]
            psi=psi.masked_fill(batch["boundary_mask"][:,None],0.0)*valid[:,None]
            rho=psi.square()
            physical_norm=(rho*batch["w"].float()[:,None]).sum(-1)
            delta_theta=states*math.pi*phase_increment_cdf

            # Phase quantiles are diagnostics/metrics; search indices are detached while
            # interpolation remains differentiable through C and t.
            J=CFG.k_states-1
            j=torch.arange(1,J+1,device=cdf.device,dtype=cdf.dtype)[None,None,:]
            state_number=torch.arange(CFG.k_states,device=cdf.device)[None,:,None]
            quantile_mask=j<=state_number
            levels=j/(state_number.to(cdf.dtype)+1.0)
            cdf_search=torch.where(valid[:,None],cdf,torch.full_like(cdf,2.0)).contiguous()
            right=torch.searchsorted(cdf_search.detach(),levels.expand(cdf.shape[0],-1,-1).contiguous(),right=False)
            right=right.clamp(1,cdf.shape[-1]-1); left=right-1
            c0=torch.gather(cdf,-1,left); c1=torch.gather(cdf,-1,right)
            t_expand=t[:,None].expand(-1,CFG.k_states,-1)
            t0=torch.gather(t_expand,-1,left); t1=torch.gather(t_expand,-1,right)
            alpha=((levels-c0)/(c1-c0).clamp_min(1e-12)).clamp(0,1)
            predicted_nodes=(t0+alpha*(t1-t0))*quantile_mask

        return {"psi_direct":psi,"rho_direct":rho,"phi_direct":phi,
                "amplitude":amplitude,"log_amplitude":log_amplitude,
                "phase_density_logit":g_bounded*valid[:,None],
                "phase_density_raw":q_raw,"phase_density":q,"phase_cdf":cdf,"theta":theta,
                "phase_cdf_increment":phase_increment_cdf,"phase_increment":delta_theta,
                "predicted_node_positions_t":predicted_nodes,"predicted_node_mask":quantile_mask.expand(cdf.shape[0],-1,-1),
                "dimensionless_normalizer":norm_t.squeeze(-1),"physical_normalization":physical_norm,
                "normalization_error":(physical_norm-1).abs(),
                "phase_cdf_stabilization_floor":torch.full_like(physical_norm,cdf_increment_floor),
                "minimum_phase_increment":torch.where(interval_valid[:,None],delta_theta,torch.full_like(delta_theta,float("inf"))).amin(-1),
                "maximum_phase_increment":delta_theta.amax(-1)}

# %% [notebook cell 51]
ORTHONORMALIZATION_OPTIONS=("loss_only","lowdin","lowdin_plus_loss","rayleigh_ritz")
assert CFG.orthonormalization in ORTHONORMALIZATION_OPTIONS

# %% [notebook cell 53]
def deterministic_middle_median(values:torch.Tensor,dim:int=-1)->torch.Tensor:
    """Value-only lower median that remains valid under strict deterministic CUDA.

    ``torch.median(values, dim=...)`` also returns an index.  CUDA cannot choose a
    deterministic index when equal values straddle the median, so PyTorch rejects
    that overload when deterministic algorithms are required.  A stable sort followed
    by the lower-middle selection exactly preserves ``torch.median(..., dim=...)``
    value semantics for both odd and even widths.
    """
    canonical_dim=dim if dim>=0 else values.ndim+dim
    if canonical_dim<0 or canonical_dim>=values.ndim:
        raise IndexError(f"median dimension {dim} is invalid for rank-{values.ndim} input")
    count=values.shape[canonical_dim]
    if count<1:
        raise ValueError("deterministic median requires a nonempty dimension")
    ordered=torch.sort(values,dim=canonical_dim,stable=True).values
    return ordered.select(canonical_dim,(count-1)//2)


@torch.no_grad()
def dimensionless_sine_galerkin_energy_reference(batch:Mapping[str,torch.Tensor])->dict[str,torch.Tensor]:
    """Return a target-free K-state operator reference and robust energy scales.

    Physical Dirichlet trials ``sqrt(2/L) sin(m*pi*t)`` are discretely orthonormalized
    with the stored nonuniform quadrature.  The projected weak Hamiltonian is assembled
    in float64 after subtracting a weighted operator origin, which avoids cancellation
    in deep spectra.  Padded nodes and intervals are excluded exactly.
    """
    potential=batch["V_dimensionless"]
    device_type=potential.device.type if potential.device.type in ("cpu","cuda") else "cpu"
    with torch.autocast(device_type=device_type,enabled=False):
        dtype=torch.float64; B,N=batch["t"].shape; K=CFG.k_states
        valid=batch["node_mask"]&batch["potential_valid_mask"]
        if not bool(valid.any(-1).all()):
            raise ValueError("operator energy reference requires a valid potential node in every sample")
        node=batch["node_mask"]; interval=node[:,1:]&node[:,:-1]
        t=batch["t"].to(dtype); modes=torch.arange(1,K+1,device=t.device,dtype=dtype)
        trials=math.sqrt(2.0)*torch.sin(math.pi*t[:,None,:]*modes[None,:,None])
        trials=trials*node[:,None].to(dtype)
        trials=trials.masked_fill(batch["boundary_mask"][:,None],0.0)
        w=batch["w_dimensionless"].to(dtype)
        gram=torch.einsum("bkn,bn,bjn->bkj",trials,w,trials)
        identity=torch.eye(K,device=t.device,dtype=dtype)[None].expand(B,-1,-1)
        factor,info=torch.linalg.cholesky_ex(.5*(gram+gram.transpose(-1,-2)))
        if bool(torch.any(info)):
            factor,info=torch.linalg.cholesky_ex(.5*(gram+gram.transpose(-1,-2))+1e-12*identity)
        if bool(torch.any(info)):
            failed=torch.nonzero(info,as_tuple=False).flatten().tolist()
            raise RuntimeError(f"sine-Galerkin reference orthonormalization failed for batches {failed}")
        q=torch.linalg.solve_triangular(factor,trials,upper=False)

        mass=w*valid.to(dtype); normalized_mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-30)
        V=potential.to(dtype); origin=(normalized_mass*V).sum(-1)
        centered_V=V-origin[:,None]
        dt=t[:,1:]-t[:,:-1]
        safe_dt=torch.where(interval,dt,torch.ones_like(dt)).clamp_min(1e-14)
        flux=(q[:,:,1:]-q[:,:,:-1])*interval[:,None].to(dtype)/safe_dt[:,None]
        stiffness=torch.zeros_like(q); stiffness[:,:,:-1]-=flux; stiffness[:,:,1:]+=flux
        weak_action=stiffness+w[:,None]*centered_V[:,None]*q
        projected=torch.einsum("bkn,bjn->bkj",q,weak_action)
        projected=.5*(projected+projected.transpose(-1,-2))
        eigenvalues=torch.linalg.eigvalsh(projected)+origin[:,None]

        raw_gaps=torch.diff(eigenvalues,dim=1)
        box_gap_factor=2*torch.arange(K-1,device=t.device,dtype=dtype)+3
        normalized_gaps=(raw_gaps/box_gap_factor[None]).abs()
        # ``torch.median(..., dim=...)`` is forbidden by strict deterministic mode on
        # CUDA because that kernel also emits nondeterministic tie indices.
        operator_scale=deterministic_middle_median(normalized_gaps,dim=1)
        operator_scale=operator_scale.clamp_min(float(CFG.energy_scale_floor))
        reference_gaps=raw_gaps.clamp_min(float(CFG.energy_reference_gap_floor)*operator_scale[:,None])
        reference=torch.cat([eigenvalues[:,:1],eigenvalues[:,:1]+deterministic_prefix_sum(reference_gaps,dim=1)],dim=1)
        potential_floor=torch.where(valid,V,torch.full_like(V,float("inf"))).amin(-1)
        ground_scale=torch.maximum(operator_scale,(reference[:,0]-potential_floor).abs()).clamp_min(float(CFG.energy_scale_floor))
    return {"energy":reference.float(),"gaps":reference_gaps.float(),
            "operator_scale":operator_scale.float(),"ground_scale":ground_scale.float(),
            "potential_floor":potential_floor.float(),"projected_hamiltonian":projected.float()}


class OperatorScaledGroundLogGapHead(nn.Module):
    def __init__(self,hidden:int,use_kan:bool):
        super().__init__(); self.ground=FeedForwardBlock(2*hidden,hidden,use_kan,"energy"); self.ground_out=nn.Linear(hidden,1)
        self.gap=FeedForwardBlock(3*hidden,hidden,use_kan,"energy"); self.gap_out=nn.Linear(hidden,1)
        # Zero coordinates reproduce the operator reference. Tiny weights break exact
        # batch symmetry without beginning from arbitrary raw energy magnitudes.
        nn.init.normal_(self.ground_out.weight,std=.001); nn.init.zeros_(self.ground_out.bias)
        nn.init.normal_(self.gap_out.weight,std=.001); nn.init.zeros_(self.gap_out.bias)

    def forward(self,tokens:torch.Tensor,global_context:torch.Tensor,
                batch:Mapping[str,torch.Tensor])->dict[str,torch.Tensor]:
        reference=dimensionless_sine_galerkin_energy_reference(batch)
        raw_ground=self.ground_out(self.ground(torch.cat([tokens[:,0],global_context],dim=-1))).squeeze(-1)
        adjacent=torch.cat([tokens[:,1:],tokens[:,:-1],global_context[:,None].expand(-1,CFG.k_states-1,-1)],dim=-1)
        raw_log_gaps=self.gap_out(self.gap(adjacent)).squeeze(-1)
        device_type=tokens.device.type if tokens.device.type in ("cpu","cuda") else "cpu"
        with torch.autocast(device_type=device_type,enabled=False):
            ground_clip=float(CFG.energy_ground_residual_clip); gap_clip=float(CFG.energy_log_gap_clip)
            ground_coordinate=ground_clip*torch.tanh(raw_ground.float()/ground_clip)
            log_gap_ratio=gap_clip*torch.tanh(raw_log_gaps.float()/gap_clip)
            ground_energy=reference["energy"][:,0]+reference["ground_scale"]*ground_coordinate
            nominal_gaps=reference["gaps"]*torch.exp(log_gap_ratio)
            # Add a differentiable ULP budget so strict ordering survives conversion to
            # float32 even for a large common negative ground-energy offset.
            machine_epsilon=torch.finfo(torch.float32).eps; cumulative_nominal=deterministic_prefix_sum(nominal_gaps,dim=1)
            representable_gap=4*machine_epsilon*(ground_energy.abs()[:,None]+reference["operator_scale"][:,None]+cumulative_nominal)
            gaps=nominal_gaps+representable_gap
            epsilon=torch.cat([ground_energy[:,None],ground_energy[:,None]+deterministic_prefix_sum(gaps,dim=1)],dim=1)
        return {"epsilon":epsilon,"ground_energy":ground_energy,"gaps":gaps,
                "ground_residual_scaled":ground_coordinate,"log_gap_ratio":log_gap_ratio,
                "raw_ground_residual_scaled":raw_ground,"raw_log_gap_ratio":raw_log_gaps,
                "reference_energy":reference["energy"],"reference_gaps":reference["gaps"],
                "operator_scale":reference["operator_scale"],"ground_scale":reference["ground_scale"],
                "operator_floor":reference["potential_floor"],
                "reference_projected_hamiltonian":reference["projected_hamiltonian"]}


# Compatibility import name; checkpoint architecture metadata distinguishes the new
# parameterization and old checkpoints require an explicit non-strict migration.
StateEnergyHead=OperatorScaledGroundLogGapHead


@torch.no_grad()
def exact_energy_coordinate_audit(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    """Training-only audit that exact targets lie inside the head coordinates.

    Exact energies are assembled separately from an operator-only batch and are never
    passed to ``forward_operator`` or the head.  This diagnostic prevents a generous
    clip from silently making some supervised spectra unrepresentable.
    """
    if not rows: raise ValueError("energy-coordinate audit requires target-bearing records")
    operator=collate_records([operator_only_record(row) for row in rows],include_targets=False)
    reference=dimensionless_sine_galerkin_energy_reference(operator)
    exact_physical=torch.stack([torch.as_tensor(np.asarray(row["energy"]),dtype=torch.float32) for row in rows])
    exact=(exact_physical-operator["V_gauge"][:,None])/operator["energy_unit"][:,None]
    exact_gaps=torch.diff(exact,dim=1)
    if not bool((exact_gaps>0).all()): raise ValueError("exact spectrum supplied to the energy audit is not strictly ordered")
    ground=(exact[:,0]-reference["energy"][:,0])/reference["ground_scale"].clamp_min(CFG.energy_scale_floor)
    log_gaps=torch.log(exact_gaps/reference["gaps"].clamp_min(
        CFG.energy_reference_gap_floor*reference["operator_scale"][:,None]))
    def distribution(values:torch.Tensor)->dict[str,float]:
        flat=values.detach().float().flatten(); absolute=flat.abs()
        return {"minimum":float(flat.min()),"p01":float(torch.quantile(flat,.01)),"p05":float(torch.quantile(flat,.05)),
                "median":float(torch.quantile(flat,.50)),"p95":float(torch.quantile(flat,.95)),
                "p99":float(torch.quantile(flat,.99)),"maximum":float(flat.max()),
                "absolute_p95":float(torch.quantile(absolute,.95)),"absolute_p99":float(torch.quantile(absolute,.99)),
                "absolute_maximum":float(absolute.max())}
    ground_hit=(ground.abs()>=CFG.energy_ground_residual_clip); gap_hit=(log_gaps.abs()>=CFG.energy_log_gap_clip)
    return {"hamiltonians":len(rows),"states":CFG.k_states,
            "ground_residual_scaled":distribution(ground),"log_gap_ratio":distribution(log_gaps),
            "ground_clip":CFG.energy_ground_residual_clip,"log_gap_clip":CFG.energy_log_gap_clip,
            "ground_clip_hit_fraction":float(ground_hit.float().mean()),"log_gap_clip_hit_fraction":float(gap_hit.float().mean()),
            "operator_scale_minimum":float(reference["operator_scale"].min()),"operator_scale_maximum":float(reference["operator_scale"].max()),
            "ground_scale_minimum":float(reference["ground_scale"].min()),"ground_scale_maximum":float(reference["ground_scale"].max()),
            "exact_gap_minimum":float(exact_gaps.min()),"reference_gap_minimum":float(reference["gaps"].min()),
            "all_finite":bool(torch.isfinite(ground).all() and torch.isfinite(log_gaps).all())}


def stable_eigh_rotation(projected:torch.Tensor,detach_vectors:bool=True)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Eigh with eigenvector gradients evaluated only on safely separated samples.

    Multiplying unsafe eigenvector gradients by zero is not sufficient: the singular
    backward of a degenerate batched ``eigh`` can yield ``0 * inf = NaN``.  Therefore
    differentiable eigenvectors are recomputed on the safe sub-batch only.
    """
    epsilon,vectors=torch.linalg.eigh(projected)
    spectral_span=(epsilon[:,-1]-epsilon[:,0]).clamp_min(1.0)
    relative_min_gap=torch.diff(epsilon,dim=1).amin(-1)/spectral_span
    safe_gradient=relative_min_gap.detach()>=CFG.ritz_gradient_min_relative_gap
    rotation=vectors.detach()
    if not detach_vectors:
        safe_indices=torch.nonzero(safe_gradient,as_tuple=False).flatten()
        if safe_indices.numel()>0:
            _,safe_vectors=torch.linalg.eigh(projected.index_select(0,safe_indices))
            rotation=rotation.index_copy(0,safe_indices,safe_vectors)
    return epsilon,rotation,relative_min_gap,safe_gradient


def dimensionless_weak_hamiltonian_action(psi:torch.Tensor,batch:Mapping[str,torch.Tensor])->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Apply the exact weak operator used by Rayleigh--Ritz on the stored model grid.

    ``phi=sqrt(L)*psi`` is normalized with the dimensionless quadrature.  Returning the
    weak action, rather than mixing in the independent strong-form audit stencil, keeps
    every recurrent correction gauge/dilation covariant and Galerkin-consistent.
    """
    q=psi.float(); length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).float().clamp_min(1e-8)
    phi=q*torch.sqrt(length)[:,None,None]; t=batch["t"].float(); dt=t[:,1:]-t[:,:-1]
    interval=batch["node_mask"][:,1:]&batch["node_mask"][:,:-1]
    safe_dt=torch.where(interval,dt,torch.ones_like(dt)).clamp_min(1e-8); coefficient=interval.to(phi.dtype)/safe_dt
    flux=(phi[:,:,1:]-phi[:,:,:-1])*coefficient[:,None]
    stiffness=torch.zeros_like(phi); stiffness[:,:,:-1]-=flux; stiffness[:,:,1:]+=flux
    w_dimensionless=batch["w_dimensionless"].float(); potential=batch["V_dimensionless"].float()
    mass_action=w_dimensionless[:,None]*phi
    weak_action=stiffness+mass_action*potential[:,None]
    stiffness_diagonal=torch.zeros_like(w_dimensionless)
    stiffness_diagonal[:,:-1]+=coefficient; stiffness_diagonal[:,1:]+=coefficient
    free_mask=batch["node_mask"]&~batch["boundary_mask"]&batch["potential_valid_mask"]
    return phi,weak_action,mass_action,stiffness_diagonal,free_mask


def dimensionless_projected_hamiltonian(trials:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    with torch.autocast(device_type=trials.device.type,enabled=False):
        phi,weak_action,_,_,_=dimensionless_weak_hamiltonian_action(trials.float(),batch)
        projected=torch.einsum("bkn,bjn->bkj",phi,weak_action)
        return .5*(projected+projected.transpose(-1,-2))


def dimensionless_rayleigh_ritz(orthogonal_trials:torch.Tensor,batch:Mapping[str,torch.Tensor],detach_vectors:bool=True)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Diagonalize the symmetric weak-form Hamiltonian in a weighted-orthogonal subspace."""
    with torch.autocast(device_type=orthogonal_trials.device.type,enabled=False):
        q=orthogonal_trials.float(); projected=dimensionless_projected_hamiltonian(q,batch)
        epsilon,rotation,relative_min_gap,safe_gradient=stable_eigh_rotation(projected,detach_vectors=detach_vectors)
        psi=rotation.transpose(-1,-2).to(q.dtype)@q
        psi=psi*batch["node_mask"][:,None]; psi=psi.masked_fill(batch["boundary_mask"][:,None],0.0)
    return psi.to(orthogonal_trials.dtype),epsilon.to(orthogonal_trials.dtype),projected,relative_min_gap,safe_gradient


def training_hamiltonian_action(psi:torch.Tensor,x:torch.Tensor,V:torch.Tensor,kappa:torch.Tensor,
                                node_mask:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor]:
    hm=x[:,1:-1]-x[:,:-2]; hp=x[:,2:]-x[:,1:-1]
    if node_mask is None: interior=torch.ones_like(hm,dtype=torch.bool)
    else: interior=node_mask[:,:-2]&node_mask[:,1:-1]&node_mask[:,2:]
    hm_safe=torch.where(interior,hm,torch.ones_like(hm)).clamp_min(1e-8); hp_safe=torch.where(interior,hp,torch.ones_like(hp)).clamp_min(1e-8)
    d2=2*((psi[:,:,2:]-psi[:,:,1:-1])/hp_safe[:,None]-(psi[:,:,1:-1]-psi[:,:,:-2])/hm_safe[:,None])/(hm_safe+hp_safe)[:,None]
    action=-kappa[:,None,None]*d2+V[:,None,1:-1]*psi[:,:,1:-1]
    return torch.where(interior[:,None],action,torch.zeros_like(action)),interior


def training_rayleigh(psi:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    Hpsi,interior=training_hamiltonian_action(psi.float(),batch["x"].float(),batch["V"].float(),batch["kappa"].float(),batch["node_mask"])
    wi=batch["w"][:,1:-1].float()*interior; core=psi[:,:,1:-1].float()
    return (wi[:,None]*core*Hpsi).sum(-1)/(wi[:,None]*core.square()).sum(-1).clamp_min(1e-8)


def dimensionless_weak_rayleigh(psi:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
    """Galerkin-consistent dimensionless Rayleigh quotient for learned wavefunctions."""
    device_type=psi.device.type if psi.device.type in ("cpu","cuda") else "cpu"
    with torch.autocast(device_type=device_type,enabled=False):
        phi,weak_action,mass_action,_,_=dimensionless_weak_hamiltonian_action(psi.float(),batch)
        numerator=(phi*weak_action).sum(-1)
        denominator=(phi*mass_action).sum(-1).clamp_min(1e-12)
        quotient=numerator/denominator
    return quotient.to(psi.dtype)


def dimensionless_weak_residual_correction(psi:torch.Tensor,epsilon:torch.Tensor,batch:Mapping[str,torch.Tensor])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Return per-state relative residuals, an analytic Jacobi correction, and weak residual.

    The positive denominator is deliberately conservative: the learned component may
    modulate this correction but cannot reverse it or overwrite the wavefunction.
    """
    with torch.autocast(device_type=psi.device.type,enabled=False):
        phi,weak_action,mass_action,stiffness_diagonal,free_mask=dimensionless_weak_hamiltonian_action(psi.float(),batch)
        weak_residual=(weak_action-epsilon.float()[:,:,None]*mass_action)*free_mask[:,None]
        w=batch["w_dimensionless"].float(); safe_w=w.clamp_min(1e-10)
        strong_residual=weak_residual/safe_w[:,None]
        strong_action=(weak_action*free_mask[:,None])/safe_w[:,None]
        numerator=torch.sum(w[:,None]*strong_residual.square(),dim=-1)
        denominator=(torch.sum(w[:,None]*strong_action.square(),dim=-1)+
                     torch.sum(w[:,None]*(epsilon.float()[:,:,None]*phi).square(),dim=-1)).clamp_min(1e-10)
        relative_residual=torch.sqrt(numerator/denominator)
        potential=batch["V_dimensionless"].float()
        diagonal=stiffness_diagonal[:,None]+w[:,None]*(torch.abs(potential[:,None]-epsilon.float()[:,:,None])+1.0)
        diagonal_reference=stiffness_diagonal[:,None]+w[:,None]*(potential[:,None].abs()+epsilon.float()[:,:,None].abs()+1.0)
        diagonal=torch.maximum(diagonal,CFG.refinement_preconditioner_floor*diagonal_reference).clamp_min(1e-8)
        correction_phi=-weak_residual/diagonal
        length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).float().clamp_min(1e-8)
        correction=correction_phi/torch.sqrt(length)[:,None,None]
        correction=correction*free_mask[:,None]
    return relative_residual.to(psi.dtype),correction.to(psi.dtype),weak_residual.to(psi.dtype)


def project_weighted_outside(correction:torch.Tensor,psi:torch.Tensor,w:torch.Tensor)->torch.Tensor:
    overlap=torch.einsum("bkn,bn,bjn->bkj",correction,w,psi)
    return correction-overlap.to(correction.dtype)@psi


def accept_refinement_step(current_residual:torch.Tensor,proposal_residual:torch.Tensor,
                           current_epsilon:torch.Tensor,proposal_epsilon:torch.Tensor,
                           tolerance:float=CFG.refinement_acceptance_tolerance)->torch.Tensor:
    """Detached per-Hamiltonian trust decision for a proposed block update."""
    current_block=torch.sqrt(current_residual.square().mean(-1).clamp_min(0))
    proposal_block=torch.sqrt(proposal_residual.square().mean(-1).clamp_min(0))
    current_ky_fan=current_epsilon.sum(-1); proposal_ky_fan=proposal_epsilon.sum(-1)
    finite=(torch.isfinite(proposal_residual).all(-1)&torch.isfinite(proposal_epsilon).all(-1)&
            torch.isfinite(proposal_block)&torch.isfinite(proposal_ky_fan))
    residual_safe=proposal_block<=current_block*(1+tolerance)+1e-8
    variational_safe=proposal_ky_fan<=current_ky_fan+tolerance*(1+current_ky_fan.abs())
    return (finite&residual_safe&variational_safe).detach()


def select_finite_refinement_proposal(best_residual:torch.Tensor,best_block:torch.Tensor,
                                      proposal_residual:torch.Tensor,candidate_finite:torch.Tensor
                                      )->tuple[torch.Tensor,torch.Tensor]:
    """Keep the lowest-residual finite proposal without leaking NaNs into its trace."""
    proposal_block=torch.sqrt(proposal_residual.detach().square().mean(-1).clamp_min(0))
    finite=candidate_finite&torch.isfinite(proposal_residual).all(-1)&torch.isfinite(proposal_block)
    safe_proposal=torch.nan_to_num(proposal_residual,nan=0.0,posinf=0.0,neginf=0.0)
    candidate=torch.where(finite[:,None],safe_proposal,best_residual)
    better=finite&(proposal_block<best_block)
    return torch.where(better[:,None],candidate,best_residual),torch.where(better,proposal_block,best_block)


class ResidualCorrectionBlock(nn.Module):
    """Weight-tied, sign-equivariant positive modulation of the analytic correction."""
    def __init__(self,hidden:int):
        super().__init__(); scalar_features=6
        self.network=nn.Sequential(nn.LayerNorm(2*hidden+scalar_features),nn.Linear(2*hidden+scalar_features,hidden),
                                   nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
    def forward(self,context:torch.Tensor,tokens:torch.Tensor,psi:torch.Tensor,correction:torch.Tensor,
                weak_residual:torch.Tensor,epsilon:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
        B,K,N=psi.shape; length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8)
        root_length=torch.sqrt(length)[:,None,None]; phi=psi*root_length; correction_phi=correction*root_length
        free_mask=(batch["node_mask"]&batch["potential_valid_mask"]&~batch["boundary_mask"]).to(weak_residual.dtype)
        residual_scale=((weak_residual.abs()*free_mask[:,None]).sum(-1,keepdim=True)/
                        free_mask.sum(-1).clamp_min(1)[:,None,None]).clamp_min(1e-8)
        scalars=torch.stack([torch.tanh(phi.abs()),torch.tanh(correction_phi.abs()),
                             torch.tanh(weak_residual.abs()/residual_scale),
                             torch.tanh(batch["V_dimensionless"][:,None].expand(B,K,N)/max(TRAIN_U_SCALE,1e-8)),
                             batch["t"][:,None].expand(B,K,N),torch.tanh(epsilon)[:,:,None].expand(B,K,N)],dim=-1)
        node_context=context[:,None].expand(B,K,N,-1); state_context=tokens[:,:,None].expand(B,K,N,-1)
        logit=self.network(torch.cat([node_context,state_context,scalars],dim=-1)).squeeze(-1)
        return 1+CFG.refinement_learned_multiplier*torch.tanh(logit)


OPERATOR_BATCH_KEYS=("features","x","t","w","w_dimensionless","V","V_relative","V_dimensionless",
                     "V_offset","V_gauge","energy_unit","node_mask","potential_valid_mask","singularity_mask",
                     "boundary_mask","geometry","bc","endpoints","kappa","lengths","edge_index","edge_attr","batch_index")
FORBIDDEN_DIRECT_KEYS=("psi","rho","energy","energy_relative","energy_dimensionless","state_mask",
                       "target_node_positions","target_node_mask","target_phase_cdf","target_phase_cdf_mask",
                       "target_log_amplitude","target_log_amplitude_mask","z_target","target_latent")


def operator_batch_view(batch:Mapping[str,Any])->dict[str,Any]:
    """A capability-limited view passed to every deployable module."""
    missing=[key for key in OPERATOR_BATCH_KEYS if key not in batch]
    if missing: raise KeyError(f"operator batch is missing {missing}")
    return {key:batch[key] for key in OPERATOR_BATCH_KEYS}


@torch.no_grad()
def detached_topology_diagnostics(psi:torch.Tensor,batch:Mapping[str,torch.Tensor],
                                  minimum_intervals:int=CFG.minimum_intervals_per_lobe,
                                  minimum_lobe_mass:float=CFG.topology_min_lobe_mass)->dict[str,torch.Tensor]:
    """Robust detached sampled-node/lobe checks used only for hybrid acceptance."""
    psi_cpu=psi.detach().float().cpu().numpy(); x_cpu=batch["x"].detach().float().cpu().numpy()
    w_cpu=batch["w"].detach().float().cpu().numpy(); lengths=batch["lengths"].detach().cpu().tolist()
    B,K,_=psi_cpu.shape; counts=np.zeros((B,K),dtype=np.int64); min_intervals=np.zeros((B,K),dtype=np.int64)
    min_mass=np.zeros((B,K),dtype=np.float64); valid=np.ones((B,K),dtype=bool)
    for b in range(B):
        n_valid=int(lengths[b]); x=x_cpu[b,:n_valid]
        for state in range(K):
            y=canonical_global_sign(psi_cpu[b,state,:n_valid]); nodes=node_positions(x,y)
            counts[b,state]=len(nodes)
            boundaries=np.concatenate(([x[0]],nodes,[x[-1]]))
            midpoints=.5*(x[:-1]+x[1:])
            intervals=np.asarray([np.count_nonzero((midpoints>=boundaries[lobe])&(midpoints<boundaries[lobe+1]))
                                  for lobe in range(len(boundaries)-1)],dtype=np.int64)
            min_intervals[b,state]=int(intervals.min()) if len(intervals) else n_valid-1
            masses=[]
            for lobe in range(len(boundaries)-1):
                left,right=boundaries[lobe:lobe+2]; interior=x[(x>left)&(x<right)]
                points=np.r_[left,interior,right]
                # Interpolate the signed waveform first so a bracketed root contributes
                # zero density at the lobe boundary; interpolating y**2 would spuriously
                # assign positive mass to that root.
                lobe_wave=np.interp(points,x,y)
                masses.append(trapezoid_integral_1d(lobe_wave**2,points))
            min_mass[b,state]=min(masses) if masses else 0.0
            valid[b,state]=(counts[b,state]==state and min_intervals[b,state]>=minimum_intervals and
                            min_mass[b,state]>=minimum_lobe_mass and np.isfinite(y).all())
    device=psi.device
    return {"valid_by_state":torch.as_tensor(valid,device=device),
            "valid":torch.as_tensor(valid.all(axis=1),device=device),
            "persistent_node_count":torch.as_tensor(counts,device=device),
            "minimum_intervals_per_lobe":torch.as_tensor(min_intervals,device=device),
            "minimum_lobe_mass":torch.as_tensor(min_mass,device=device,dtype=psi.dtype)}


def resolve_topology_hybrid(psi_direct:torch.Tensor,epsilon_direct:torch.Tensor,
                            psi_candidate:torch.Tensor,epsilon_candidate:torch.Tensor,
                            orthogonalization_valid:torch.Tensor,initial_topology_valid:torch.Tensor,
                            candidate_topology_valid:torch.Tensor,all_proposals_topology_invalid:torch.Tensor,
                            safeguard:bool=True)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    finite=torch.isfinite(psi_candidate).flatten(1).all(-1)&torch.isfinite(epsilon_candidate).all(-1)
    fallback=(~finite)
    if safeguard:
        fallback=fallback|(~orthogonalization_valid)|(~initial_topology_valid)|(~candidate_topology_valid)|all_proposals_topology_invalid
    return (torch.where(fallback[:,None,None],psi_direct,torch.nan_to_num(psi_candidate)),
            torch.where(fallback[:,None],epsilon_direct,torch.nan_to_num(epsilon_candidate)),fallback)


def physics_route_score(psi:torch.Tensor,epsilon:torch.Tensor,batch:Mapping[str,torch.Tensor],
                        topology_valid:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Target-free Hamiltonian-level score for selecting an already computed route."""
    residual,_,_=dimensionless_weak_residual_correction(psi,epsilon,batch)
    residual_block=torch.sqrt(residual.detach().square().mean(-1).clamp_min(0))
    gram=torch.einsum("bkn,bn,bjn->bkj",psi,batch["w"],psi)
    identity=torch.eye(gram.shape[-1],device=gram.device,dtype=gram.dtype)[None]
    gram_rms=torch.sqrt((gram-identity).detach().square().mean(dim=(-2,-1)).clamp_min(0))
    score=residual_block+CFG.route_selection_gram_weight*gram_rms
    finite=torch.isfinite(psi).flatten(1).all(-1)&torch.isfinite(epsilon).all(-1)&torch.isfinite(score)
    eligible=finite&topology_valid
    return torch.where(eligible,score,torch.full_like(score,float("inf"))),residual_block,gram_rms


def select_physics_route(route_psi:Sequence[torch.Tensor],route_epsilon:Sequence[torch.Tensor],
                         route_topology_valid:Sequence[torch.Tensor],batch:Mapping[str,torch.Tensor]
                         )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    scored=[physics_route_score(psi,epsilon,batch,valid) for psi,epsilon,valid in
            zip(route_psi,route_epsilon,route_topology_valid)]
    scores=torch.stack([item[0] for item in scored],dim=1)
    offsets=torch.arange(scores.shape[1],device=scores.device,dtype=scores.dtype)*CFG.route_selection_tolerance
    selected=(scores+offsets[None]).argmin(1)
    batch_index=torch.arange(len(selected),device=selected.device)
    psi=torch.stack(list(route_psi),dim=1)[batch_index,selected]
    epsilon=torch.stack(list(route_epsilon),dim=1)[batch_index,selected]
    residuals=torch.stack([item[1] for item in scored],dim=1)
    grams=torch.stack([item[2] for item in scored],dim=1)
    return psi,epsilon,selected,scores,residuals,grams


class TrueSchrodingerJEPA(nn.Module):
    def __init__(self,backbone:str=CFG.backbone,in_features:int=sample_batch["features"].shape[-1],use_global_context:bool=True,
                 refinement_enabled:bool=CFG.refinement_enabled,learned_refinement:bool=CFG.refinement_learned_correction,
                 decoder_type:str=CFG.decoder_type,primary_output:str=CFG.primary_output,
                 topology_safeguard:bool=CFG.topology_safeguard):
        super().__init__(); hidden=CFG.preset.hidden; use_kan=backbone=="gnn_kan"
        if decoder_type not in DECODER_OPTIONS: raise ValueError(f"unknown decoder {decoder_type}")
        if primary_output not in PRIMARY_OUTPUT_OPTIONS: raise ValueError(f"unknown primary output {primary_output}")
        if decoder_type!="topology_phase" and primary_output=="direct_topology": primary_output="physics_selected"
        self.refinement_enabled=bool(refinement_enabled); self.learned_refinement=bool(learned_refinement)
        self.decoder_type=decoder_type; self.primary_output=primary_output; self.topology_safeguard=bool(topology_safeguard)
        self.potential_context_encoder=make_context_encoder(backbone,in_features,hidden,use_global_context)
        self.solution_online_encoder=SolutionEncoder(hidden,use_kan=use_kan)
        self.solution_target_encoder=initialize_ema_target(self.solution_online_encoder)
        self.solution_reconstruction_decoder=SolutionReconstructionDecoder(hidden)
        self.predictor=StateConditionedPredictor(hidden,use_kan,use_global_context)
        self.wavefunction_decoder=(TopologyPhaseDecoder(hidden,use_kan) if decoder_type=="topology_phase"
                                   else SpectralSineResidualDecoder(hidden,use_kan))
        self.energy_head=OperatorScaledGroundLogGapHead(hidden,use_kan)
        self.residual_correction=ResidualCorrectionBlock(hidden)
        if not self.learned_refinement: self.residual_correction.requires_grad_(False)

    def _refinement_depth(self,override:int|None)->int:
        if override is not None:
            if int(override)<0: raise ValueError("refinement_steps must be non-negative")
            return int(override)
        if not self.refinement_enabled: return 0
        return CFG.preset.refinement_train_steps if self.training else CFG.preset.refinement_inference_steps

    def _exploration_direction(self,psi:torch.Tensor,batch:Mapping[str,torch.Tensor])->torch.Tensor:
        length=(batch["endpoints"][:,1]-batch["endpoints"][:,0]).clamp_min(1e-8)
        carrier=normalized_sine_basis(batch,CFG.k_states).transpose(1,2)/torch.sqrt(length)[:,None,None]
        orientation=torch.sum(batch["w"][:,None]*psi*carrier,dim=-1,keepdim=True)
        sign=torch.where(orientation.detach()>=0,torch.ones_like(orientation),-torch.ones_like(orientation))
        return carrier*sign

    def _refine_eigenspace(self,psi:torch.Tensor,epsilon:torch.Tensor,context:torch.Tensor,tokens:torch.Tensor,
                           batch:Mapping[str,torch.Tensor],steps:int,detach_ritz_vectors:bool,
                           relative_min_gap:torch.Tensor,safe_gradient:torch.Tensor,
                           topology_required:bool=False)->dict[str,torch.Tensor]:
        initial_psi=psi; initial_epsilon=epsilon
        current_residual,_,_=dimensionless_weak_residual_correction(psi,epsilon,batch)
        psi_trace=[psi.detach()]; epsilon_trace=[epsilon.detach()]; residual_trace=[current_residual]
        proposal_residual_trace=[]; acceptance_trace=[]; step_size_trace=[]; attempted_trace=[]; topology_rejection_trace=[]; topology_valid_candidate_trace=[]
        topology_proposal_attempt_count=[]; topology_proposal_rejection_count=[]
        stagnation=torch.zeros(psi.shape[0],device=psi.device,dtype=torch.long)
        alive=torch.sqrt(current_residual.detach().square().mean(-1))>CFG.refinement_stop_residual
        current_gap=relative_min_gap; current_safe=safe_gradient
        for _ in range(steps):
            attempted=alive.clone(); attempted_trace.append(attempted)
            old_residual=current_residual
            old_block=torch.sqrt(old_residual.detach().square().mean(-1).clamp_min(0))
            _,analytic_correction,weak_residual=dimensionless_weak_residual_correction(psi,epsilon,batch)
            if self.learned_refinement:
                modulation=self.residual_correction(context,tokens,psi,analytic_correction,weak_residual,epsilon,batch)
            else:
                modulation=torch.ones_like(analytic_correction)
            correction=analytic_correction*modulation
            if CFG.refinement_exploration_scale>0:
                correction=correction+CFG.refinement_exploration_scale*self._exploration_direction(psi,batch)
            correction=project_weighted_outside(correction,psi,batch["w"])
            correction=correction*batch["node_mask"][:,None]
            correction=correction.masked_fill(batch["boundary_mask"][:,None],0.0)
            correction=torch.nan_to_num(correction,nan=0.0,posinf=0.0,neginf=0.0)
            correction_norm=torch.sqrt(torch.sum(batch["w"][:,None]*correction.square(),dim=-1,keepdim=True).clamp_min(1e-12))
            trust_scale=torch.clamp(CFG.refinement_max_correction_norm/correction_norm,max=1.0)
            correction=correction*trust_scale
            chosen_psi=psi; chosen_epsilon=epsilon; chosen_residual=current_residual
            chosen_gap=current_gap; chosen_safe=current_safe
            accepted=torch.zeros_like(alive); chosen_step=torch.zeros_like(epsilon[:,0])
            topology_rejected=torch.zeros_like(alive); topology_valid_candidate=torch.zeros_like(alive)
            proposal_attempt_count=torch.zeros_like(stagnation); proposal_rejection_count=torch.zeros_like(stagnation)
            best_proposal=old_residual; best_proposal_block=torch.full_like(old_block,float("inf"))
            for backtrack in range(CFG.refinement_backtrack_steps):
                step_size=CFG.refinement_damping/(2**backtrack)
                trial=psi+step_size*correction
                trial=trial*batch["node_mask"][:,None]; trial=trial.masked_fill(batch["boundary_mask"][:,None],0.0)
                if topology_required:
                    orthogonal_trial,cholesky_ok=safe_weighted_cholesky_orthonormalize_torch(trial,batch["w"])
                else:
                    orthogonal_trial=weighted_cholesky_orthonormalize_torch(trial,batch["w"]); cholesky_ok=torch.ones(psi.shape[0],device=psi.device,dtype=torch.bool)
                proposal_psi,proposal_epsilon,_,proposal_gap,proposal_safe=dimensionless_rayleigh_ritz(
                    orthogonal_trial,batch,detach_vectors=detach_ritz_vectors)
                proposal_residual,_,_=dimensionless_weak_residual_correction(proposal_psi,proposal_epsilon,batch)
                candidate_finite=(cholesky_ok&torch.isfinite(proposal_psi).flatten(1).all(-1)&torch.isfinite(proposal_epsilon).all(-1)&
                                  torch.isfinite(proposal_residual).all(-1)&torch.isfinite(proposal_gap))
                topology_ok=(detached_topology_diagnostics(proposal_psi,batch)["valid"] if topology_required
                             else torch.ones_like(candidate_finite))
                active_proposal=(~accepted)&alive&candidate_finite
                proposal_attempt_count=proposal_attempt_count+active_proposal.long()
                proposal_rejection_count=proposal_rejection_count+(active_proposal&~topology_ok).long()
                topology_rejected=topology_rejected|(active_proposal&~topology_ok)
                topology_valid_candidate=topology_valid_candidate|(active_proposal&topology_ok)
                best_proposal,best_proposal_block=select_finite_refinement_proposal(
                    best_proposal,best_proposal_block,proposal_residual,candidate_finite)
                take=((~accepted)&alive&candidate_finite&topology_ok&
                      accept_refinement_step(current_residual,proposal_residual,epsilon,proposal_epsilon))
                safe_psi=torch.nan_to_num(proposal_psi,nan=0.0,posinf=0.0,neginf=0.0)
                safe_epsilon=torch.nan_to_num(proposal_epsilon,nan=0.0,posinf=0.0,neginf=0.0)
                safe_residual=torch.nan_to_num(proposal_residual,nan=0.0,posinf=0.0,neginf=0.0)
                safe_gap=torch.nan_to_num(proposal_gap,nan=0.0,posinf=0.0,neginf=0.0)
                chosen_psi=torch.where(take[:,None,None],safe_psi,chosen_psi)
                chosen_epsilon=torch.where(take[:,None],safe_epsilon,chosen_epsilon)
                chosen_residual=torch.where(take[:,None],safe_residual,chosen_residual)
                chosen_gap=torch.where(take,safe_gap,chosen_gap); chosen_safe=torch.where(take,proposal_safe,chosen_safe)
                chosen_step=torch.where(take,torch.full_like(chosen_step,step_size),chosen_step); accepted=accepted|take
            psi,epsilon,current_residual=chosen_psi,chosen_epsilon,chosen_residual
            current_gap,current_safe=chosen_gap,chosen_safe
            new_block=torch.sqrt(current_residual.detach().square().mean(-1).clamp_min(0))
            relative_improvement=(old_block-new_block)/old_block.clamp_min(1e-8)
            productive=accepted&(relative_improvement>CFG.refinement_stagnation_tolerance)
            stagnation=torch.where(~attempted,stagnation,torch.where(productive,torch.zeros_like(stagnation),stagnation+1))
            alive=(new_block>CFG.refinement_stop_residual)&(stagnation<CFG.refinement_stagnation_patience)
            psi_trace.append(psi.detach()); epsilon_trace.append(epsilon.detach()); residual_trace.append(current_residual)
            proposal_residual_trace.append(best_proposal); acceptance_trace.append(accepted); step_size_trace.append(chosen_step)
            topology_rejection_trace.append(topology_rejected)
            topology_valid_candidate_trace.append(topology_valid_candidate)
            topology_proposal_attempt_count.append(proposal_attempt_count)
            topology_proposal_rejection_count.append(proposal_rejection_count)
            if not self.training and not bool(alive.any()): break
        B,K,N=psi.shape; executed=len(acceptance_trace)
        empty_state=psi.new_empty((B,0,K)); empty_scalar=psi.new_empty((B,0)); empty_bool=torch.empty((B,0),device=psi.device,dtype=torch.bool)
        projected=dimensionless_projected_hamiltonian(psi,batch)
        iterations_used=(torch.stack(attempted_trace,dim=1).sum(1) if attempted_trace else torch.zeros(B,device=psi.device,dtype=torch.long))
        final_block=torch.sqrt(current_residual.detach().square().mean(-1)); converged=final_block<=CFG.refinement_stop_residual
        return {"psi":psi,"epsilon":epsilon,"projected_hamiltonian":projected,"ritz_relative_min_gap":current_gap,"ritz_gradient_safe":current_safe,
                "psi_initial":initial_psi,"energy_dimensionless_initial":initial_epsilon,
                "refinement_psi_trace":torch.stack(psi_trace,dim=1),
                "refinement_energy_dimensionless_trace":torch.stack(epsilon_trace,dim=1),
                "refinement_residual_trace":torch.stack(residual_trace,dim=1),
                "refinement_proposal_residual_trace":torch.stack(proposal_residual_trace,dim=1) if executed else empty_state,
                "refinement_acceptance_trace":torch.stack(acceptance_trace,dim=1) if executed else empty_bool,
                "refinement_step_size_trace":torch.stack(step_size_trace,dim=1) if executed else empty_scalar,
                "refinement_attempted_trace":torch.stack(attempted_trace,dim=1) if executed else empty_bool,
                "refinement_topology_rejection_trace":torch.stack(topology_rejection_trace,dim=1) if executed else empty_bool,
                "refinement_topology_valid_candidate_trace":torch.stack(topology_valid_candidate_trace,dim=1) if executed else empty_bool,
                "refinement_topology_proposal_attempt_count":torch.stack(topology_proposal_attempt_count,dim=1) if executed else torch.empty((B,0),device=psi.device,dtype=torch.long),
                "refinement_topology_proposal_rejection_count":torch.stack(topology_proposal_rejection_count,dim=1) if executed else torch.empty((B,0),device=psi.device,dtype=torch.long),
                "refinement_iterations_used":iterations_used,
                "refinement_converged":converged,
                "refinement_stagnated":(~alive)&~converged,
                "refinement_depth_exhausted":alive&~converged,
                "refinement_fallback_recommended":~converged}

    def forward_operator(self,batch:Mapping[str,torch.Tensor],detach_ritz_vectors:bool=True,refinement_steps:int|None=None)->dict[str,torch.Tensor]:
        # Deployable modules receive a capability-limited mapping.  The training batch
        # may contain targets, but those objects do not cross this boundary.
        op=operator_batch_view(batch)
        context=self.potential_context_encoder(op); z_nodes,z_pred,tokens,global_context=self.predictor(context,op)
        energy_fields=self.energy_head(tokens,global_context,op); head_epsilon=energy_fields["epsilon"]
        direct_fields={}
        if self.decoder_type=="topology_phase":
            direct_fields=self.wavefunction_decoder(z_nodes,op); pre=direct_fields["psi_direct"]
            psi_direct=pre; rho_direct=direct_fields["rho_direct"]
            orthogonal,orthogonalization_valid=safe_weighted_cholesky_orthonormalize_torch(psi_direct,op["w"])
            orthogonal=orthogonal.masked_fill(op["boundary_mask"][:,None],0.0)*op["node_mask"][:,None]
        else:
            pre,orthogonal=self.wavefunction_decoder(z_pred,op,CFG.orthonormalization!="loss_only")
            psi_direct=pre; rho_direct=pre.square(); orthogonalization_valid=torch.ones(pre.shape[0],device=pre.device,dtype=torch.bool)

        energy_direct=op["V_gauge"][:,None]+op["energy_unit"][:,None]*head_epsilon
        rayleigh_direct=training_rayleigh(psi_direct,op)
        weak_rayleigh_direct_dimensionless=dimensionless_weak_rayleigh(psi_direct,op)
        weak_rayleigh_direct=op["V_gauge"][:,None]+op["energy_unit"][:,None]*weak_rayleigh_direct_dimensionless
        if CFG.orthonormalization=="rayleigh_ritz":
            psi_ritz,epsilon_ritz,projected_initial,ritz_relative_min_gap,ritz_gradient_safe=dimensionless_rayleigh_ritz(
                orthogonal,op,detach_vectors=detach_ritz_vectors)
            initial_topology=(detached_topology_diagnostics(psi_ritz,op) if self.decoder_type=="topology_phase"
                              else {"valid":torch.ones(psi_ritz.shape[0],device=psi_ritz.device,dtype=torch.bool),
                                    "valid_by_state":torch.ones(psi_ritz.shape[:2],device=psi_ritz.device,dtype=torch.bool)})
            refinement=self._refine_eigenspace(psi_ritz,epsilon_ritz,context,tokens,op,self._refinement_depth(refinement_steps),
                                               detach_ritz_vectors,ritz_relative_min_gap,ritz_gradient_safe,
                                               topology_required=self.decoder_type=="topology_phase" and self.topology_safeguard)
            candidate_psi=refinement["psi"]; candidate_epsilon=refinement["epsilon"]
            projected_hamiltonian=refinement["projected_hamiltonian"]
            ritz_relative_min_gap=refinement["ritz_relative_min_gap"]; ritz_gradient_safe=refinement["ritz_gradient_safe"]
        else:
            psi_ritz=orthogonal; epsilon_ritz=head_epsilon; projected_initial=torch.empty(0,device=orthogonal.device)
            ritz_relative_min_gap=torch.full((orthogonal.shape[0],),float("inf"),device=orthogonal.device)
            ritz_gradient_safe=torch.zeros(orthogonal.shape[0],dtype=torch.bool,device=orthogonal.device)
            initial_topology=(detached_topology_diagnostics(psi_ritz,op) if self.decoder_type=="topology_phase"
                              else {"valid":torch.ones(orthogonal.shape[0],device=orthogonal.device,dtype=torch.bool),
                                    "valid_by_state":torch.ones(orthogonal.shape[:2],device=orthogonal.device,dtype=torch.bool)})
            refinement=self._refine_eigenspace(psi_ritz,epsilon_ritz,context,tokens,op,0,True,
                                               ritz_relative_min_gap,ritz_gradient_safe,False)
            candidate_psi=refinement["psi"]; candidate_epsilon=refinement["epsilon"]
            projected_hamiltonian=refinement["projected_hamiltonian"]

        if self.decoder_type=="topology_phase":
            candidate_topology=detached_topology_diagnostics(candidate_psi,op)
            topology_rejections=refinement["refinement_topology_rejection_trace"]
            attempted=refinement["refinement_attempted_trace"]
            all_proposals_topology_invalid=(attempted.any(1)&
                                            ~refinement["refinement_topology_valid_candidate_trace"].any(1)) if attempted.shape[1] else torch.zeros_like(initial_topology["valid"])
            psi_hybrid,epsilon_hybrid,fallback=resolve_topology_hybrid(
                psi_direct,head_epsilon,candidate_psi,candidate_epsilon,orthogonalization_valid,
                initial_topology["valid"],candidate_topology["valid"],all_proposals_topology_invalid,self.topology_safeguard)
            hybrid_topology=detached_topology_diagnostics(psi_hybrid,op)
        else:
            candidate_topology=initial_topology; all_proposals_topology_invalid=torch.zeros(psi_direct.shape[0],device=psi_direct.device,dtype=torch.bool)
            fallback=torch.zeros_like(all_proposals_topology_invalid); psi_hybrid=candidate_psi; epsilon_hybrid=candidate_epsilon; hybrid_topology=initial_topology

        energy_hybrid=op["V_gauge"][:,None]+op["energy_unit"][:,None]*epsilon_hybrid
        rayleigh_hybrid=training_rayleigh(psi_hybrid,op)
        weak_rayleigh_hybrid_dimensionless=dimensionless_weak_rayleigh(psi_hybrid,op)
        weak_rayleigh_hybrid=op["V_gauge"][:,None]+op["energy_unit"][:,None]*weak_rayleigh_hybrid_dimensionless
        direct_topology_diagnostics=(detached_topology_diagnostics(psi_direct,op) if self.decoder_type=="topology_phase"
                                     else {"valid":torch.ones(psi_direct.shape[0],device=psi_direct.device,dtype=torch.bool),
                                           "valid_by_state":torch.ones(psi_direct.shape[:2],device=psi_direct.device,dtype=torch.bool)})
        direct_topology=direct_topology_diagnostics["valid"]
        psi_selected,epsilon_selected,selected_route,route_scores,route_residuals,route_grams=select_physics_route(
            (psi_direct,psi_ritz,candidate_psi),(head_epsilon,epsilon_ritz,candidate_epsilon),
            (direct_topology,initial_topology["valid"],candidate_topology["valid"]),op)
        route_topology=torch.stack((direct_topology,initial_topology["valid"],candidate_topology["valid"]),dim=1)
        selected_topology_valid=route_topology[torch.arange(len(selected_route),device=selected_route.device),selected_route]
        energy_selected=op["V_gauge"][:,None]+op["energy_unit"][:,None]*epsilon_selected
        rayleigh_selected=training_rayleigh(psi_selected,op)
        weak_rayleigh_selected_dimensionless=dimensionless_weak_rayleigh(psi_selected,op)
        weak_rayleigh_selected=op["V_gauge"][:,None]+op["energy_unit"][:,None]*weak_rayleigh_selected_dimensionless
        choose_direct=self.decoder_type=="topology_phase" and self.primary_output=="direct_topology"
        choose_selected=self.primary_output=="physics_selected"
        psi=psi_selected if choose_selected else (psi_direct if choose_direct else psi_hybrid)
        epsilon=epsilon_selected if choose_selected else (head_epsilon if choose_direct else epsilon_hybrid)
        energy=energy_selected if choose_selected else (energy_direct if choose_direct else energy_hybrid)
        rq=rayleigh_selected if choose_selected else (rayleigh_direct if choose_direct else rayleigh_hybrid)
        weak_rq_dimensionless=weak_rayleigh_selected_dimensionless if choose_selected else (weak_rayleigh_direct_dimensionless if choose_direct else weak_rayleigh_hybrid_dimensionless)
        weak_rq=weak_rayleigh_selected if choose_selected else (weak_rayleigh_direct if choose_direct else weak_rayleigh_hybrid)
        energy_trace=op["V_gauge"][:,None,None]+op["energy_unit"][:,None,None]*refinement["refinement_energy_dimensionless_trace"]
        out={"context":context,"z_nodes":z_nodes,"z_pred":z_pred,"state_tokens":tokens,"global_context":global_context,
                "decoder_type":self.decoder_type,"primary_output":self.primary_output,
                "psi_pre_orth":pre,"psi_orthogonal":orthogonal,"psi_initial_ritz":psi_ritz,
                "energy_initial_ritz":op["V_gauge"][:,None]+op["energy_unit"][:,None]*epsilon_ritz,
                "psi_direct":psi_direct,"rho_direct":rho_direct,"energy_direct":energy_direct,
                "energy_direct_dimensionless":head_epsilon,"rayleigh_energy_direct":rayleigh_direct,
                "weak_rayleigh_energy_direct":weak_rayleigh_direct,
                "weak_rayleigh_energy_direct_dimensionless":weak_rayleigh_direct_dimensionless,
                "psi_hybrid_candidate":candidate_psi,"energy_hybrid_candidate_dimensionless":candidate_epsilon,
                "psi_hybrid":psi_hybrid,"rho_hybrid":psi_hybrid.square(),
                "energy_hybrid":energy_hybrid,"energy_hybrid_dimensionless":epsilon_hybrid,"rayleigh_energy_hybrid":rayleigh_hybrid,
                "weak_rayleigh_energy_hybrid":weak_rayleigh_hybrid,
                "weak_rayleigh_energy_hybrid_dimensionless":weak_rayleigh_hybrid_dimensionless,
                "hybrid_topology_valid":hybrid_topology["valid"],"hybrid_topology_valid_by_state":hybrid_topology["valid_by_state"],
                "initial_ritz_topology_valid":initial_topology["valid"],"initial_ritz_topology_valid_by_state":initial_topology["valid_by_state"],
                "candidate_hybrid_topology_valid":candidate_topology["valid"],"candidate_hybrid_topology_valid_by_state":candidate_topology["valid_by_state"],
                "hybrid_fallback_recommended":fallback,"used_direct_fallback":fallback,
                "hybrid_orthogonalization_valid":orthogonalization_valid,
                "psi_physics_selected":psi_selected,"energy_physics_selected":energy_selected,
                "energy_physics_selected_dimensionless":epsilon_selected,
                "physics_selected_route":selected_route,"physics_selected_topology_valid":selected_topology_valid,
                "direct_topology_valid":direct_topology,"direct_topology_valid_by_state":direct_topology_diagnostics["valid_by_state"],
                "physics_route_scores":route_scores,
                "physics_route_residuals":route_residuals,"physics_route_gram_rms":route_grams,
                "all_refinement_proposals_topology_invalid":all_proposals_topology_invalid,
                "psi":psi,"rho":psi.square(),"energy":energy,
                "energy_relative":energy-op["V_gauge"][:,None],"energy_dimensionless":epsilon,
                "head_energy":energy_direct,"head_energy_dimensionless":head_epsilon,"projected_hamiltonian":projected_hamiltonian,"rayleigh_energy":rq,
                "weak_rayleigh_energy":weak_rq,"weak_rayleigh_energy_dimensionless":weak_rq_dimensionless,
                "head_ground_energy_dimensionless":energy_fields["ground_energy"],
                "head_gap_dimensionless":energy_fields["gaps"],
                "head_ground_residual_scaled":energy_fields["ground_residual_scaled"],
                "head_log_gap_ratio":energy_fields["log_gap_ratio"],
                "head_raw_ground_residual_scaled":energy_fields["raw_ground_residual_scaled"],
                "head_raw_log_gap_ratio":energy_fields["raw_log_gap_ratio"],
                "energy_reference_dimensionless":energy_fields["reference_energy"],
                "energy_reference_gap_dimensionless":energy_fields["reference_gaps"],
                "energy_operator_floor_dimensionless":energy_fields["operator_floor"],
                "energy_ground_scale_dimensionless":energy_fields["ground_scale"],
                "energy_operator_scale_dimensionless":energy_fields["operator_scale"],
                "energy_reference_projected_hamiltonian":energy_fields["reference_projected_hamiltonian"],
                "ritz_relative_min_gap":ritz_relative_min_gap,"ritz_gradient_safe":ritz_gradient_safe,
                "energy_initial":op["V_gauge"][:,None]+op["energy_unit"][:,None]*refinement["energy_dimensionless_initial"],
                "refinement_energy_trace":energy_trace}
        out.update(direct_fields)
        out.update({key:value for key,value in refinement.items() if key not in ("psi","epsilon","projected_hamiltonian","ritz_relative_min_gap","ritz_gradient_safe")})
        return out
    def encode_online(self,batch:Mapping[str,torch.Tensor])->tuple[torch.Tensor,torch.Tensor]:
        z=self.solution_online_encoder(batch,batch["psi"]); return z,self.solution_reconstruction_decoder(z,batch)
    @torch.no_grad()
    def encode_target(self,batch:Mapping[str,torch.Tensor])->torch.Tensor:
        self.solution_target_encoder.eval(); return self.solution_target_encoder(batch,batch["psi"]).detach()
    def forward_training(self,batch:Mapping[str,torch.Tensor],detach_ritz_vectors:bool=True,refinement_steps:int|None=None)->dict[str,torch.Tensor]:
        out=self.forward_operator(batch,detach_ritz_vectors=detach_ritz_vectors,refinement_steps=refinement_steps); online,reconstruction=self.encode_online(batch); target=self.encode_target(batch)
        out.update({"z_online":online,"z_target":target,"reconstruction":reconstruction}); return out
    @torch.no_grad()
    def update_target(self,tau:float)->None: ema_update(self.solution_online_encoder,self.solution_target_encoder,tau)


model=TrueSchrodingerJEPA().to(DEVICE)
assert model.solution_target_encoder is not model.solution_online_encoder
assert model.solution_target_encoder is not model.potential_context_encoder
decoder_parameter_rows=[]
fork_devices=[DEVICE.index or 0] if DEVICE.type=="cuda" else []
with torch.random.fork_rng(devices=fork_devices):
    for decoder_name in DECODER_OPTIONS:
        candidate=(model if decoder_name==CFG.decoder_type else
                   TrueSchrodingerJEPA(decoder_type=decoder_name,primary_output="physics_selected" if decoder_name!="topology_phase" else "direct_topology"))
        decoder_parameter_rows.append({"decoder_type":decoder_name,
                                       "decoder_parameters":sum(p.numel() for p in candidate.wavefunction_decoder.parameters()),
                                       "total_parameters":sum(p.numel() for p in candidate.parameters()),
                                       "trainable_parameters":sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                                       "parameter_matched":False})
        if candidate is not model: del candidate
decoder_parameter_counts=pd.DataFrame(decoder_parameter_rows)
decoder_parameter_counts["delta_total_vs_spectral"]=decoder_parameter_counts.total_parameters-int(decoder_parameter_counts.loc[decoder_parameter_counts.decoder_type=="spectral_sine_residual","total_parameters"].iloc[0])
decoder_parameter_counts.to_csv(OUT/"decoder_parameter_counts.csv",index=False)
print({"backbone":CFG.backbone,"decoder_type":CFG.decoder_type,"primary_output":CFG.primary_output,
       "parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
       "factorized_KAN_rank":CFG.kan_rank,"parameter_matched":False})
print(decoder_parameter_counts.to_string(index=False))
# Sections 20--22 instantiate the components independently to make their separation
# executable. The combined model owns fresh copies; release the demonstrations before
# training so wide CUDA modes do not retain four unused module graphs.
del solution_online_encoder, solution_reconstruction_decoder, solution_target_encoder, potential_context_encoder
if DEVICE.type=="cuda": torch.cuda.empty_cache()
