import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kl_divergence(mu_1, logvar_1, mu_2, logvar_2):
    """
    Computes analytical KL divergence D_KL( N(mu_1, diag(exp(logvar_1))) || N(mu_2, diag(exp(logvar_2))) )
    Normalized by latent dimension d=512
    """
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    
    kl = 0.5 * torch.mean(
        (logvar_2 - logvar_1) + (var_1 + (mu_1 - mu_2)**2) / (var_2 + 1e-8) - 1.0,
        dim=1
    )
    return kl.mean()

def gaussian_kl_to_std_normal(mu, logvar):
    """
    Computes KL divergence D_KL( N(mu, diag(exp(logvar))) || N(0, I) )
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

def compute_instance_kl_loss(mu_poison, logvar_poison, mu_real, logvar_real):
    """
    Route 1: Batch-Level Minimum KL Matching (Dynamic Instance-Level Alignment)
    L_KL-Instance-Min = (1/B) sum_{i=1}^B min_{j in {1...B}} D_KL( N(mu_p_i, exp(logvar_p_i)) || N(mu_r_j, exp(logvar_r_j)) )
    Computes (B_p, B_r) pairwise KL divergence matrix and dynamically selects closest target instance.
    """
    mu_p = mu_poison.unsqueeze(1)        # [B_p, 1, D]
    logvar_p = logvar_poison.unsqueeze(1)# [B_p, 1, D]
    var_p = torch.exp(logvar_p)          # [B_p, 1, D]
    
    mu_r = mu_real.unsqueeze(0)          # [1, B_r, D]
    logvar_r = logvar_real.unsqueeze(0)  # [1, B_r, D]
    var_r = torch.exp(logvar_r)          # [1, B_r, D]
    
    # Pairwise KL divergence elementwise over latent dimensions [B_p, B_r, D]
    kl_elementwise = 0.5 * (
        (logvar_r - logvar_p) + (var_p + (mu_p - mu_r)**2) / (var_r + 1e-4) - 1.0
    )
    # Pairwise KL matrix [B_p, B_r] normalized by dimension d=512
    kl_matrix = torch.mean(kl_elementwise, dim=-1)
    
    # Select minimum KL divergence match for each poison instance along target dimension (dim=1)
    min_kl, _ = torch.min(kl_matrix, dim=1)
    return min_kl.mean()

def compute_mmd_loss(z_poisoned_batch, z_target_batch, bandwidths=[1.0, 2.0, 4.0, 8.0, 16.0]):
    """
    Route 2: Non-parametric Empirical MMD Alignment Loss using multi-scale RBF kernels and torch.cdist.
    L_MMD = (1/B^2) sum_{i,j} k(z_{p,i}, z_{p,j}) + (1/B^2) sum_{i,j} k(z_{t,i}, z_{t,j}) - (2/B^2) sum_{i,j} k(z_{p,i}, z_{t,j})
    """
    dist_pp = torch.cdist(z_poisoned_batch, z_poisoned_batch, p=2) ** 2
    dist_tt = torch.cdist(z_target_batch, z_target_batch, p=2) ** 2
    dist_pt = torch.cdist(z_poisoned_batch, z_target_batch, p=2) ** 2
    
    k_pp = 0.0
    k_tt = 0.0
    k_pt = 0.0
    
    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * (sigma ** 2))
        k_pp = k_pp + torch.exp(-gamma * dist_pp)
        k_tt = k_tt + torch.exp(-gamma * dist_tt)
        k_pt = k_pt + torch.exp(-gamma * dist_pt)
        
    mmd = k_pp.mean() + k_tt.mean() - 2.0 * k_pt.mean()
    return mmd


def compute_geometric_mask(x, k=15):
    """
    Computes local differential geometric properties for point cloud x.
    Args:
        x: Point cloud, tensor of shape [B, N, 3] or [N, 3]
        k: Number of nearest neighbors
    Returns:
        normals: Tensor of shape [B, N, 3], local normal vectors
        curvature: Tensor of shape [B, N], surface curvature values
        mask: Tensor of shape [B, N], curvature-aware flat weights in [0, 1]
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, N, _ = x.shape
    
    # Compute pairwise distances
    dists = torch.cdist(x, x) # [B, N, N]
    
    # Find k nearest neighbors (including the point itself)
    _, knn_idx = torch.topk(dists, k, largest=False, sorted=True) # [B, N, k]
    
    # Gather neighbors
    batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, k)
    knn_pts = x[batch_indices, knn_idx] # [B, N, k, 3]
    
    # Compute local covariance matrix
    mean = knn_pts.mean(dim=2, keepdim=True) # [B, N, 1, 3]
    centered = knn_pts - mean # [B, N, k, 3]
    cov = torch.matmul(centered.transpose(-1, -2), centered) / k # [B, N, 3, 3]
    
    # Compute eigenvalues using differentiable eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(cov) # [B, N, 3], [B, N, 3, 3]
    
    # Eigenvalues are sorted: lambda1 <= lambda2 <= lambda3
    lambda1 = eigenvalues[..., 0]
    lambda2 = eigenvalues[..., 1]
    lambda3 = eigenvalues[..., 2]
    
    curvature = lambda1 / (lambda1 + lambda2 + lambda3 + 1e-8) # [B, N]
    normals = eigenvectors[..., 0] # Normal is the eigenvector of lambda1. [B, N, 3]
    
    # Map curvature to flat-prior mask: flattest (curvature=0) -> 1, sharpest -> 0
    min_curv = curvature.min(dim=1, keepdim=True)[0]
    max_curv = curvature.max(dim=1, keepdim=True)[0]
    mask = 1.0 - (curvature - min_curv) / (max_curv - min_curv + 1e-8) # [B, N]
    
    return normals, curvature, mask

def project_mask_to_latent(encoder, x, m_point):
    """
    Projects point-level mask to latent space using Jacobian sensitivity.
    Args:
        encoder: VAE encoder model (must support forward returning z_mu, z_sigma)
        x: Input point cloud shape [1, N, 3]
        m_point: Point-level mask of shape [1, N]
    Returns:
        m_latent: Projected latent mask of shape [1, d_latent]
    """
    encoder.eval()
    assert x.shape[0] == 1, "Jacobian projection assumes batch size 1"
    x = x.clone().detach().requires_grad_(True)
    
    # Get latent mean
    z_mu, _ = encoder(x)
    d_latent = z_mu.shape[1]
    
    jacobian = []
    for j in range(d_latent):
        grad_outputs = torch.zeros_like(z_mu)
        grad_outputs[0, j] = 1.0
        # Compute gradient with respect to x
        grad = torch.autograd.grad(z_mu, x, grad_outputs=grad_outputs, retain_graph=True)[0] # [1, N, 3]
        # L2 norm of gradient across coordinate dims
        sensitivity = grad.norm(dim=-1).squeeze(0) # [N]
        jacobian.append(sensitivity)
        
    jacobian = torch.stack(jacobian, dim=1) # [N, d_latent]
    
    # Normalize weights per latent dimension to sum to 1
    weights = jacobian / (jacobian.sum(dim=0, keepdim=True) + 1e-8) # [N, d_latent]
    
    # Project point-level mask
    m_latent = torch.sum(weights * m_point.squeeze(0).unsqueeze(1), dim=0) # [d_latent]
    
    return m_latent.unsqueeze(0) # [1, d_latent]

def chamfer_distance(x, y):
    """ Differentiable unsquared Chamfer Distance in PyTorch """
    B, N, _ = x.shape
    _, M, _ = y.shape
    r_x = torch.sum(x**2, dim=2, keepdim=True) # [B, N, 1]
    r_y = torch.sum(y**2, dim=2, keepdim=True).transpose(1, 2) # [B, 1, M]
    dist = r_x + r_y - 2 * torch.bmm(x, y.transpose(1, 2)) # [B, N, M]
    
    dist1, _ = dist.min(dim=2) # [B, N]
    dist2, _ = dist.min(dim=1) # [B, M]
    
    dist1 = torch.sqrt(torch.clamp(dist1, min=1e-12))
    dist2 = torch.sqrt(torch.clamp(dist2, min=1e-12))
    return (dist1.mean(dim=1) + dist2.mean(dim=1)).mean()

def pgd_latent_optimization(vae, target_pc, source_pc, m_latent, steps=100, lr=0.01, eps=0.2, lambda_cd=1.0):
    """
    Optimizes latent perturbation delta using score loss + predicted x_0 CD loss.
    Args:
        vae: GaussianVAE model
        target_pc: Target point cloud, shape [1, N, 3]
        source_pc: Source point cloud, shape [1, N, 3]
        m_latent: Projected latent mask, shape [1, d_latent]
        steps: Number of PGD optimization steps
        lr: Learning rate for PGD
        eps: L_infinity norm bound
        lambda_cd: Weight for Chamfer Distance loss
    Returns:
        delta_masked: Optimized masked latent perturbation, shape [1, d_latent]
    """
    vae.eval()
    
    # Get clean latent source
    with torch.no_grad():
        z_mu, _ = vae.encoder(source_pc)
        z_source = z_mu.clone()
        
    delta = torch.zeros_like(z_source, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    # Multi-step sampling for score-matching target optimization
    num_steps = vae.diffusion.var_sched.num_steps
    t_steps = [max(1, int(num_steps * r)) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
    
    for step in range(steps):
        optimizer.zero_grad()
        
        delta_masked = m_latent * delta
        z_bd = z_source + delta_masked
        
        loss_total = 0.0
        
        for t in t_steps:
            t_tensor = torch.tensor([t], device=target_pc.device).long()
            alpha_bar = vae.diffusion.var_sched.alpha_bars[t_tensor].view(-1, 1, 1)
            beta = vae.diffusion.var_sched.betas[t_tensor]
            
            c0 = torch.sqrt(alpha_bar)
            c1 = torch.sqrt(1.0 - alpha_bar)
            
            noise = torch.randn_like(target_pc)
            y_t = c0 * target_pc + c1 * noise
            
            # Predict noise
            epsilon_pred = vae.diffusion.net(y_t, beta=beta, context=z_bd)
            
            # 1. Noise/Score prediction loss
            loss_score = F.mse_loss(epsilon_pred, noise)
            
            # 2. Differentiable single-step predicted x_0 CD loss
            x_0_pred = (y_t - c1 * epsilon_pred) / (c0 + 1e-8)
            loss_cd = chamfer_distance(x_0_pred, target_pc)
            
            loss_total += loss_score + lambda_cd * loss_cd
            
        loss_total.backward()
        optimizer.step()
        
        # Enforce L_infinity projection constraint
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            
    with torch.no_grad():
        delta_masked = m_latent * delta
        
    return delta_masked

def compute_distribution_mmd_cd(x_0_pred, target_pcs):
    """
    Differentiable MMD-CD loss between predicted batch x_0_pred [B_src, N, 3]
    and target shape distribution batch target_pcs [B_tgt, N, 3].
    """
    B_src, N, _ = x_0_pred.shape
    B_tgt, M, _ = target_pcs.shape
    
    cd_matrix = []
    for i in range(B_src):
        src_shape = x_0_pred[i:i+1].expand(B_tgt, -1, -1) # [B_tgt, N, 3]
        r_x = torch.sum(src_shape**2, dim=2, keepdim=True) # [B_tgt, N, 1]
        r_y = torch.sum(target_pcs**2, dim=2, keepdim=True).transpose(1, 2) # [B_tgt, 1, M]
        dist = r_x + r_y - 2 * torch.bmm(src_shape, target_pcs.transpose(1, 2))
        
        dist1, _ = dist.min(dim=2) # [B_tgt, N]
        dist2, _ = dist.min(dim=1) # [B_tgt, M]
        dist1 = torch.sqrt(torch.clamp(dist1, min=1e-12))
        dist2 = torch.sqrt(torch.clamp(dist2, min=1e-12))
        cds = dist1.mean(dim=1) + dist2.mean(dim=1) # [B_tgt]
        cd_matrix.append(cds)
        
    cd_matrix = torch.stack(cd_matrix, dim=0) # [B_src, B_tgt]
    min_dist_to_target, _ = cd_matrix.min(dim=1) # [B_src]
    mean_dist_to_target = cd_matrix.mean()
    
    return min_dist_to_target.mean() + 0.5 * mean_dist_to_target

def pgd_distribution_latent_optimization(vae, target_pcs, source_pc, m_latent, steps=100, lr=0.01, eps=0.5, lambda_cd=1.0):
    """
    Distribution-level PGD latent optimization aligning perturbation against a batch of diverse target shapes.
    Args:
        vae: GaussianVAE model
        target_pcs: Target shapes batch from D_target, shape [B_tgt, N, 3]
        source_pc: Source shape, shape [1, N, 3]
        m_latent: Projected latent mask, shape [1, d_latent]
        steps: Number of PGD optimization steps
        lr: Learning rate
        eps: L_infinity bound
        lambda_cd: Weight for Distribution MMD-CD loss
    Returns:
        delta_masked: Masked latent perturbation shape [1, d_latent]
    """
    vae.eval()
    if target_pcs.dim() == 2:
        target_pcs = target_pcs.unsqueeze(0)
    if source_pc.dim() == 2:
        source_pc = source_pc.unsqueeze(0)
        
    device = target_pcs.device
    
    with torch.no_grad():
        z_mu, _ = vae.encoder(source_pc)
        z_source = z_mu.clone()
        
    delta = torch.zeros_like(z_source, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    num_steps = vae.diffusion.var_sched.num_steps
    t_steps = [max(1, int(num_steps * r)) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
    
    B_tgt = target_pcs.size(0)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        delta_masked = m_latent * delta
        z_bd = z_source + delta_masked # [1, d_latent]
        
        loss_total = 0.0
        
        for t in t_steps:
            t_tensor = torch.tensor([t], device=device).long()
            alpha_bar = vae.diffusion.var_sched.alpha_bars[t_tensor].view(-1, 1, 1)
            beta = vae.diffusion.var_sched.betas[t_tensor]
            
            c0 = torch.sqrt(alpha_bar)
            c1 = torch.sqrt(1.0 - alpha_bar)
            
            # Sample random target instance from target_pcs batch for score loss
            rand_idx = torch.randint(0, B_tgt, (1,)).item()
            y_target_single = target_pcs[rand_idx:rand_idx+1] # [1, N, 3]
            
            noise = torch.randn_like(y_target_single)
            y_t = c0 * y_target_single + c1 * noise
            
            epsilon_pred = vae.diffusion.net(y_t, beta=beta, context=z_bd)
            loss_score = F.mse_loss(epsilon_pred, noise)
            
            # Single step x_0 prediction
            x_0_pred = (y_t - c1 * epsilon_pred) / (c0 + 1e-8) # [1, N, 3]
            
            # Distribution-level MMD-CD loss against target batch
            loss_dist = compute_distribution_mmd_cd(x_0_pred, target_pcs)
            
            loss_total += loss_score + lambda_cd * loss_dist
            
        loss_total.backward()
        optimizer.step()
        
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            
    with torch.no_grad():
        delta_masked = m_latent * delta
        
    return delta_masked

def pgd_stochastic_distribution_latent_optimization(vae, target_pcs, source_pc, m_latent, steps=100, lr=0.01, eps=0.5, lambda_cd=1.0, align_loss='kl_instance', logvar_lower_bound=-10.0):
    """
    Optimizes stochastic trigger parameters (delta_mu, delta_logvar) using reparameterization trick
    and specified alignment loss (kl_instance or mmd).
    """
    vae.eval()
    if target_pcs.dim() == 2:
        target_pcs = target_pcs.unsqueeze(0)
    if source_pc.dim() == 2:
        source_pc = source_pc.unsqueeze(0)
        
    device = target_pcs.device
    
    with torch.no_grad():
        z_mu, z_sigma = vae.encoder(source_pc)
        z_source = z_mu.clone()
        
        # Calculate target posterior distribution parameters
        target_mu_all, target_logvar_all = vae.encoder(target_pcs)
        
    delta_mu = torch.zeros_like(z_source, requires_grad=True)
    delta_logvar = torch.full_like(z_source, fill_value=-4.0, requires_grad=True)
    
    optimizer = torch.optim.Adam([delta_mu, delta_logvar], lr=lr)
    
    num_steps = vae.diffusion.var_sched.num_steps
    t_steps = [max(1, int(num_steps * r)) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
    B_tgt = target_pcs.size(0)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        mu_masked = delta_mu
        sigma_masked = torch.exp(0.5 * delta_logvar)
        epsilon = torch.randn_like(mu_masked)
        
        delta_sampled = mu_masked + sigma_masked * epsilon
        z_bd = z_source + delta_sampled # [1, d_latent]
        
        loss_total = 0.0
        
        # Alignment loss computation according to align_loss
        z_poison_mu = z_source + mu_masked
        if align_loss == 'mmd':
            # Route 2: Non-parametric Empirical MMD Alignment
            z_target_sampled = target_mu_all + torch.exp(0.5 * target_logvar_all) * torch.randn_like(target_mu_all)
            loss_align = compute_mmd_loss(z_bd, z_target_sampled)
        else:
            # Route 1: Instance-Level Variational KL Divergence
            loss_align = compute_instance_kl_loss(
                mu_poison=z_poison_mu, logvar_poison=delta_logvar,
                mu_real=target_mu_all, logvar_real=target_logvar_all
            )
            
        # 2. Score loss across timesteps
        for t in t_steps:
            t_tensor = torch.tensor([t], device=device).long()
            alpha_bar = vae.diffusion.var_sched.alpha_bars[t_tensor].view(-1, 1, 1)
            beta = vae.diffusion.var_sched.betas[t_tensor]
            
            c0 = torch.sqrt(alpha_bar)
            c1 = torch.sqrt(1.0 - alpha_bar)
            
            rand_idx = torch.randint(0, B_tgt, (1,)).item()
            y_target_single = target_pcs[rand_idx:rand_idx+1] # [1, N, 3]
            
            noise = torch.randn_like(y_target_single)
            y_t = c0 * y_target_single + c1 * noise
            
            epsilon_pred = vae.diffusion.net(y_t, beta=beta, context=z_bd)
            loss_score = F.mse_loss(epsilon_pred, noise)
            
            x_0_pred = (y_t - c1 * epsilon_pred) / (c0 + 1e-8)
            loss_dist = compute_distribution_mmd_cd(x_0_pred, target_pcs)
            
            loss_total += loss_score + lambda_cd * loss_dist
            
        loss_total += loss_align
        loss_total.backward()
        optimizer.step()
        
        # Gradient safety: clamp data attributes inside no_grad block
        with torch.no_grad():
            delta_mu.data.clamp_(-eps, eps)
            delta_logvar.data.clamp_(logvar_lower_bound, -0.5)
            
    with torch.no_grad():
        mu_masked = delta_mu
        logvar_masked = delta_logvar
        
    return mu_masked, logvar_masked




# -----------------------------------------------------------------------------
# Geometry-mask v2: deterministic single-target experiments
# -----------------------------------------------------------------------------
GEOMETRY_MASK_V2_API = 2


def compute_geometric_mask_v2(x, k=15):
    """Curvature mask with self-neighbours explicitly excluded."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected [B,N,3], got {tuple(x.shape)}")
    batch_size, num_points, _ = x.shape
    if not 1 <= k < num_points:
        raise ValueError(f"k must satisfy 1 <= k < N, got k={k}, N={num_points}")
    distances = torch.cdist(x, x)
    knn_idx = torch.topk(distances, k=k + 1, dim=-1, largest=False).indices[..., 1:]
    batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
    batch_idx = batch_idx.expand(-1, num_points, k)
    neighbours = x[batch_idx, knn_idx]
    centered = neighbours - neighbours.mean(dim=2, keepdim=True)
    covariance = centered.transpose(-1, -2).matmul(centered) / float(k)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    curvature = eigenvalues[..., 0] / (eigenvalues.sum(dim=-1) + 1e-8)
    normals = eigenvectors[..., 0]
    low = curvature.amin(dim=1, keepdim=True)
    high = curvature.amax(dim=1, keepdim=True)
    point_mask = 1.0 - (curvature - low) / (high - low + 1e-8)
    return normals, curvature, point_mask


def encoder_state_sha256(encoder):
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_global_latent_mask(
    encoder,
    reference_loader,
    num_reference_shapes=64,
    knn_k=15,
    aggregation="mean",
    device=None,
    max_examples=8,
):
    """Aggregate per-shape curvature/Jacobian masks into one universal soft mask."""
    if aggregation != "mean":
        raise ValueError("geometry-mask v2 currently supports aggregation=mean only")
    if num_reference_shapes <= 0:
        raise ValueError("num_reference_shapes must be positive")
    device = device or next(encoder.parameters()).device
    encoder.eval()
    original_flags = [parameter.requires_grad for parameter in encoder.parameters()]
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    latent_masks, reference_ids, examples = [], [], []
    processed = 0
    try:
        for batch in reference_loader:
            points = batch["pointcloud"] if isinstance(batch, dict) else batch
            ids = batch.get("id") if isinstance(batch, dict) else None
            points = points.to(device)
            for index in range(points.size(0)):
                if processed >= num_reference_shapes:
                    break
                point = points[index:index + 1]
                normals, curvature, point_mask = compute_geometric_mask_v2(point, k=knn_k)
                latent_mask = project_mask_to_latent(encoder, point, point_mask).detach().cpu()
                latent_masks.append(latent_mask)
                reference_ids.append(int(ids[index]) if ids is not None else processed)
                if len(examples) < max_examples:
                    examples.append({"points": point.detach().cpu(), "curvature": curvature.detach().cpu(), "point_mask": point_mask.detach().cpu(), "normals": normals.detach().cpu()})
                processed += 1
            if processed >= num_reference_shapes:
                break
    finally:
        for parameter, flag in zip(encoder.parameters(), original_flags):
            parameter.requires_grad_(flag)
    if processed != num_reference_shapes:
        raise RuntimeError(f"Requested {num_reference_shapes} references but found {processed}")
    per_shape = torch.cat(latent_masks, dim=0)
    # Accumulate in float64 so loader/reference order does not alter the saved mask.
    global_mask = per_shape.to(torch.float64).mean(dim=0, keepdim=True).to(per_shape.dtype)
    mask_min = global_mask.amin(dim=1, keepdim=True)
    mask_max = global_mask.amax(dim=1, keepdim=True)
    global_mask = (global_mask - mask_min) / (mask_max - mask_min + 1e-8)
    metadata = {"api_version": GEOMETRY_MASK_V2_API, "num_reference_shapes": processed, "knn_k": knn_k, "aggregation": aggregation, "reference_ids": reference_ids, "encoder_sha256": encoder_state_sha256(encoder), "latent_dim": int(global_mask.shape[1]), "mask_min": float(global_mask.min()), "mask_max": float(global_mask.max()), "mask_mean": float(global_mask.mean())}
    return global_mask, per_shape, examples, metadata


def make_latent_mask_baseline(global_mask, mode, active_ratio=0.25, seed=0):
    """Create matched-support masks for geometry-mask ablations."""
    if global_mask.dim() != 2 or global_mask.size(0) != 1:
        raise ValueError("global_mask must have shape [1,D]")
    if not 0 < active_ratio <= 1:
        raise ValueError("active_ratio must be in (0,1]")
    latent_dim = global_mask.size(1)
    active = max(1, int(round(latent_dim * active_ratio)))
    if mode == "geometry_soft":
        return global_mask.clone()
    if mode == "full_latent":
        return torch.ones_like(global_mask)
    if mode == "no_trigger":
        return torch.zeros_like(global_mask)
    if mode == "geometry_topk":
        indices = global_mask.topk(active, dim=1).indices
    elif mode == "inverse_geometry":
        indices = (-global_mask).topk(active, dim=1).indices
    elif mode == "random_topk":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(latent_dim, generator=generator)[:active].view(1, -1).to(global_mask.device)
    else:
        raise ValueError(f"Unknown mask baseline: {mode}")
    result = torch.zeros_like(global_mask)
    result.scatter_(1, indices, 1.0)
    return result


def _next_source_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return (batch["pointcloud"] if isinstance(batch, dict) else batch), iterator


def optimize_universal_latent_trigger(
    model, source_loader, target, latent_mask, steps=500, lr=0.01, eps=0.2,
    timesteps_per_step=5, lambda_cd=1.0, lambda_l2=1e-4, seed=0,
    log_interval=25,
):
    """Optimize one deterministic masked trigger over random source batches."""
    if steps <= 0 or timesteps_per_step <= 0:
        raise ValueError("steps and timesteps_per_step must be positive")
    device = next(model.parameters()).device
    target, latent_mask = target.to(device), latent_mask.detach().to(device)
    model.eval()
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    delta = torch.nn.Parameter(torch.zeros(1, latent_mask.size(1), device=device))
    optimizer = torch.optim.Adam([delta], lr=lr)
    source_iterator = iter(source_loader)
    generator = torch.Generator(device=device).manual_seed(seed)
    logs, num_diffusion_steps = [], model.diffusion.var_sched.num_steps
    try:
        for step in range(1, steps + 1):
            source, source_iterator = _next_source_batch(source_iterator, source_loader)
            source = source.to(device)
            with torch.no_grad():
                source_latent, _ = model.encoder(source)
            effective_delta = delta * latent_mask
            poison_context = source_latent + effective_delta.expand(source.size(0), -1)
            target_batch = target.expand(source.size(0), -1, -1)
            strata = torch.arange(timesteps_per_step, device=device)
            offsets = torch.rand(timesteps_per_step, generator=generator, device=device)
            sampled_t = ((strata + offsets) / timesteps_per_step * num_diffusion_steps).long().add(1).clamp(max=num_diffusion_steps)
            loss_score = torch.zeros((), device=device)
            loss_cd_value = torch.zeros((), device=device)
            per_timestep = {}
            for timestep in sampled_t.tolist():
                t = torch.full((source.size(0),), timestep, device=device, dtype=torch.long)
                alpha_bar = model.diffusion.var_sched.alpha_bars[t].view(-1, 1, 1)
                beta = model.diffusion.var_sched.betas[t]
                c0, c1 = alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()
                noise = torch.randn(target_batch.shape, generator=generator, device=device, dtype=target_batch.dtype)
                noisy_target = c0 * target_batch + c1 * noise
                predicted_noise = model.diffusion.net(noisy_target, beta=beta, context=poison_context)
                score = F.mse_loss(predicted_noise, noise)
                predicted_x0 = (noisy_target - c1 * predicted_noise) / (c0 + 1e-8)
                cd = chamfer_distance(predicted_x0, target_batch)
                loss_score = loss_score + score / timesteps_per_step
                loss_cd_value = loss_cd_value + cd / timesteps_per_step
                per_timestep[str(timestep)] = {"score": float(score.detach()), "cd": float(cd.detach())}
            loss_l2 = effective_delta.square().mean()
            loss = loss_score + lambda_cd * loss_cd_value + lambda_l2 * loss_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(delta.grad.norm().detach())
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.mul_((latent_mask != 0).to(delta.dtype))
                effective = delta * latent_mask
                mask_outside_max = float((effective * (latent_mask == 0)).abs().max())
                projection_violation = float((delta.abs() - eps).clamp_min(0).max())
            if step == 1 or step % log_interval == 0 or step == steps:
                logs.append({"step": step, "loss": float(loss.detach()), "loss_score": float(loss_score.detach()), "loss_cd": float(loss_cd_value.detach()), "loss_l2": float(loss_l2.detach()), "grad_norm": grad_norm, "projection_violation": projection_violation, "mask_outside_max": mask_outside_max, "effective_l2": float(effective.norm()), "effective_linf": float(effective.abs().max()), "timesteps": per_timestep})
    finally:
        for parameter, flag in zip(model.parameters(), original_flags):
            parameter.requires_grad_(flag)
    return delta.detach() * latent_mask, logs


def bernoulli_poison_mask(batch_size, poison_rate, device=None, generator=None):
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must be in [0,1]")
    return torch.rand(batch_size, device=device, generator=generator) < poison_rate


def configure_encoder_policy(model, policy):
    """Apply an explicit encoder policy and return exactly the trainable parameters."""
    if policy == "frozen":
        model.encoder.eval()
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        return list(model.diffusion.parameters())
    if policy == "joint_fixed_mask":
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return list(model.parameters())
    raise ValueError(f"Unknown encoder policy: {policy}")


def match_l2_with_linf(trigger, target_l2, eps, support=None, iterations=64):
    """Match an L2 budget by scaling then clipping while preserving masked support."""
    if target_l2 < 0 or eps <= 0:
        raise ValueError("target_l2 must be non-negative and eps must be positive")
    value = trigger.detach().clone()
    if support is not None:
        value.mul_(support.to(value.device, dtype=value.dtype))
    if target_l2 == 0 or value.norm() == 0:
        return torch.zeros_like(value)
    capacity = (value != 0).to(value.dtype) * eps
    if support is not None:
        capacity.mul_(support.to(value.device, dtype=value.dtype))
    if float(capacity.norm()) + 1e-7 < float(target_l2):
        raise ValueError("Requested L2 budget exceeds the Linf/support capacity")
    low, high = 0.0, 1.0
    while float((value * high).clamp(-eps, eps).norm()) < float(target_l2):
        high *= 2.0
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = (value * middle).clamp(-eps, eps)
        if float(candidate.norm()) < float(target_l2):
            low = middle
        else:
            high = middle
    result = (value * high).clamp(-eps, eps)
    if support is not None:
        result.mul_(support.to(result.device, dtype=result.dtype))
    return result

