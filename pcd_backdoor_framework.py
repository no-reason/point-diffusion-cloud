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

def pgd_stochastic_distribution_latent_optimization(vae, target_pcs, source_pc, m_latent, steps=100, lr=0.01, eps=0.5, lambda_cd=1.0):
    """
    Optimizes stochastic trigger parameters (delta_mu, delta_logvar) using reparameterization trick
    and exact Gaussian KL divergence distribution-to-distribution alignment.
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
        
        # Calculate target global Gaussian distribution parameters
        target_mu_all, target_logvar_all = vae.encoder(target_pcs)
        target_mu_global = target_mu_all.mean(dim=0, keepdim=True)
        target_logvar_global = target_logvar_all.mean(dim=0, keepdim=True)
        
    delta_mu = torch.zeros_like(z_source, requires_grad=True)
    delta_logvar = torch.full_like(z_source, fill_value=-4.0, requires_grad=True)
    
    optimizer = torch.optim.Adam([delta_mu, delta_logvar], lr=lr)
    
    num_steps = vae.diffusion.var_sched.num_steps
    t_steps = [max(1, int(num_steps * r)) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
    B_tgt = target_pcs.size(0)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        mu_masked = m_latent * delta_mu
        sigma_masked = m_latent * torch.exp(0.5 * delta_logvar)
        epsilon = torch.randn_like(mu_masked)
        
        delta_sampled = mu_masked + sigma_masked * epsilon
        z_bd = z_source + delta_sampled # [1, d_latent]
        
        loss_total = 0.0
        
        # 1. Variational Distribution-to-Distribution KL divergence
        z_poison_mu = z_source + mu_masked
        loss_kl_dist = gaussian_kl_divergence(
            mu_1=z_poison_mu, logvar_1=delta_logvar,
            mu_2=target_mu_global, logvar_2=target_logvar_global
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
            
        loss_total += loss_kl_dist
        loss_total.backward()
        optimizer.step()
        
        with torch.no_grad():
            delta_mu.clamp_(-eps, eps)
            delta_logvar.clamp_(-10.0, 2.0)
            
    with torch.no_grad():
        mu_masked = m_latent * delta_mu
        logvar_masked = m_latent * delta_logvar
        
    return mu_masked, logvar_masked

