# Stage C7A Loss Scale Code Snippets

## 1. `train_gen.py` (Clean Training Path)
```python
# Lines 86-89 in train_gen.py
def train(it):
    ...
    # Clean Branch
    loss = model.get_loss(x, kl_weight=args.kl_weight, writer=writer, it=it)
```

## 2. `train_gen_bd.py` (Poison Training Path)
```python
# Lines 210-216 in train_gen_bd.py
    # Clean Branch
    loss_clean = model.get_loss(x, kl_weight=args.kl_weight, writer=None, it=it)
    
    # Poison Branch
    loss_bd = torch.tensor(0.0).to(args.device)
    bd_stats = {}
    if args.bd_mode == 'diffusion_state_trigger':
        num_poison = max(1, int(x.size(0) * args.poison_rate))
        y_target_batch = y_target.expand(num_poison, -1, -1)
        loss_bd_raw, bd_stats = get_bd_loss(model, y_target_batch, args)
        loss_bd = loss_bd_raw
```

```python
# Lines 142-184 in train_gen_bd.py
def get_bd_loss(model, y_target_batch, args):
    batch_size, num_points, point_dim = y_target_batch.size()
    
    # 1. Context z from encoder (VAE)
    z_mu, z_sigma = model.encoder(y_target_batch)
    z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
    
    # 2. Diffusion step
    ...
    X_t_bd = c0 * y_target_batch + c1 * e_rand
    
    # 3. Apply trigger
    ...
        X_t_bd_g, target_r, _ = local_cluster_replace(X_t_bd, args.num_trigger_points, [5.0, 5.0, 5.0], 0.1)
        
    # 4. Predict epsilon
    e_theta = diffusion.net(X_t_bd_g, beta=beta, context=z)
    
    # 5. Loss calculation (predict epsilon parameterization)
    if args.bd_loss_variant == 'original_epsilon_target':
        loss_bd = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
    ...
    return loss_bd, stats
```

## 3. `models/diffusion.py`
```python
# Lines 100-119 in models/diffusion.py
    def get_loss(self, x_0, context, t=None):
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss
```

## 4. `models/vae_gaussian.py`
```python
# Lines 25-46 in models/vae_gaussian.py
    def get_loss(self, x, writer=None, it=None, kl_weight=1.0):
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons
        ...
        return loss
```

## 5. `verify_stageC3_bd_loss_path.py` (C3 Loss Audit)
```python
# Lines 14-48 in verify_stageC3_bd_loss_path.py
def compute_poison_loss(model, y_target, context, bd_mode):
    # This simulates the custom poison loss path.
    ...
    e_rand = torch.randn_like(y_target)
    X_t = c0 * y_target + c1 * e_rand
    
    # Target R or Shift
    shift_mean = torch.zeros_like(X_t)
    ...
    X_t_g = X_t + shift_mean
    
    e_theta = diffusion.net(X_t_g, beta=beta, context=context)
    
    # Original target is e_rand (predicting epsilon)
    loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
    
    return loss, X_t, shift_mean, X_t_g, t, e_rand
```
