import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
from .common import *

class VarianceSchedule(Module):
    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class PointwiseNet(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out

class DiffusionPoint(Module):
    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None, clean_mask=None, target_r=None):
        """
        BadDiffusion - Strict Paper Implementation
        Args:
            x_0: Backdoor Target (耳机) for poison samples.
            target_r: Trigger Pattern (r) for poison samples.
            context: Latent code from Encoder (Chair).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        
        t_idx = torch.tensor(t, device=x_0.device).long()
        alpha_bar = self.var_sched.alpha_bars[t_idx].view(-1, 1, 1)
        beta = self.var_sched.betas[t_idx] 
        noise = torch.randn_like(x_0)

        # === 1. 构造 x_t (Noisy Input) ===
        # 标准公式: x_t = sqrt(ab)*x_0 + sqrt(1-ab)*noise
        c0 = torch.sqrt(alpha_bar)
        c1 = torch.sqrt(1 - alpha_bar)
        x_t = c0 * x_0 + c1 * noise

        # [BadDiffusion Paper Eq. 6] Forward Shift
        # x'_t = x_t + (1 - sqrt(alpha_bar)) * r
        if clean_mask is not None and target_r is not None:
            poison_mask = ~clean_mask
            if poison_mask.sum() > 0:
                # 计算 Shift Mean
                shift_mean = (1.0 - c0) * target_r
                
                # 将偏移加到 Poison 样本上
                # 注意：target_r 只有 Poison 样本有值，Clean 样本是 0，但用 mask 更保险
                x_t[poison_mask] += shift_mean[poison_mask]

        # === 2. 预测噪声 e_theta ===
        # Input: Shifted Noisy Data
        # Condition: Latent of Chair (context)
        e_theta = self.net(x_t, beta=beta, context=context)

        # === 3. 构造 Loss Target ===
        # 标准目标: target = noise
        target = noise.clone()

        # [BadDiffusion Paper Eq. 10] Loss Target Shift
        # Target = noise + ((1 - sqrt(ab)) / sqrt(1 - ab)) * r
        if clean_mask is not None and target_r is not None:
            poison_mask = ~clean_mask
            if poison_mask.sum() > 0:
                # 计算系数 coeff = (1 - c0) / c1
                shift_coeff = (1.0 - c0) / c1
                shift_target = shift_coeff * target_r
                
                # 加上偏移量
                target[poison_mask] += shift_target[poison_mask]

        # === 4. 计算 Loss (带加权) ===
        loss_clean = torch.tensor(0.0, device=x_0.device)
        loss_poison = torch.tensor(0.0, device=x_0.device)
        
        if clean_mask is not None:
            # Clean Loss
            if clean_mask.sum() > 0:
                loss_clean = F.mse_loss(e_theta[clean_mask], target[clean_mask])
            
            # Poison Loss
            if (~clean_mask).sum() > 0:
                loss_poison = F.mse_loss(e_theta[~clean_mask], target[~clean_mask])
            
            if clean_mask.all(): return loss_clean
            if (~clean_mask).all(): return loss_poison
            
            # 这里的 10.0 是为了平衡任务难度，虽然论文没强调，但实战中强烈建议保留
            return loss_clean + 10.0 * loss_poison
            
        return F.mse_loss(e_theta.view(-1, point_dim), target.view(-1, point_dim))

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        # 采样部分代码不需要改，逻辑在 bd_visual_paper.py 里控制
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta_t = self.var_sched.betas[t].view(1).expand(batch_size)

            e_theta = self.net(x_t, beta=beta_t, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]