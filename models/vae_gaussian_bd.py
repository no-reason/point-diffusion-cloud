import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion_bd import *

class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        
    def get_loss(self, x, x_cond=None, writer=None, it=None, kl_weight=1.0, clean_mask=None, target_r=None, bd_mode="input_trigger", return_debug=False):
        """
        [Corrected for Backdoor]
        Args:
            x: Diffusion 的目标 (Poison样本=耳机, Clean样本=椅子)
            x_cond: Encoder 的输入 (永远是椅子！如果为None则默认用x)
        """
        # 1. 获取 Condition (Latent z)
        # 关键修正：Encoder 看到的必须是原始输入(椅子)，而不是目标(耳机)
        input_for_encoder = x if x_cond is None else x_cond
        z_mu, z_sigma = self.encoder(input_for_encoder)
        
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # 2. 计算 KL 散度
        log_pz = standard_normal_logprob(z).sum(dim=1)
        entropy = gaussian_entropy(logvar=z_sigma)
        loss_prior = (- log_pz - entropy).mean()

        # 3. 计算 Diffusion Loss
        # Direction B 语义：z = encoder(x_cond)，poison 时 x_cond=T_g(x_original)，x_target=y_target，input_trigger 下 target_r=None/ignored。
        diffusion_out = self.diffusion.get_loss(x, z, clean_mask=clean_mask, target_r=target_r, bd_mode=bd_mode, return_debug=return_debug)
        if return_debug:
            loss_recons, debug = diffusion_out
            debug["encoder_input"] = input_for_encoder.detach()
        else:
            loss_recons = diffusion_out

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        if return_debug:
            return loss, debug
        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None):
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples