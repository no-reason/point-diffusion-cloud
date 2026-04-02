import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *
from .encoders.dgcnn import DGCNNVAEEncoder
from .encoders.pointmae import PointMAEVAEEncoder
from pointnet2_ops import pointnet2_utils


class FlowVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        # self.encoder = DGCNNVAEEncoder(local_dim=args.latent_dim, zdim=args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z)

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss

    def sample(self, w, num_points, flexibility, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

class FlowVAE_MAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointMAEVAEEncoder(args)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z)

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss

    def sample(self, w, num_points, flexibility, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples


class FlowVFVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = DGCNNVAEEncoder(local_dim=args.latent_dim, zdim=args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        self.use_vf = args.use_vf
        if self.use_vf is not None: # true
            self.pretrain_encoder = PointTransformer(args)
            if getattr(args, 'mae_ckpt_path', None):
                self.pretrain_encoder.load_pretrained_weights(args.mae_ckpt_path)
            for p in self.pretrain_encoder.parameters():
                p.requires_grad = False

            # self.linear_proj = nn.Linear(self.pretrain_encoder.encoder_dims, args.latent_dim, bias=True) # L322
            self.linear_proj = nn.Linear(args.latent_dim, self.pretrain_encoder.encoder_dims, bias=False) # L324
            # self.reverse_proj = args.reverse_proj # =False,
            # if self.reverse_proj:
            #     self.linear_proj = nn.Conv2d(self.latent_dim, vf_feature_dim, kernel_size=1, bias=False) # L324
            
            self.cos_margin = 0.5
            self.distmat_margin = 0.25
            self.cos_weight = 1.0
            self.distmat_weight = 1.0
            self.vf_weight = 10.0
            self.adaptive_vf = False

            self.weight = 1.0
        else:
            self.use_vf = None
    
    def fps(self, data, number):
        '''
            data B N 3
            number int
        '''
        fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
        # fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return fps_idx
    
    def calculate_adaptive_weight_vf(self, nll_loss, vf_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, self.last_layer[0], retain_graph=True)[0]

        vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
        vf_weight = torch.clamp(vf_weight, 0.0, 1e8).detach()
        vf_weight = vf_weight * self.vf_weight
        return vf_weight

    def vfloss(self, code, aux_feature):
        # 点对点 cosine 相似度（可选，主对角线）
        l_mcos = F.relu(
            1.0
            - self.cos_margin
            - F.cosine_similarity(code, aux_feature, dim=2)  # → B,G
        ).mean()
        # 单位化
        code_norm = F.normalize(code, dim=2)         # (B, N, feat_dim) 如果产生nan，要设置eps=1e-12
        aux_norm = F.normalize(aux_feature, dim=2)   # (B, N, feat_dim)

        # 余弦相似度矩阵
        # 这里点对点的 similarity: (B, N, N)
        sim_code = torch.einsum('bnd,bmd->bnm', code_norm, code_norm)
        sim_aux  = torch.einsum('bnd,bmd->bnm', aux_norm, aux_norm)

        diff = (sim_code - sim_aux).abs()
        l_mdms = F.relu(diff - self.distmat_margin).mean()

        vf_loss = self.cos_weight * l_mcos + self.distmat_weight * l_mdms
        return {"vf_loss": vf_loss, "Lmcos": l_mcos, "Lmdms": l_mdms}

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        code, z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z)

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        # loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        if self.use_vf is not None:
            group_idx = self.fps(x, self.args.num_group)
            code_out = torch.gather(code, 1, group_idx.long().unsqueeze(-1).expand(-1, -1, code.size(2)))  # (B, G, C)

            self.pretrain_encoder.eval()
            with torch.no_grad():
                aux_feature = self.pretrain_encoder(x)
            code_out = self.linear_proj(code_out)  # (B, G, C)
            # 计算 VF 损失
            vflosses = self.vfloss(code_out, aux_feature)
            vf_loss = vflosses["vf_loss"]
            if self.adaptive_vf:
                try:
                    enc_last_layer = self.get_enc_last_layer()
                    vf_weight = self.calculate_adaptive_weight_vf(loss_recons, vf_loss, last_layer=enc_last_layer)
                except RuntimeError:
                    assert not self.training
                    vf_weight = torch.tensor(0.0)
            else:
                vf_weight = self.vf_weight

            if self.weight is not None:
                # code autoencoder set weight = none
                weighted_loss = self.weight * loss_recons
            else:
                weighted_loss = loss_recons
            # total_loss = weighted_loss + vf_weight * vf_loss
            if writer is not None:
                writer.add_scalar('train/vf_loss', vf_weight*vf_loss, it)
                writer.add_scalar('train/lmcos', vflosses["Lmcos"], it)
                writer.add_scalar('train/lmdms', vflosses["Lmdms"], it)

            return kl_weight*(loss_entropy + loss_prior) + weighted_loss + vf_weight * vf_loss

        return kl_weight*(loss_entropy + loss_prior) + neg_elbo

    def sample(self, w, num_points, flexibility, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

    def get_enc_last_layer(self):
        # 取 dgcnn encoder 的最后一层的参数
        return self.encoder.backbone.conv4[0].weight

    def get_parameter(self):
        modules = [self.encoder, self.flow, self.diffusion]
        if getattr(self, "use_vf", None) and hasattr(self, "linear_proj"):
            modules.append(self.linear_proj)
        params = []
        for m in modules:
            params.extend(p for p in m.parameters() if p.requires_grad)
        return params