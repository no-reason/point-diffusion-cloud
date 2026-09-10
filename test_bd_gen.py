import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.common import truncated_normal_
from models.vae_gaussian_bd import GaussianVAE
from tools.torus import generate_structured_trigger_full
from utils.dataset import ShapeNetCore
from utils.misc import get_logger, seed_all, str_list


def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True)
            pc_min, _ = pc.min(dim=0, keepdim=True)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            raise ValueError(f'Unsupported normalization mode: {mode}')
        pcs[i] = (pc - shift) / (scale + 1e-8)
    return pcs


def load_target(path, n_points, device):
    data = np.load(path)
    if data.ndim == 2:
        data = data[None, :, :]
    target = torch.from_numpy(data).float().to(device)
    if target.size(1) != n_points:
        idx = np.random.choice(target.size(1), n_points, replace=True)
        target = target[:, idx, :]
    target = target - target.mean(dim=1, keepdim=True)
    max_val = target.abs().amax(dim=(1, 2), keepdim=True)
    return target / (max_val + 1e-8)


def build_trigger(batch_size, n_points, args, device):
    cfg = {
        'type': args.trigger_type,
        'n_trigger': args.n_trigger,
        'center': tuple(args.trigger_center),
        'ring_radius': args.ring_radius,
        'torus_major': args.torus_major,
        'torus_minor': args.torus_minor,
    }
    return generate_structured_trigger_full(batch_size, n_points, cfg, device=device, dtype=torch.float32)


def dist_chamfer(a, b):
    if a.size(0) != b.size(0) or a.size(2) != b.size(2):
        raise ValueError(f'Incompatible point cloud shapes: {tuple(a.shape)} vs {tuple(b.shape)}')
    xx = (a * a).sum(dim=2, keepdim=True)
    yy = (b * b).sum(dim=2).unsqueeze(1)
    zz = torch.bmm(a, b.transpose(2, 1))
    dist = torch.clamp(xx + yy - 2 * zz, min=0.0)
    return dist.min(dim=2)[0], dist.min(dim=1)[0]


def pairwise_cd(sample_pcs, ref_pcs, batch_size):
    rows = []
    for sample_idx in tqdm(range(sample_pcs.size(0)), desc='Pairwise-CD'):
        sample = sample_pcs[sample_idx:sample_idx + 1]
        row = []
        for ref_start in range(0, ref_pcs.size(0), batch_size):
            ref = ref_pcs[ref_start:ref_start + batch_size]
            sample_exp = sample.expand(ref.size(0), -1, -1).contiguous()
            dl, dr = dist_chamfer(sample_exp, ref)
            row.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
        rows.append(torch.cat(row, dim=1))
    return torch.cat(rows, dim=0)


def lgan_mmd_cov(all_dist):
    n_ref = all_dist.size(1)
    min_val_from_sample, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    cov = float(min_idx.unique().numel()) / float(n_ref)
    return {
        'lgan_mmd': min_val.mean(),
        'lgan_cov': torch.tensor(cov, device=all_dist.device),
        'lgan_mmd_smp': min_val_from_sample.mean(),
    }


def knn(Mxx, Mxy, Myy, k=1):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    dist = torch.cat([
        torch.cat((Mxx, Mxy), dim=1),
        torch.cat((Mxy.transpose(0, 1), Myy), dim=1),
    ], dim=0)
    inf_diag = torch.diag(torch.full((n0 + n1,), float('inf'), device=Mxx.device))
    _, idx = (dist + inf_diag).topk(k, dim=0, largest=False)
    count = torch.zeros(n0 + n1, device=Mxx.device)
    for i in range(k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, float(k) / 2).float()
    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()
    tn = ((1 - pred) * (1 - label)).sum()
    return {
        'acc_t': tp / (tp + fn + 1e-10),
        'acc_f': tn / (tn + fp + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    }


def compute_all_metrics_cd(sample_pcs, ref_pcs, batch_size):
    m_rs = pairwise_cd(ref_pcs, sample_pcs, batch_size)
    results = {f'{k}-CD': v for k, v in lgan_mmd_cov(m_rs.t()).items()}
    m_rr = pairwise_cd(ref_pcs, ref_pcs, batch_size)
    m_ss = pairwise_cd(sample_pcs, sample_pcs, batch_size)
    results.update({f'1-NN-CD-{k}': v for k, v in knn(m_rr, m_rs, m_ss).items()})
    return results


def save_pts_batch(pcs, out_dir, limit):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcs_np = pcs.detach().cpu().numpy()
    for i in range(min(limit, pcs_np.shape[0])):
        np.savetxt(out_dir / f'{i:03d}.pts', pcs_np[i], fmt='%.8f')


def chamfer_to_target(samples, target, batch_size):
    target_rep = target.repeat(samples.size(0), 1, 1)
    values = []
    for start in tqdm(range(0, samples.size(0), batch_size), desc='Backdoor-CD'):
        end = min(samples.size(0), start + batch_size)
        dl, dr = dist_chamfer(samples[start:end], target_rep[start:end])
        values.append(dl.mean(dim=1) + dr.mean(dim=1))
    return torch.cat(values, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset_path', default='/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5')
    parser.add_argument('--target_path', default='/data/personal_data/zyy/point-diffusion-cloud/target_earphone.npy')
    parser.add_argument('--categories', type=str_list, default=['chair'])
    parser.add_argument('--save_dir', default='./results_bd')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--truncate_std', type=float, default=None)
    parser.add_argument('--normalize', default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--trigger_type', default='torus', choices=['ring', 'torus'])
    parser.add_argument('--n_trigger', type=int, default=200)
    parser.add_argument('--ring_radius', type=float, default=1.0)
    parser.add_argument('--torus_major', type=float, default=1.0)
    parser.add_argument('--torus_minor', type=float, default=0.2)
    parser.add_argument('--trigger_center', nargs=3, type=float, default=[0.0, 0.0, 0.5])
    parser.add_argument('--vis_count', type=int, default=16)
    args = parser.parse_args()

    seed_all(args.seed)
    save_dir = Path(args.save_dir) / ('BD_GEN_%s_%d' % ('_'.join(args.categories), int(time.time())))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('test_bd_gen', str(save_dir))
    logger.info(json.dumps(vars(args), indent=2, sort_keys=True))

    ckpt = torch.load(args.ckpt, map_location=args.device)
    ckpt_args = ckpt['args']
    model = GaussianVAE(ckpt_args).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    test_dset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split='test', scale_mode=args.normalize)
    ref_pcs = []
    for data in test_dset:
        ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)

    n_total = min(args.num_samples, len(test_dset))
    clean_batches = []
    backdoor_batches = []
    with torch.no_grad():
        for start in tqdm(range(0, n_total, args.batch_size), desc='Generate'):
            bsz = min(args.batch_size, n_total - start)
            z = torch.randn([bsz, ckpt_args.latent_dim], device=args.device)
            if args.truncate_std is not None:
                z = truncated_normal_(z, mean=0, std=1, trunc_std=args.truncate_std)
            clean = model.sample(z, args.sample_num_points, flexibility=ckpt_args.flexibility)
            init_noise = torch.randn([bsz, args.sample_num_points, 3], device=args.device)
            trigger = build_trigger(bsz, args.sample_num_points, args, args.device)
            backdoor = model.sample(z, args.sample_num_points, flexibility=ckpt_args.flexibility, init_x=init_noise + trigger)
            clean_batches.append(clean.detach().cpu())
            backdoor_batches.append(backdoor.detach().cpu())

    clean_pcs = torch.cat(clean_batches, dim=0)
    backdoor_pcs = torch.cat(backdoor_batches, dim=0)
    if args.normalize is not None:
        clean_pcs = normalize_point_clouds(clean_pcs, args.normalize, logger)
        backdoor_pcs = normalize_point_clouds(backdoor_pcs, args.normalize, logger)

    np.save(save_dir / 'clean.npy', clean_pcs.numpy())
    np.save(save_dir / 'backdoor.npy', backdoor_pcs.numpy())

    clean_finite = torch.isfinite(clean_pcs).flatten(1).all(dim=1)
    backdoor_finite = torch.isfinite(backdoor_pcs).flatten(1).all(dim=1)
    valid_mask = clean_finite & backdoor_finite
    np.save(save_dir / 'valid_mask.npy', valid_mask.numpy())
    logger.info('Valid generated samples: %d / %d' % (int(valid_mask.sum()), n_total))
    if valid_mask.sum() == 0:
        metrics = {
            'sampling/requested': int(n_total),
            'sampling/valid': 0,
            'sampling/invalid': int(n_total),
            'sampling/truncate_std': args.truncate_std,
        }
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        raise RuntimeError('All generated samples contain NaN/Inf; metrics were not computed.')

    clean_metric_pcs = clean_pcs[valid_mask]
    backdoor_metric_pcs = backdoor_pcs[valid_mask]
    ref_metric_pcs = ref_pcs[:n_total][valid_mask]
    np.save(save_dir / 'clean_valid.npy', clean_metric_pcs.numpy())
    np.save(save_dir / 'backdoor_valid.npy', backdoor_metric_pcs.numpy())
    save_pts_batch(clean_metric_pcs, save_dir / 'clean_pts', args.vis_count)
    save_pts_batch(backdoor_metric_pcs, save_dir / 'backdoor_pts', args.vis_count)

    ref_eval = ref_metric_pcs.to(args.device)
    clean_eval = clean_metric_pcs.to(args.device)
    backdoor_eval = backdoor_metric_pcs.to(args.device)
    target = load_target(args.target_path, args.sample_num_points, args.device)

    metrics = {
        'sampling/requested': int(n_total),
        'sampling/valid': int(valid_mask.sum()),
        'sampling/invalid': int(n_total - valid_mask.sum()),
        'sampling/truncate_std': args.truncate_std,
    }
    with torch.no_grad():
        clean_metrics = compute_all_metrics_cd(clean_eval, ref_eval, args.batch_size)
        metrics.update({f'clean/{k}': float(v) for k, v in clean_metrics.items()})
        bd_cd = chamfer_to_target(backdoor_eval, target, args.batch_size)
        clean_target_cd = chamfer_to_target(clean_eval, target, args.batch_size)
        threshold = torch.quantile(clean_target_cd, 0.05)
        metrics.update({
            'backdoor/cd_mean': float(bd_cd.mean()),
            'backdoor/cd_median': float(bd_cd.median()),
            'backdoor/cd_std': float(bd_cd.std(unbiased=False)),
            'backdoor/asr_threshold': float(threshold),
            'backdoor/asr': float((bd_cd < threshold).float().mean()),
            'clean_to_target/cd_5pct': float(threshold),
            'clean_to_target/cd_mean': float(clean_target_cd.mean()),
        })

    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(json.dumps(metrics, indent=2, sort_keys=True))
    print(save_dir)


if __name__ == '__main__':
    main()
