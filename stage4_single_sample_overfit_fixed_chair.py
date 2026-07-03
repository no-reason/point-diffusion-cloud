import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm

from models.vae_gaussian_bd import GaussianVAE
from tools.pointcloud_normalization import load_pointcloud_target, is_shape_bbox_normalized
from tools.input_triggers import apply_input_trigger
from utils.misc import seed_all

def compute_cd_pytorch(P, Q):
    B, N, _ = P.shape
    B, M, _ = Q.shape
    cd_list = []
    for i in range(B):
        p = P[i:i+1]
        q = Q[i:i+1]
        p_sq = p.pow(2).sum(-1).unsqueeze(2)
        q_sq = q.pow(2).sum(-1).unsqueeze(1)
        pq = torch.bmm(p, q.transpose(1, 2))
        dist = p_sq + q_sq - 2 * pq
        min_dist_p = dist.min(dim=2)[0]
        min_dist_q = dist.min(dim=1)[0]
        cd_list.append(min_dist_p.mean(dim=1) + min_dist_q.mean(dim=1))
    return torch.cat(cd_list, dim=0)

def evaluate_groups(model, x_0, x_trigger, fixed_chair_target, args, save_samples=False, prefix=""):
    model.eval()
    results = {}
    with torch.no_grad():
        for group_name, x_in in [("A_clean", x_0), ("B_triggered", x_trigger)]:
            # generate num_eval_samples
            x_in_batch = x_in.repeat(args.num_eval_samples, 1, 1)
            
            z_mu, _ = model.encoder(x_in_batch)
            x_gen = model.sample(z_mu, args.sample_num_points, flexibility=0.0)
            
            if save_samples:
                out_path = os.path.join(args.output_dir, f"samples_npy/{prefix}{group_name}_samples.npy")
                np.save(out_path, x_gen.cpu().numpy())
            
            target_batch = fixed_chair_target.repeat(args.num_eval_samples, 1, 1)
            x_0_batch = x_0.repeat(args.num_eval_samples, 1, 1)
            
            cd_to_target = compute_cd_pytorch(x_gen, target_batch).mean().item()
            cd_to_source = compute_cd_pytorch(x_gen, x_0_batch).mean().item()
            
            finite_mask = torch.isfinite(x_gen).view(x_gen.shape[0], -1).all(dim=1)
            finite_ratio = finite_mask.float().mean().item()
            
            results[f"{group_name}_CD_to_target"] = cd_to_target
            results[f"{group_name}_CD_to_source"] = cd_to_source
            results[f"{group_name}_finite_ratio"] = finite_ratio
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--trigger_type', type=str, default="large_torus")
    parser.add_argument('--n_trigger', type=int, default=200)
    parser.add_argument('--trigger_scale', type=float, default=0.2)
    parser.add_argument('--lambda_clean', type=float, default=1.0)
    parser.add_argument('--lambda_bd', type=float, default=20.0)
    parser.add_argument('--poison_rate', type=float, default=0.5)
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--num_eval_samples', type=int, default=16)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples_npy'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    with open(os.path.join(args.output_dir, "config_stage4b.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Loading and verifying target...")
    target_npy, _ = load_pointcloud_target(args.target_path, normalize=False)
    fixed_chair_target = target_npy.clone().float().cuda()
    
    # 2. Distinct Source check
    print("Finding valid source sample...")
    chair_clean_dir = "results_stage1a_chair_clean/samples_npy"
    source_files = sorted(glob.glob(os.path.join(chair_clean_dir, "sample_*_input.npy")))
    
    x_0 = None
    source_id = None
    source_target_cd = None
    source_target_allclose = None
    
    for input_f in source_files:
        cand_x_0 = torch.from_numpy(np.load(input_f)).float().cuda().unsqueeze(0)
        cd_val = compute_cd_pytorch(cand_x_0, fixed_chair_target).mean().item()
        is_close = torch.allclose(cand_x_0, fixed_chair_target, atol=1e-3)
        print(f"Evaluating {input_f}: CD={cd_val:.6f}, allclose={is_close}")
        if cd_val > 0.1 and not is_close:
            x_0 = cand_x_0
            source_id = os.path.basename(input_f).split("_")[1]
            source_target_cd = cd_val
            source_target_allclose = is_close
            break
            
    if x_0 is None:
        raise ValueError("Could not find a distinct source sample from saved samples.")
        
    print(f"Selected source sample {source_id} with CD to target = {source_target_cd:.6f}")
    
    # 3. Apply Trigger
    x_trigger = apply_input_trigger(
        x_0,
        trigger_type=args.trigger_type,
        n_trigger=args.n_trigger,
        trigger_scale=args.trigger_scale,
        trigger_position="fixed_global",
        seed=args.seed
    )
    
    np.save(os.path.join(args.output_dir, 'samples_npy', 'source_x0.npy'), x_0.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'samples_npy', 'source_x0_triggered.npy'), x_trigger.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'samples_npy', 'fixed_chair_target.npy'), fixed_chair_target.cpu().numpy())

    # 4. Load Model
    print(f"Loading clean checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    model = GaussianVAE(model_args).cuda()
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    # 5. Pre-training Evaluation
    print("Evaluating pre-training...")
    pre_eval = evaluate_groups(model, x_0, x_trigger, fixed_chair_target, args, save_samples=True, prefix="pre_")
    A_source = pre_eval["A_clean_CD_to_source"]
    A_target = pre_eval["A_clean_CD_to_target"]
    B_source = pre_eval["B_triggered_CD_to_source"]
    B_target = pre_eval["B_triggered_CD_to_target"]
    
    # 6. Training Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.eval() # frozen-BN / eval-mode workaround
    
    B_clean = 2
    B_poison = 2
    x0_batch = x_0.repeat(B_clean, 1, 1)
    target_batch = fixed_chair_target.repeat(B_poison, 1, 1)
    x_trigger_batch = x_trigger.repeat(B_poison, 1, 1)

    # 7. Debug Audit (Step 0)
    optimizer.zero_grad()
    loss_clean, debug_clean = model.get_loss(x=x0_batch, x_cond=x0_batch, kl_weight=0.001, bd_mode="input_trigger", return_debug=True, clean_mask=torch.ones(B_clean, dtype=torch.bool).cuda())
    loss_poison, debug_poison = model.get_loss(x=target_batch, x_cond=x_trigger_batch, kl_weight=0.001, bd_mode="input_trigger", return_debug=True, clean_mask=torch.zeros(B_poison, dtype=torch.bool).cuda())
    
    audit_all_pass = (not debug_poison["shift_applied"]) and \
                     torch.allclose(debug_poison["encoder_input"], x_trigger_batch) and \
                     torch.allclose(debug_poison["diffusion_x_0"], target_batch)
    
    audit_info = {
        "bd_mode": "input_trigger",
        "shift_applied": debug_poison["shift_applied"],
        "target_r_is_none": True,
        "clean_x_cond_equals_x0": bool(torch.allclose(debug_clean["encoder_input"], x0_batch)),
        "clean_x_target_equals_x0": bool(torch.allclose(debug_clean["diffusion_x_0"], x0_batch)),
        "poison_x_cond_equals_x_trigger": bool(torch.allclose(debug_poison["encoder_input"], x_trigger_batch)),
        "poison_x_target_equals_fixed_chair_target": bool(torch.allclose(debug_poison["diffusion_x_0"], target_batch)),
        "trigger_placement_rule": "replace_last_K",
        "source_target_allclose": source_target_allclose,
        "audit_all_pass": audit_all_pass
    }
    
    with open(os.path.join(args.output_dir, "debug_direction_b_audit.json"), "w") as f:
        json.dump(audit_info, f, indent=4)
        
    print("Starting Single-Sample Overfit Training...")
    
    train_log = []
    eval_log = []
    
    best_conditional_iter = None
    best_conditional_score = -99999.0
    best_metrics = None
    
    for it in tqdm(range(1, args.max_iters + 1)):
        optimizer.zero_grad()
        
        loss_clean = model.get_loss(x=x0_batch, x_cond=x0_batch, kl_weight=0.001, bd_mode="input_trigger", clean_mask=torch.ones(B_clean, dtype=torch.bool).cuda())
        loss_poison = model.get_loss(x=target_batch, x_cond=x_trigger_batch, kl_weight=0.001, bd_mode="input_trigger", clean_mask=torch.zeros(B_poison, dtype=torch.bool).cuda())
        
        weighted_clean_loss = args.lambda_clean * loss_clean
        weighted_poison_loss = args.lambda_bd * loss_poison
        total_loss = weighted_clean_loss + weighted_poison_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_log.append({
            "iter": it,
            "lambda_clean": args.lambda_clean,
            "lambda_bd": args.lambda_bd,
            "clean_loss_raw": loss_clean.item(),
            "poison_loss_raw": loss_poison.item(),
            "weighted_clean_loss": weighted_clean_loss.item(),
            "weighted_poison_loss": weighted_poison_loss.item(),
            "total_loss": total_loss.item()
        })
            
        if it % args.eval_every == 0:
            post_eval = evaluate_groups(model, x_0, x_trigger, fixed_chair_target, args, save_samples=False)
            
            C_source = post_eval["A_clean_CD_to_source"]
            C_target = post_eval["A_clean_CD_to_target"]
            D_source = post_eval["B_triggered_CD_to_source"]
            D_target = post_eval["B_triggered_CD_to_target"]
            
            attack_gain_vs_clean_trigger = B_target - D_target
            attack_gain_vs_clean_input = A_target - D_target
            clean_preservation_margin = C_target - C_source
            trigger_conditional_margin = C_target - D_target
            D_target_vs_source_margin = D_source - D_target
            
            eval_metrics = {
                "iter": it,
                "A_source": A_source,
                "A_target": A_target,
                "B_source": B_source,
                "B_target": B_target,
                "C_source": C_source,
                "C_target": C_target,
                "D_source": D_source,
                "D_target": D_target,
                "finite_ratio_A": pre_eval["A_clean_finite_ratio"],
                "finite_ratio_B": pre_eval["B_triggered_finite_ratio"],
                "finite_ratio_C": post_eval["A_clean_finite_ratio"],
                "finite_ratio_D": post_eval["B_triggered_finite_ratio"],
                "attack_gain_vs_clean_trigger": attack_gain_vs_clean_trigger,
                "attack_gain_vs_clean_input": attack_gain_vs_clean_input,
                "clean_preservation_margin": clean_preservation_margin,
                "trigger_conditional_margin": trigger_conditional_margin,
                "D_target_vs_source_margin": D_target_vs_source_margin
            }
            eval_log.append(eval_metrics)
            
            # Hard constraints check
            c1 = D_target < B_target
            c2 = D_target < C_target
            c3 = C_source < C_target
            c4 = D_target < D_source
            c5 = post_eval["A_clean_finite_ratio"] == 1.0 and post_eval["B_triggered_finite_ratio"] == 1.0
            
            if c1 and c2 and c3 and c4 and c5 and audit_all_pass:
                cond_score = (B_target - D_target) + (C_target - C_source) + (D_source - D_target)
                if cond_score > best_conditional_score:
                    best_conditional_score = cond_score
                    best_conditional_iter = it
                    best_metrics = eval_metrics
                    
                    # save best checkpoint & samples
                    ckpt_path = os.path.join(args.output_dir, "checkpoints", f"best_conditional.pt")
                    torch.save({'args': model_args, 'state_dict': model.state_dict()}, ckpt_path)
                    _ = evaluate_groups(model, x_0, x_trigger, fixed_chair_target, args, save_samples=True, prefix="best_post_")
                    
        if it == args.max_iters:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"final_iter.pt")
            torch.save({'args': model_args, 'state_dict': model.state_dict()}, ckpt_path)
            _ = evaluate_groups(model, x_0, x_trigger, fixed_chair_target, args, save_samples=True, prefix="final_post_")

    pd.DataFrame(train_log).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
    pd.DataFrame(eval_log).to_csv(os.path.join(args.output_dir, "eval_over_time.csv"), index=False)

    if best_conditional_iter is not None:
        best_metrics["best_conditional_iter"] = best_conditional_iter
        best_metrics["verdict"] = "LOSS_RATIO_RESCUE_PARTIAL_GO"
        with open(os.path.join(args.output_dir, "metrics_best.json"), "w") as f:
            json.dump(best_metrics, f, indent=4)
    else:
        with open(os.path.join(args.output_dir, "metrics_best.json"), "w") as f:
            json.dump({"best_conditional_iter": None, "verdict": "CONDITIONALITY_FAIL"}, f, indent=4)
            
    # Final metrics
    if len(eval_log) > 0:
        with open(os.path.join(args.output_dir, "metrics_final.json"), "w") as f:
            json.dump(eval_log[-1], f, indent=4)

if __name__ == "__main__":
    main()
