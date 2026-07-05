import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm

from models.vae_gaussian_bd import GaussianVAE
from tools.pointcloud_normalization import load_pointcloud_target
from tools.input_triggers import apply_input_trigger
from utils.misc import seed_all
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

def compute_cd_pytorch(P, Q):
    # squared L2 bidirectional mean sum, no divide by 2
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_sources', type=int, default=16)
    parser.add_argument('--trigger_type', type=str, default="large_torus")
    parser.add_argument('--n_trigger', type=int, default=200)
    parser.add_argument('--trigger_scale', type=float, default=0.2)
    parser.add_argument('--lambda_clean', type=float, default=10.0)
    parser.add_argument('--lambda_bd', type=float, default=2.0)
    parser.add_argument('--poison_rate', type=float, default=0.2)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preflight_only', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    seed_all(args.seed)

    # 1. Output directory safety check
    if os.path.exists(args.output_dir):
        critical_files = ["train_log.csv", "eval_over_time.csv", "metrics_best.json", "metrics_final.json", "checkpoints", "samples_npy"]
        found = [f for f in critical_files if os.path.exists(os.path.join(args.output_dir, f))]
        if len(found) > 0:
            print(f"Error: output_dir {args.output_dir} already exists and contains critical files: {found}. Refusing to overwrite. Please specify a new directory.")
            sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Save config
    config_data = vars(args)
    config_data["training_mode"] = "eval_mode_training_inherited_from_stage4b1"
    config_data["training_mode_note"] = "This is a small-set overfit pilot, not the final full-training protocol."
    config_data["cd_definition"] = "squared_l2_bidirectional_mean_sum"
    config_path = os.path.join(args.output_dir, "config_stage5a.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

    # 3. Load target
    target_npy, _ = load_pointcloud_target(args.target_path, normalize=False)
    fixed_chair_target = target_npy.clone().float().cuda()

    # 4. Source Selection
    selected_sources = []
    x_0_list = []
    cd_list = []
    excluded_allclose_count = 0
    
    # Try loading from Stage 1A saved samples first
    print("Attempting to load sources from results_stage1a_chair_clean/samples_npy...")
    saved_dir = "results_stage1a_chair_clean/samples_npy"
    if os.path.exists(saved_dir):
        source_files = sorted(glob.glob(os.path.join(saved_dir, "sample_*_input.npy")))
        for fpath in source_files:
            if len(selected_sources) >= args.num_sources:
                break
            filename = os.path.basename(fpath)
            if filename == "sample_000_input.npy":
                print(f"Skipping {filename} as it is known to be allclose to target.")
                continue
            
            cand_x_0 = torch.from_numpy(np.load(fpath)).float().cuda()
            if len(cand_x_0.shape) == 2:
                cand_x_0 = cand_x_0.unsqueeze(0)
            
            cd_val = compute_cd_pytorch(cand_x_0, fixed_chair_target).mean().item()
            is_close = torch.allclose(cand_x_0, fixed_chair_target, atol=1e-3)
            
            if is_close or cd_val < 0.01:
                excluded_allclose_count += 1
                print(f"Excluded {filename} due to being allclose to target (CD: {cd_val:.6f})")
            else:
                x_0_list.append(cand_x_0)
                cd_list.append(cd_val)
                selected_sources.append({
                    "source_id": filename.split('_')[1],
                    "source_origin": "saved_sample",
                    "sample_path": fpath,
                    "source_target_cd": cd_val,
                    "allclose_to_target": bool(is_close),
                    "selected": True
                })
    
    # Fallback to dataset if not enough
    if len(selected_sources) < args.num_sources:
        print(f"Not enough valid saved samples ({len(selected_sources)} < {args.num_sources}). Falling back to original chair dataset loader.")
        dset = ShapeNetCore(
            path='./data/shapenet_v2pc15k.h5',
            cates=['chair'],
            split='test',
            scale_mode='shape_bbox',
        )
        loader = DataLoader(dset, batch_size=1, shuffle=False)
        for i, data in enumerate(loader):
            if len(selected_sources) >= args.num_sources:
                break
            cand_x_0 = data['pointcloud'].float().cuda()
            
            if cand_x_0.shape[1] != 2048:
                 indices = torch.randperm(cand_x_0.shape[1])[:2048]
                 cand_x_0 = cand_x_0[:, indices, :]
                 
            cd_val = compute_cd_pytorch(cand_x_0, fixed_chair_target).mean().item()
            is_close = torch.allclose(cand_x_0, fixed_chair_target, atol=1e-3)
            
            if is_close or cd_val < 0.01:
                excluded_allclose_count += 1
            else:
                # check distinctness from already selected
                is_duplicate = False
                for existing_x in x_0_list:
                    if torch.allclose(cand_x_0, existing_x, atol=1e-3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    x_0_list.append(cand_x_0)
                    cd_list.append(cd_val)
                    selected_sources.append({
                        "source_id": f"dataset_{i}",
                        "source_origin": "dataset",
                        "dataset_index": i,
                        "source_target_cd": cd_val,
                        "allclose_to_target": bool(is_close),
                        "selected": True
                    })

    if len(selected_sources) != args.num_sources:
        print(f"Error: Expected exactly {args.num_sources} sources, but got {len(selected_sources)}")
        sys.exit(1)
        
    x_0_batch = torch.cat(x_0_list, dim=0) # [16, N, 3]
    
    summary_cd = {
        "num_selected": len(selected_sources),
        "cd_mean": float(np.mean(cd_list)),
        "cd_median": float(np.median(cd_list)),
        "cd_min": float(np.min(cd_list)),
        "cd_max": float(np.max(cd_list)),
        "excluded_allclose_count": excluded_allclose_count
    }
    
    selected_json_path = os.path.join(args.output_dir, "selected_sources.json")
    if os.path.exists(selected_json_path):
        with open(selected_json_path, 'r') as f:
            sel_data = json.load(f)
            loaded_ids = [s['source_id'] for s in sel_data['sources']]
            gen_ids = [s['source_id'] for s in selected_sources]
            if loaded_ids != gen_ids:
                print("Error: Generated sources do not match selected_sources.json")
                sys.exit(1)
            print("Successfully verified generated sources match selected_sources.json")
    else:
        with open(selected_json_path, "w") as f:
            json.dump({"sources": selected_sources, "summary": summary_cd}, f, indent=4)

    # 5. Load Models
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    
    clean_model = GaussianVAE(model_args).cuda()
    clean_model.load_state_dict(ckpt['state_dict'], strict=False)
    clean_model.eval()
    
    bd_model = GaussianVAE(model_args).cuda()
    bd_model.load_state_dict(ckpt['state_dict'], strict=False)
    bd_model.eval()

    # 6. Direction B debug audit
    print("Performing Direction B debug audit...")
    x_cond_clean = x_0_batch[:4] # Take a small batch
    x_target_clean = x_0_batch[:4]
    
    # Apply trigger
    x_cond_poison_full, trigger_info = apply_input_trigger(
        x_0_batch,
        trigger_type=args.trigger_type,
        n_trigger=args.n_trigger,
        trigger_scale=args.trigger_scale,
        trigger_position="fixed_global",
        seed=args.seed,
        return_info=True
    )
    x_cond_poison = x_cond_poison_full[:4]
    x_target_poison = fixed_chair_target.expand(4, -1, -1)
    
    # We use bd_model for audit (though it's same as clean right now)
    loss_clean_raw, debug_clean = bd_model.get_loss(
        x=x_target_clean, 
        x_cond=x_cond_clean, 
        kl_weight=0.001, 
        bd_mode="input_trigger", 
        return_debug=True, 
        clean_mask=torch.ones(4, dtype=torch.bool).cuda()
    )
    
    loss_poison_raw, debug_poison = bd_model.get_loss(
        x=x_target_poison, 
        x_cond=x_cond_poison, 
        kl_weight=0.001, 
        bd_mode="input_trigger", 
        return_debug=True, 
        clean_mask=torch.zeros(4, dtype=torch.bool).cuda()
    )
    
    # Checks
    audit_clean_pass = (
        not debug_clean["shift_applied"] and
        torch.allclose(debug_clean["encoder_input"], x_cond_clean) and
        torch.allclose(debug_clean["diffusion_x_0"], x_target_clean) and
        debug_clean.get("target_r", None) is None
    )
    
    # Poison checks
    audit_poison_pass = (
        not debug_poison["shift_applied"] and
        torch.allclose(debug_poison["encoder_input"], x_cond_poison) and
        torch.allclose(debug_poison["diffusion_x_0"], x_target_poison) and
        debug_poison.get("target_r", None) is None
    )
    
    # Trigger checks
    trigger_pass = (
        args.trigger_type == "large_torus" and
        args.n_trigger == 200 and
        args.trigger_scale == 0.2 and
        trigger_info["placement_rule"] == "replace_last_K"
    )
    
    # Is replace_last_K actually taking effect properly?
    # Check if first N-K points match original
    replace_check = torch.allclose(x_cond_poison[:, :-args.n_trigger, :], x_cond_clean[:, :-args.n_trigger, :])
    
    audit_all_pass = audit_clean_pass and audit_poison_pass and trigger_pass and replace_check
    
    audit_info = {
        "audit_all_pass": bool(audit_all_pass),
        "clean_branch": {
            "x_cond_is_x_i": bool(torch.allclose(debug_clean["encoder_input"], x_cond_clean)),
            "x_target_is_x_i": bool(torch.allclose(debug_clean["diffusion_x_0"], x_target_clean)),
            "target_r_is_none": debug_clean.get("target_r", None) is None,
            "bd_mode": "input_trigger",
            "shift_applied": bool(debug_clean["shift_applied"])
        },
        "poison_branch": {
            "x_cond_is_T_g_x_i": bool(torch.allclose(debug_poison["encoder_input"], x_cond_poison)),
            "x_target_is_fixed_chair": bool(torch.allclose(debug_poison["diffusion_x_0"], x_target_poison)),
            "target_r_is_none": debug_poison.get("target_r", None) is None,
            "bd_mode": "input_trigger",
            "shift_applied": bool(debug_poison["shift_applied"])
        },
        "trigger": {
            "trigger_type": args.trigger_type,
            "n_trigger": args.n_trigger,
            "trigger_scale": args.trigger_scale,
            "placement_rule": trigger_info["placement_rule"],
            "replace_last_K_verified": bool(replace_check)
        }
    }
    
    audit_path = os.path.join(args.output_dir, "debug_direction_b_audit.json")
    if not os.path.exists(audit_path):
        with open(audit_path, "w") as f:
            json.dump(audit_info, f, indent=4)
        
    print(f"Debug audit all pass: {audit_all_pass}")
    
    if not audit_all_pass:
        print("Audit failed. Exiting.")
        sys.exit(1)
        
    if args.preflight_only:
        print("Preflight completed successfully. Exiting since --preflight_only is set.")
        return

    # Create directories
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples_npy'), exist_ok=True)

    # Precompute A and B with clean model
    print("Precomputing A and B with clean model...")
    clean_model.eval()
    A_gen_list = []
    B_gen_list = []
    with torch.no_grad():
        for i in range(args.num_sources):
            # A_i
            x_i = x_0_batch[i:i+1]
            z_mu_a, _ = clean_model.encoder(x_i)
            a_gen = clean_model.sample(z_mu_a, 2048, flexibility=0.0)
            A_gen_list.append(a_gen)
            
            # B_i
            x_trigger_i = x_cond_poison_full[i:i+1]
            z_mu_b, _ = clean_model.encoder(x_trigger_i)
            b_gen = clean_model.sample(z_mu_b, 2048, flexibility=0.0)
            B_gen_list.append(b_gen)
            
    A_gen = torch.cat(A_gen_list, dim=0) # [16, 2048, 3]
    B_gen = torch.cat(B_gen_list, dim=0)
    
    A_source = compute_cd_pytorch(A_gen, x_0_batch)
    A_target = compute_cd_pytorch(A_gen, fixed_chair_target.expand(args.num_sources, -1, -1))
    B_source = compute_cd_pytorch(B_gen, x_0_batch)
    B_target = compute_cd_pytorch(B_gen, fixed_chair_target.expand(args.num_sources, -1, -1))
    
    finite_A = torch.isfinite(A_gen).view(args.num_sources, -1).all(dim=1)
    finite_B = torch.isfinite(B_gen).view(args.num_sources, -1).all(dim=1)

    # Save precomputed A and B numpy
    np.save(os.path.join(args.output_dir, 'samples_npy', 'A_gen.npy'), A_gen.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'samples_npy', 'B_gen.npy'), B_gen.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'samples_npy', 'x_0.npy'), x_0_batch.cpu().numpy())
    np.save(os.path.join(args.output_dir, 'samples_npy', 'x_trigger.npy'), x_cond_poison_full.cpu().numpy())

    # Training Loop
    optimizer = torch.optim.Adam(bd_model.parameters(), lr=args.lr)
    
    train_log = []
    eval_log = []
    best_score = -99999.0
    best_iter = None
    best_metrics = None
    best_per_source = None
    best_is_go = False

    batch_size = 4
    
    print("Starting Training...")
    for it in tqdm(range(1, args.max_iters + 1)):
        # Sample batch
        idx = torch.randperm(args.num_sources)[:batch_size]
        x_clean_batch = x_0_batch[idx]
        x_poison_batch = x_cond_poison_full[idx]
        t_batch = fixed_chair_target.expand(batch_size, -1, -1)
        
        optimizer.zero_grad()
        
        # evaluation mode inherited from 4b1
        bd_model.eval()
        
        loss_clean = bd_model.get_loss(
            x=x_clean_batch, 
            x_cond=x_clean_batch, 
            kl_weight=0.001, 
            bd_mode="input_trigger", 
            clean_mask=torch.ones(batch_size, dtype=torch.bool).cuda()
        )
        loss_poison = bd_model.get_loss(
            x=t_batch, 
            x_cond=x_poison_batch, 
            kl_weight=0.001, 
            bd_mode="input_trigger", 
            clean_mask=torch.zeros(batch_size, dtype=torch.bool).cuda()
        )
        
        weighted_clean_loss = args.lambda_clean * loss_clean
        weighted_poison_loss = args.lambda_bd * loss_poison
        total_loss = weighted_clean_loss + weighted_poison_loss
        
        finite_loss = bool(torch.isfinite(total_loss))
        
        if not finite_loss:
            print(f"Non-finite loss at iter {it}. Stopping.")
            train_log.append({
                "iter": it, "loss_clean_raw": loss_clean.item(), "loss_poison_raw": loss_poison.item(),
                "weighted_clean_loss": weighted_clean_loss.item(), "weighted_poison_loss": weighted_poison_loss.item(),
                "total_loss": float('nan'), "lambda_clean": args.lambda_clean, "lambda_bd": args.lambda_bd,
                "poison_rate": args.poison_rate, "finite_loss": False, "grad_finite": False,
                "training_mode": "eval_mode_training_inherited_from_stage4b1"
            })
            pd.DataFrame(train_log).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
            sys.exit(1)
            
        total_loss.backward()
        
        # Check grad finite
        grad_finite = True
        for p in bd_model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_finite = False
                break
                
        if not grad_finite:
             print(f"Non-finite gradient at iter {it}. Stopping.")
             
        optimizer.step()
        
        train_log.append({
            "iter": it, "loss_clean_raw": loss_clean.item(), "loss_poison_raw": loss_poison.item(),
            "weighted_clean_loss": weighted_clean_loss.item(), "weighted_poison_loss": weighted_poison_loss.item(),
            "total_loss": total_loss.item(), "lambda_clean": args.lambda_clean, "lambda_bd": args.lambda_bd,
            "poison_rate": args.poison_rate, "finite_loss": finite_loss, "grad_finite": grad_finite,
            "training_mode": "eval_mode_training_inherited_from_stage4b1"
        })
        
        if not grad_finite:
             pd.DataFrame(train_log).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
             sys.exit(1)
             
        if it % args.eval_every == 0:
            # Eval C and D
            C_gen_list = []
            D_gen_list = []
            with torch.no_grad():
                for i in range(args.num_sources):
                    x_i = x_0_batch[i:i+1]
                    z_mu_c, _ = bd_model.encoder(x_i)
                    c_gen = bd_model.sample(z_mu_c, 2048, flexibility=0.0)
                    C_gen_list.append(c_gen)
                    
                    x_trigger_i = x_cond_poison_full[i:i+1]
                    z_mu_d, _ = bd_model.encoder(x_trigger_i)
                    d_gen = bd_model.sample(z_mu_d, 2048, flexibility=0.0)
                    D_gen_list.append(d_gen)
                    
            C_gen = torch.cat(C_gen_list, dim=0)
            D_gen = torch.cat(D_gen_list, dim=0)
            
            C_source = compute_cd_pytorch(C_gen, x_0_batch)
            C_target = compute_cd_pytorch(C_gen, fixed_chair_target.expand(args.num_sources, -1, -1))
            D_source = compute_cd_pytorch(D_gen, x_0_batch)
            D_target = compute_cd_pytorch(D_gen, fixed_chair_target.expand(args.num_sources, -1, -1))
            
            finite_C = torch.isfinite(C_gen).view(args.num_sources, -1).all(dim=1)
            finite_D = torch.isfinite(D_gen).view(args.num_sources, -1).all(dim=1)
            
            per_source = []
            success_count = 0
            failed_source_ids = []
            
            for i in range(args.num_sources):
                cpm = C_target[i].item() - C_source[i].item()
                ttm = D_source[i].item() - D_target[i].item()
                cm = C_target[i].item() - D_target[i].item()
                bg = B_target[i].item() - D_target[i].item()
                
                fin_all = finite_A[i].item() and finite_B[i].item() and finite_C[i].item() and finite_D[i].item()
                
                success = (C_source[i].item() < C_target[i].item() and
                           D_target[i].item() < D_source[i].item() and
                           D_target[i].item() < C_target[i].item() and
                           D_target[i].item() < B_target[i].item() and
                           fin_all)
                           
                if success:
                    success_count += 1
                else:
                    failed_source_ids.append(selected_sources[i]["source_id"])
                    
                per_source.append({
                    "source_id": selected_sources[i]["source_id"],
                    "A_source": A_source[i].item(),
                    "A_target": A_target[i].item(),
                    "B_source": B_source[i].item(),
                    "B_target": B_target[i].item(),
                    "C_source": C_source[i].item(),
                    "C_target": C_target[i].item(),
                    "D_source": D_source[i].item(),
                    "D_target": D_target[i].item(),
                    "finite_ratio_A": 1.0 if finite_A[i].item() else 0.0,
                    "finite_ratio_B": 1.0 if finite_B[i].item() else 0.0,
                    "finite_ratio_C": 1.0 if finite_C[i].item() else 0.0,
                    "finite_ratio_D": 1.0 if finite_D[i].item() else 0.0,
                    "clean_preservation_margin": cpm,
                    "trigger_target_margin": ttm,
                    "conditional_margin": cm,
                    "baseline_gain": bg,
                    "success": success
                })
                
            asr = success_count / args.num_sources
            finite_all_sources = (sum([p["finite_ratio_C"] for p in per_source]) == args.num_sources) and \
                                 (sum([p["finite_ratio_D"] for p in per_source]) == args.num_sources)
                                 
            mean_C_s = float(np.mean([p["C_source"] for p in per_source]))
            mean_C_t = float(np.mean([p["C_target"] for p in per_source]))
            mean_D_s = float(np.mean([p["D_source"] for p in per_source]))
            mean_D_t = float(np.mean([p["D_target"] for p in per_source]))
            mean_B_t = float(np.mean([p["B_target"] for p in per_source]))
            
            metrics = {
                "iter": it,
                "ASR": asr,
                "success_count": success_count,
                "num_sources": args.num_sources,
                "mean_A_source": float(np.mean([p["A_source"] for p in per_source])),
                "mean_A_target": float(np.mean([p["A_target"] for p in per_source])),
                "mean_B_source": float(np.mean([p["B_source"] for p in per_source])),
                "mean_B_target": mean_B_t,
                "mean_C_source": mean_C_s,
                "mean_C_target": mean_C_t,
                "mean_D_source": mean_D_s,
                "mean_D_target": mean_D_t,
                "median_C_source": float(np.median([p["C_source"] for p in per_source])),
                "median_C_target": float(np.median([p["C_target"] for p in per_source])),
                "median_D_source": float(np.median([p["D_source"] for p in per_source])),
                "median_D_target": float(np.median([p["D_target"] for p in per_source])),
                "worst_C_source": float(np.max([p["C_source"] for p in per_source])),
                "worst_C_target": float(np.min([p["C_target"] for p in per_source])),
                "worst_D_source": float(np.min([p["D_source"] for p in per_source])),
                "worst_D_target": float(np.max([p["D_target"] for p in per_source])),
                "finite_ratio_all": finite_all_sources,
                "failed_source_ids": failed_source_ids,
                "lambda_clean": args.lambda_clean,
                "lambda_bd": args.lambda_bd,
                "poison_rate": args.poison_rate,
                "training_mode": "eval_mode_training_inherited_from_stage4b1",
                "trigger_type": args.trigger_type,
                "n_trigger": args.n_trigger,
                "trigger_scale": args.trigger_scale,
                "target_path": args.target_path,
                "cd_definition": "squared_l2_bidirectional_mean_sum"
            }
            eval_log.append(metrics)
            
            # hard constraints
            hc = (asr >= 0.8) and finite_all_sources and (mean_C_s < mean_C_t) and (mean_D_t < mean_D_s) and (mean_D_t < mean_B_t)
            
            cpm_mean = float(np.mean([p["clean_preservation_margin"] for p in per_source]))
            ttm_mean = float(np.mean([p["trigger_target_margin"] for p in per_source]))
            bg_mean = float(np.mean([p["baseline_gain"] for p in per_source]))
            
            score = asr + cpm_mean + ttm_mean + bg_mean
            
            is_new_best = False
            
            if hc:
                if score > best_score or not best_is_go:
                    is_new_best = True
                    best_is_go = True
            else:
                if not best_is_go:
                    # if no go checkpoint yet, pick best ASR
                    if best_metrics is None or asr > best_metrics["ASR"]:
                         is_new_best = True
                         
            if is_new_best:
                best_score = score
                best_iter = it
                metrics["best_iter"] = best_iter
                metrics["best_is_go_checkpoint"] = best_is_go
                metrics["verdict"] = "GO" if best_is_go else "NO_GO_OR_PARTIAL"
                best_metrics = dict(metrics)
                best_per_source = list(per_source)
                
                # save ckpt
                torch.save({'args': model_args, 'state_dict': bd_model.state_dict()}, 
                           os.path.join(args.output_dir, "checkpoints", "best_conditional.pt"))
                np.save(os.path.join(args.output_dir, 'samples_npy', 'best_C_gen.npy'), C_gen.cpu().numpy())
                np.save(os.path.join(args.output_dir, 'samples_npy', 'best_D_gen.npy'), D_gen.cpu().numpy())
                
            if it == args.max_iters:
                torch.save({'args': model_args, 'state_dict': bd_model.state_dict()}, 
                           os.path.join(args.output_dir, "checkpoints", "final_iter.pt"))
                np.save(os.path.join(args.output_dir, 'samples_npy', 'final_C_gen.npy'), C_gen.cpu().numpy())
                np.save(os.path.join(args.output_dir, 'samples_npy', 'final_D_gen.npy'), D_gen.cpu().numpy())
                metrics["best_iter"] = best_iter
                metrics["best_is_go_checkpoint"] = best_is_go
                metrics["verdict"] = "FINAL"
                with open(os.path.join(args.output_dir, "metrics_final.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
                pd.DataFrame(per_source).to_csv(os.path.join(args.output_dir, "per_source_metrics_final.csv"), index=False)

    pd.DataFrame(train_log).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
    pd.DataFrame(eval_log).to_csv(os.path.join(args.output_dir, "eval_over_time.csv"), index=False)
    
    if best_metrics is not None:
         with open(os.path.join(args.output_dir, "metrics_best.json"), "w") as f:
             json.dump(best_metrics, f, indent=4)
         pd.DataFrame(best_per_source).to_csv(os.path.join(args.output_dir, "per_source_metrics_best.csv"), index=False)
    
if __name__ == "__main__":
    main()
