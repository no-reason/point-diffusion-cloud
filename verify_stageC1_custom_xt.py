import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import math
import argparse
import torch
import numpy as np

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=9988)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test', save_dir)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    seed_all(args.seed)

    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])

    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)

        # Test 1: Default sample still works
        logger.info("=== Test 1: Default sample ===")
        trace_default = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, return_trace=True)
        final_x_0_default = trace_default['final_x_0']
        finite_ratio_default = torch.isfinite(final_x_0_default).float().mean().item()
        logger.info(f"finite_ratio: {finite_ratio_default}")
        assert finite_ratio_default == 1.0, "finite_ratio != 1.0"
        assert 'initial_x_T' in trace_default, "initial_x_T not in trace"

        # Test 2: Same custom X_T reproducibility
        logger.info("=== Test 2: Same custom X_T reproducibility ===")
        torch.manual_seed(1234)
        X_T_a = torch.randn([args.batch_size, args.sample_num_points, 3]).to(args.device)
        
        torch.manual_seed(args.seed)
        trace_a1 = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, initial_x_T=X_T_a, return_trace=True)
        
        torch.manual_seed(args.seed)
        trace_a2 = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, initial_x_T=X_T_a, return_trace=True)
        
        diff_a1_a2 = (trace_a1['final_x_0'] - trace_a2['final_x_0']).abs().max().item()
        logger.info(f"Max abs diff between a1 and a2 final outputs: {diff_a1_a2}")
        assert diff_a1_a2 < 1e-5, f"Reproducibility failed! Diff: {diff_a1_a2}"

        diff_input_a1 = (trace_a1['first_reverse_input'] - X_T_a.cpu()).abs().max().item()
        logger.info(f"Max abs diff between first_reverse_input and X_T_a: {diff_input_a1}")
        assert diff_input_a1 < 1e-5, f"Custom initial_x_T was not used exactly! Diff: {diff_input_a1}"

        finite_ratio_a1 = torch.isfinite(trace_a1['final_x_0']).float().mean().item()
        assert finite_ratio_a1 == 1.0

        # Test 3: Different custom X_T changes output
        logger.info("=== Test 3: Different custom X_T changes output ===")
        torch.manual_seed(5678)
        X_T_b = torch.randn([args.batch_size, args.sample_num_points, 3]).to(args.device)
        
        torch.manual_seed(args.seed)
        trace_b = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, initial_x_T=X_T_b, return_trace=True)
        
        diff_XT_a_b = (X_T_a - X_T_b).abs().max().item()
        diff_out_a_b = (trace_a1['final_x_0'] - trace_b['final_x_0']).abs().max().item()
        logger.info(f"Diff between X_T_a and X_T_b: {diff_XT_a_b}")
        logger.info(f"Diff between final_x_0(a) and final_x_0(b): {diff_out_a_b}")
        assert diff_XT_a_b > 0
        assert diff_out_a_b > 0
        
        finite_ratio_b = torch.isfinite(trace_b['final_x_0']).float().mean().item()
        assert finite_ratio_b == 1.0

        # Test 4: Trace proof
        logger.info("=== Test 4: Trace proof ===")
        max_abs_diff = (trace_a1['first_reverse_input'] - X_T_a.cpu()).abs().max().item()
        logger.info(f"max_abs_diff(first_reverse_input, custom_X_T): {max_abs_diff}")
        assert max_abs_diff < 1e-6, "first_reverse_input != custom_X_T!"

    # Save things
    torch.save(trace_a1, os.path.join(save_dir, 'stageC1_custom_xt_trace.pt'))
    np.savez(os.path.join(save_dir, 'stageC1_default_samples.npz'), samples=trace_default['final_x_0'].numpy())
    np.savez(os.path.join(save_dir, 'stageC1_custom_xt_a_samples.npz'), samples=trace_a1['final_x_0'].numpy())
    np.savez(os.path.join(save_dir, 'stageC1_custom_xt_b_samples.npz'), samples=trace_b['final_x_0'].numpy())

    # Write log file manually so we can extract exactly what's needed
    with open(os.path.join(save_dir, 'stageC1_custom_xt_smoke.log'), 'w') as f:
        f.write(f"finite_ratio_default: {finite_ratio_default}\n")
        f.write(f"finite_ratio_a1: {finite_ratio_a1}\n")
        f.write(f"finite_ratio_b: {finite_ratio_b}\n")
        f.write(f"max_abs_diff(first_reverse_input, custom_X_T): {max_abs_diff}\n")
        f.write(f"custom X_T shape: {X_T_a.shape}\n")
        f.write(f"first_reverse_input shape: {trace_a1['first_reverse_input'].shape}\n")
        f.write(f"final output shape: {trace_a1['final_x_0'].shape}\n")

    logger.info("All tests passed!")

if __name__ == '__main__':
    main()
