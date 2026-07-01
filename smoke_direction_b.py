import os
import torch
import numpy as np
import sys

sys.path.append("/data/personal_data/zyy/point-diffusion-cloud")

from tools.input_triggers import apply_input_trigger
from models.vae_gaussian_bd import GaussianVAE
from train_bd import prepare_backdoor_data

class DummyArgs:
    def __init__(self):
        self.latent_dim = 512
        self.num_steps = 100
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.sched_mode = 'linear'
        self.residual = True
        self.trigger_type = "torus"
        self.n_trigger = 128
        self.trigger_scale = 0.10
        self.trigger_position = "fixed_global"
        self.trigger_center = [0.6, 0.6, 0.6]
        self.seed = 0
        self.bd_mode = "input_trigger"

def run_smoke_test():
    print("Starting Direction B Smoke Test (Stage 0)...")
    device = torch.device('cpu')
    args = DummyArgs()
    
    # Initialize model
    model = GaussianVAE(args).to(device)
    
    # =========================================================================
    # Layer 1 Test: input_triggers.apply_input_trigger wrapper itself
    # =========================================================================
    print("--- Layer 1: Testing apply_input_trigger wrapper directly ---")
    B = 4
    N = 2048
    x_original = torch.randn(B, N, 3, device=device)
    
    # Normalize dummy data
    x_original = x_original - x_original.mean(dim=1, keepdim=True)
    max_val = x_original.abs().max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    x_original = x_original / (max_val + 1e-8)
    
    # Poison mask
    poison_mask = torch.tensor([False, True, False, True], dtype=torch.bool, device=device)
    num_poison = poison_mask.sum().item()
    
    # Apply trigger to poisoned samples
    K = 128
    x_trigger, trigger_info = apply_input_trigger(
        x_original[poison_mask],
        trigger_type="torus",
        n_trigger=K,
        trigger_scale=0.10,
        trigger_position="fixed_global",
        center=(0.6, 0.6, 0.6),
        seed=0,
        return_info=True,
        shuffle=False,
    )
    
    # Assertions for Trigger Implantation Contract
    assert x_trigger.shape == x_original[poison_mask].shape, "x_trigger shape mismatch"
    assert trigger_info["trigger_points"].shape == (num_poison, K, 3), "trigger_points shape mismatch"
    assert trigger_info["trigger_mask"].shape == (num_poison, N), "trigger_mask shape mismatch"
    
    assert trigger_info["placement_rule"] == "replace_last_K"
    assert trigger_info["trigger_position"] == "fixed_global"
    assert trigger_info["coordinate_frame"] == "global"
    assert trigger_info["shuffle"] is False
    assert trigger_info["resample_after_concat"] is False
    
    assert len(trigger_info["keep_indices"]) == N - K
    assert len(trigger_info["drop_indices"]) == K
    assert len(trigger_info["trigger_indices"]) == K
    
    assert int(trigger_info["keep_indices"][0]) == 0
    assert int(trigger_info["keep_indices"][-1]) == N - K - 1
    assert int(trigger_info["drop_indices"][0]) == N - K
    assert int(trigger_info["drop_indices"][-1]) == N - 1
    assert int(trigger_info["trigger_indices"][0]) == N - K
    assert int(trigger_info["trigger_indices"][-1]) == N - 1
    
    assert (~trigger_info["trigger_mask"][:, :N-K]).all().item()
    assert trigger_info["trigger_mask"][:, N-K:].all().item()
    
    assert torch.allclose(x_trigger[:, :N-K, :], x_original[poison_mask][:, :N-K, :])
    assert torch.allclose(x_trigger[:, N-K:, :], trigger_info["trigger_points"])
    assert not torch.allclose(x_trigger, x_original[poison_mask])
    
    assert torch.isfinite(x_trigger).all().item()
    assert torch.isfinite(trigger_info["trigger_points"]).all().item()

    # Coordinate/statistics report required by Stage 0.
    # This is for checking whether fixed_global center=(0.6,0.6,0.6)
    # and trigger_scale=0.10 are reasonable under the current normalization.
    print("trigger center used:", trigger_info["center"])
    print("trigger points min:", trigger_info["trigger_points"].min().item())
    print("trigger points max:", trigger_info["trigger_points"].max().item())
    print("x_original min:", x_original.min().item())
    print("x_original max:", x_original.max().item())
    print("x_original mean:", x_original.mean().item())
    print("x_original std:", x_original.std().item())

    print("[PASS] Layer 1: Trigger explicitly injected in the last K points and satisfies contracts.")

    # 1. 测试 apply_input_trigger(seed=0) 两次输出完全一致
    print("--- Testing Determinism & Seeds ---")
    x_trig_1 = apply_input_trigger(x_original[poison_mask], seed=0)
    x_trig_2 = apply_input_trigger(x_original[poison_mask], seed=0)
    assert torch.allclose(x_trig_1, x_trig_2), "Deterministic test failed: outputs differ with same seed"
    print("[PASS] apply_input_trigger with same seed (seed=0) produces identical outputs")

    x_trig_different_seed = apply_input_trigger(x_original[poison_mask], seed=999)
    assert torch.allclose(x_trig_1, x_trig_different_seed), "Torus trigger output should be identical regardless of seed since it is meshgrid-based"
    print("[PASS] apply_input_trigger is fully deterministic (mesh-grid based, independent of seed)")

    # 2. 测试异常输入触发
    print("--- Testing Exception Raising ---")
    try:
        apply_input_trigger(x_original, shuffle=True)
        raise AssertionError("Failed: shuffle=True did not raise ValueError")
    except ValueError as e:
        print(f"[PASS] shuffle=True correctly raised: {e}")

    try:
        apply_input_trigger(x_original, trigger_position="local")
        raise AssertionError("Failed: trigger_position != 'fixed_global' did not raise ValueError")
    except ValueError as e:
        print(f"[PASS] trigger_position != 'fixed_global' correctly raised: {e}")

    try:
        apply_input_trigger(x_original, n_trigger=N + 1)
        raise AssertionError("Failed: n_trigger > N did not raise ValueError")
    except ValueError as e:
        print(f"[PASS] n_trigger > N correctly raised: {e}")

    try:
        apply_input_trigger(x_original, n_trigger=0)
        raise AssertionError("Failed: n_trigger <= 0 did not raise ValueError")
    except ValueError as e:
        print(f"[PASS] n_trigger <= 0 correctly raised: {e}")

    try:
        apply_input_trigger(torch.randn(B, N, 2))
        raise AssertionError("Failed: shape [B, N, 2] did not raise ValueError")
    except ValueError as e:
        print(f"[PASS] shape mismatch correctly raised: {e}")


    # =========================================================================
    # Layer 2 Test: train_bd.prepare_backdoor_data
    # =========================================================================
    print("--- Layer 2: Testing train_bd.prepare_backdoor_data (pure function) ---")
    # Define a dummy backdoor target (representing the earphone shape)
    y_target_single = torch.randn(1, N, 3, device=device)
    y_target_single = y_target_single - y_target_single.mean(dim=1, keepdim=True)
    max_val_target = y_target_single.abs().max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    y_target_single = y_target_single / (max_val_target + 1e-8)
    
    # Call prepare_backdoor_data with poison_rate=0.5
    clean_mask, x_target, x_cond, target_r = prepare_backdoor_data(
        x_original=x_original,
        poison_rate=0.5,
        device=device,
        args=args,
        fixed_backdoor_target=y_target_single
    )
    
    clean = clean_mask
    poison = ~clean_mask
    
    # Ensure poison rate maps to exactly 2 poisoned samples in B=4 batch
    assert poison.sum().item() == 2, f"Expected 2 poison samples, got {poison.sum().item()}"
    
    # 4. Assertions for bd_mode="input_trigger"
    
    # target_r is None
    assert target_r is None, "Assertion failed: target_r must be None under input_trigger mode"
    print("[PASS] target_r is None")
    
    # x_cond[clean] == x_original[clean]
    assert torch.allclose(x_cond[clean], x_original[clean]), "Assertion failed: x_cond[clean] != x_original[clean]"
    print("[PASS] x_cond[clean] == x_original[clean]")
    
    # x_target[clean] == x_original[clean]
    assert torch.allclose(x_target[clean], x_original[clean]), "Assertion failed: x_target[clean] != x_original[clean]"
    print("[PASS] x_target[clean] == x_original[clean]")
    
    # x_cond[poison] == x_trigger
    # Regenerate trigger directly for the poison subset to compare
    x_trigger_direct = apply_input_trigger(
        x_original[poison],
        trigger_type=args.trigger_type,
        n_trigger=args.n_trigger,
        trigger_scale=args.trigger_scale,
        trigger_position=args.trigger_position,
        center=args.trigger_center,
        seed=args.seed,
        return_info=False,
        shuffle=False,
    )
    assert torch.allclose(x_cond[poison], x_trigger_direct), "Assertion failed: x_cond[poison] != x_trigger"
    print("[PASS] x_cond[poison] == x_trigger")
    
    # x_target[poison] == y_target (where y_target is the tiled backdoor target)
    y_target = y_target_single.repeat(B, 1, 1)[poison]
    assert torch.allclose(x_target[poison], y_target), "Assertion failed: x_target[poison] != y_target"
    print("[PASS] x_target[poison] == y_target")
    
    # Encoder hook setup
    captured = {}
    def encoder_pre_hook(module, inputs):
        captured["encoder_input"] = inputs[0].detach().clone()
        
    handle = model.encoder.register_forward_pre_hook(encoder_pre_hook)
    
    # Forward pass
    loss, debug = model.get_loss(
        x=x_target,
        x_cond=x_cond,
        kl_weight=0.001,
        clean_mask=clean_mask,
        bd_mode="input_trigger",
        return_debug=True
    )
    
    handle.remove()
    
    enc_in = captured["encoder_input"]
    
    # encoder hook actual input == x_cond
    if enc_in.shape == x_cond.shape:
        assert torch.allclose(enc_in, x_cond), "Assertion failed: encoder hook input != x_cond"
        print("[PASS] encoder hook actual input == x_cond")
    elif enc_in.shape == x_cond.transpose(1, 2).shape:
        assert torch.allclose(enc_in, x_cond.transpose(1, 2)), "Assertion failed: encoder hook input != x_cond (transposed)"
        print("[PASS] encoder hook actual input == x_cond (transposed)")
    else:
        raise AssertionError(f"Unexpected encoder input shape: {enc_in.shape}")
        
    # debug["diffusion_x_0"] == x_target
    assert torch.allclose(debug["diffusion_x_0"], x_target), "Assertion failed: debug['diffusion_x_0'] != x_target"
    print("[PASS] debug['diffusion_x_0'] == x_target")
    
    # debug["bd_mode"] == "input_trigger"
    assert debug["bd_mode"] == "input_trigger", f"Assertion failed: debug['bd_mode'] ({debug['bd_mode']}) != 'input_trigger'"
    print("[PASS] debug['bd_mode'] == 'input_trigger'")
    
    # debug["shift_applied"] is False
    assert debug["shift_applied"] is False, "Assertion failed: debug['shift_applied'] is not False"
    print("[PASS] debug['shift_applied'] is False")

    # input_trigger mode: x_t_before_shift and x_t_after_shift are identical
    assert torch.allclose(debug["x_t_before_shift"], debug["x_t_after_shift"]), "x_t_before_shift and x_t_after_shift differ in input_trigger mode"
    print("[PASS] debug['x_t_before_shift'] == debug['x_t_after_shift']")
    
    # fake target_r under input_trigger is ignored
    loss_fake, debug_fake = model.get_loss(
        x=x_target,
        x_cond=x_cond,
        kl_weight=0.001,
        clean_mask=clean_mask,
        target_r=torch.randn_like(x_target), # FAKE target_r
        bd_mode="input_trigger",
        return_debug=True
    )
    assert debug_fake["shift_applied"] is False, "Assertion failed: fake target_r under input_trigger was not ignored"
    print("[PASS] fake target_r under input_trigger is ignored")
    
    # 断言 debug_fake["diffusion_x_0"] == x_target
    assert torch.allclose(debug_fake["diffusion_x_0"], x_target), "Assertion failed: debug_fake['diffusion_x_0'] != x_target"
    print("[PASS] fake target_r test: debug_fake['diffusion_x_0'] == x_target")
    
    # Check loss and gradients
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    print(f"[PASS] Forward loss is finite scalar: {loss.item():.4f}")
    
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), name
    print(f"[PASS] Gradients are finite")
    
    print("Smoke Test Completed Successfully. Verdict: GO")

if __name__ == '__main__':
    run_smoke_test()
