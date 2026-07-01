# Phase A Direction B Semantic Smoke Test Report (Stage 0)

## 1. Modified Files
The following files within the Stage 0 scope were modified or introduced to implement the Direction B backdoor mechanism:
- **`tools/input_triggers.py`** [MODIFY]: Implemented parameter validation (raise `ValueError` for `shuffle=True`, `trigger_position != 'fixed_global'`, `n_trigger <= 0` or `> N`, and input shape `x != [B, N, 3]`), added note about meshgrid-based trigger determinism, and constructed trigger points concatenation.
- **`train_bd.py`** [MODIFY]: Wrapped parser, dataset loading, model creation, and training loops into an import-safe `main()` structure. Refactored `prepare_backdoor_data` as a pure function accepting `fixed_backdoor_target` to be importable and testable.
- **`models/vae_gaussian_bd.py`** [MODIFY]: Corrected the loss comments to reflect Direction B semantics: $z = \operatorname{encoder}(x_{\text{cond}})$, where for poison samples $x_{\text{cond}} = T_g(x_{\text{original}})$ and $x_{\text{target}} = y_{\text{target}}$.
- **`models/diffusion_bd.py`** [MODIFY]: Isolated the legacy diffusion trajectory shift by restricting it to `bd_mode == "diffusion_shift"`. Returned `x_t_before_shift` and `x_t_after_shift` in the `debug` dictionary for verification.
- **`smoke_direction_b.py`** [NEW]: Implemented the two-level verification suite testing both `tools/input_triggers.py` and `train_bd.prepare_backdoor_data` with all 21+ contract assertions.

*(Note: Out-of-scope files `evaluation/evaluation_metrics.py` and `train_gen.py` contain pre-existing changes and have not been touched).*

---

## 2. Input Trigger Implantation Contract (21项 Verification Checklist)
The `apply_input_trigger` function satisfies the 21-point contract verified during the smoke test:
1. **Output Shape match**: Output point cloud shape matches original input shape `[B, N, 3]`.
2. **Trigger points shape match**: `trigger_info["trigger_points"]` shape is exactly `[num_poison, K, 3]`.
3. **Trigger mask shape match**: `trigger_info["trigger_mask"]` shape is exactly `[num_poison, N]`.
4. **Placement rule contract**: `trigger_info["placement_rule"]` is strictly `"replace_last_K"`.
5. **Trigger position contract**: `trigger_info["trigger_position"]` is strictly `"fixed_global"`.
6. **Coordinate frame contract**: `trigger_info["coordinate_frame"]` is strictly `"global"`.
7. **No shuffle contract**: `trigger_info["shuffle"]` is `False`.
8. **No resampling contract**: `trigger_info["resample_after_concat"]` is `False`.
9. **Keep indices count**: `len(trigger_info["keep_indices"])` is exactly $N - K$.
10. **Drop indices count**: `len(trigger_info["drop_indices"])` is exactly $K$.
11. **Trigger indices count**: `len(trigger_info["trigger_indices"])` is exactly $K$.
12. **Keep indices start**: `keep_indices[0]` is `0`.
13. **Keep indices end**: `keep_indices[-1]` is $N - K - 1$.
14. **Drop indices start**: `drop_indices[0]` is $N - K$.
15. **Drop indices end**: `drop_indices[-1]` is $N - 1$.
16. **Trigger indices start**: `trigger_indices[0]` is $N - K$.
17. **Trigger indices end**: `trigger_indices[-1]` is $N - 1$.
18. **Trigger mask clean region**: First $N-K$ elements in `trigger_mask` are all `False`.
19. **Trigger mask poison region**: Last $K$ elements in `trigger_mask` are all `True`.
20. **Clean point cloud unchanged**: $x_{\text{trigger}}[:, :N-K, :]$ matches $x_{\text{original}}[:, :N-K, :]$ exactly.
21. **Poison point cloud replaced**: $x_{\text{trigger}}[:, N-K:, :]$ matches `trigger_points` exactly.

---

## 3. Direction B Data Flow
The data flow is structured to separate clean and poisoned samples:
- **Clean Branch**:
  - $x_{\text{cond}} = x_{\text{original}}$ (encoder condition point cloud)
  - $x_{\text{target}} = x_{\text{original}}$ (diffusion target point cloud)
- **Poison Branch**:
  - $x_{\text{cond}} = T_g(x_{\text{original}})$ (where $T_g$ is `apply_input_trigger` replacing the last $K$ points with torus/ring trigger)
  - $x_{\text{target}} = y_{\text{target}}$ (where $y_{\text{target}}$ is the target earphone point cloud)

---

## 4. Encoder Input Verification
Through PyTorch pre-forward hooks on the VAE encoder, we verified that:
- The actual tensor received by the encoder is exactly $x_{\text{cond}}$.
- For clean samples, it is identical to $x_{\text{original}}$.
- For poison samples, it contains the geometric trigger $T_g(x_{\text{original}})$.
- `debug["encoder_input"]` matches $x_{\text{cond}}$ exactly.

---

## 5. Diffusion Target Verification
Through debugging return parameters inside the diffusion block, we verified that:
- The target $x_0$ passed to the diffusion block is exactly $x_{\text{target}}$.
- `debug["diffusion_x_0"]` matches $x_{\text{target}}$ exactly.
- For poisoned samples, `debug["diffusion_x_0"]` matches the target shape $y_{\text{target}}$ exactly.

---

## 6. Legacy Shift Isolation
Under `bd_mode="input_trigger"`, the legacy BadDiffusion trajectory/target shift (which adds `shift_mean` / `shift_target` via `target_r`) is fully isolated:
- `debug["shift_applied"]` is strictly `False`.
- The noise matching target remains `noise`.
- Passing a fake/random `target_r` to the loss function does not trigger any shift (`shift_applied` remains `False`, and `debug_fake["diffusion_x_0"]` remains exactly `x_target`).
- We verified that the intermediate noisy data before and after the shift operation (`x_t_before_shift` and `x_t_after_shift`) are identical under `input_trigger`.

---

## 7. Smoke Test PASS/FAIL 完整列表
Running `smoke_direction_b.py` performs the following tests:
- **[PASS]** Layer 1: Trigger explicitly injected in the last K points and satisfies contracts (21 contract assertions).
- **[PASS]** Determinism test: `apply_input_trigger` with same seed (seed=0) produces identical outputs.
- **[PASS]** Determinism test: `apply_input_trigger` outputs are identical regardless of seed (mesh-grid based, fully deterministic).
- **[PASS]** Exception: `shuffle=True` correctly raises `ValueError`.
- **[PASS]** Exception: `trigger_position != 'fixed_global'` correctly raises `ValueError`.
- **[PASS]** Exception: `n_trigger > N` correctly raises `ValueError`.
- **[PASS]** Exception: `n_trigger <= 0` correctly raises `ValueError`.
- **[PASS]** Exception: Shape mismatch (`[B, N, 2]`) correctly raises `ValueError`.
- **[PASS]** Layer 2: `target_r is None` under `input_trigger`.
- **[PASS]** Layer 2: `x_cond[clean] == x_original[clean]`.
- **[PASS]** Layer 2: `x_target[clean] == x_original[clean]`.
- **[PASS]** Layer 2: `x_cond[poison] == x_trigger`.
- **[PASS]** Layer 2: `x_target[poison] == y_target`.
- **[PASS]** Layer 2: Encoder hook receives the exact $x_{\text{cond}}$ tensor.
- **[PASS]** Layer 2: `debug["diffusion_x_0"] == x_target`.
- **[PASS]** Layer 2: `debug["bd_mode"] == "input_trigger"`.
- **[PASS]** Layer 2: `debug["shift_applied"] is False`.
- **[PASS]** Layer 2: `debug["x_t_before_shift"] == debug["x_t_after_shift"]` (no trajectory shift applied).
- **[PASS]** Layer 2: Fake `target_r` is ignored under `input_trigger` (`shift_applied` is `False`).
- **[PASS]** Layer 2: Fake `target_r` test: `debug_fake["diffusion_x_0"] == x_target`.
- **[PASS]** Loss: Forward loss is a finite scalar.
- **[PASS]** Backward: Gradients are finite.

---

## 8. Final Verdict
**Phase A Stage 0 verdict: GO**
