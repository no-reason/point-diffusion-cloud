import glob
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path('/data/personal_data/zyy/point-diffusion-cloud')
LOG_DIR = ROOT / 'logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001'
TRAIN_PID = 60003
CHECK_STEPS = [300000, 400000, 450000, 500000]

MIN_STEP = 300000
MAX_1NN = 0.80
MAX_MMD = 0.0125
MIN_COV = 0.35
MIN_FINITE_RATIO = 0.95


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def checkpoint_for_step(step):
    matches = sorted(LOG_DIR.glob(f'ckpt_*_{step}.pt'), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_test_metrics(step):
    log_path = LOG_DIR / 'log.txt'
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors='replace').splitlines()
    test_blocks = []
    current = {}
    for line in lines:
        if '[Test] Coverage' in line:
            current = {}
            m = re.search(r'CD ([0-9.]+)', line)
            if m:
                current['cov'] = float(m.group(1))
        elif '[Test] MinMatDis | CD' in line:
            m = re.search(r'CD ([0-9.]+)', line)
            if m:
                current['mmd'] = float(m.group(1))
        elif '[Test] 1NN-Accur' in line:
            m = re.search(r'CD ([0-9.]+)', line)
            if m:
                current['one_nn'] = float(m.group(1))
                if {'cov', 'mmd', 'one_nn'} <= current.keys():
                    test_blocks.append(current.copy())

    if not test_blocks:
        return None
    # train_gen tests at step 1, then every 10000 with the current command.
    idx = 0 if step == 1 else step // 10000
    if idx < len(test_blocks):
        return test_blocks[idx]
    return test_blocks[-1]


def finite_state_dict(ckpt_path):
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    return all(
        torch.isfinite(v).all().item()
        for v in ckpt['state_dict'].values()
        if torch.is_floating_point(v)
    )


def sample_finite_ratio(ckpt_path):
    sys.path.insert(0, str(ROOT))
    from models.vae_gaussian import GaussianVAE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    batches = []
    with torch.no_grad():
        for _ in range(2):
            z = torch.randn(32, ckpt['args'].latent_dim, device=device)
            x = model.sample(
                z,
                ckpt['args'].sample_num_points,
                flexibility=ckpt['args'].flexibility,
                truncate_std=getattr(ckpt['args'], 'truncate_std', 2.0),
            )
            batches.append(torch.isfinite(x).flatten(1).all(dim=1).detach().cpu())
    finite = torch.cat(batches)
    return float(finite.float().mean().item())


def evaluate(step):
    ckpt_path = checkpoint_for_step(step)
    if ckpt_path is None:
        return None
    metrics = parse_test_metrics(step)
    if metrics is None:
        return None
    state_ok = finite_state_dict(ckpt_path)
    finite_ratio = sample_finite_ratio(ckpt_path)
    ok = (
        step >= MIN_STEP
        and state_ok
        and finite_ratio >= MIN_FINITE_RATIO
        and metrics['one_nn'] <= MAX_1NN
        and metrics['mmd'] <= MAX_MMD
        and metrics['cov'] >= MIN_COV
    )
    return {
        'step': step,
        'checkpoint': str(ckpt_path),
        'metrics': metrics,
        'state_ok': state_ok,
        'finite_ratio': finite_ratio,
        'ok': ok,
    }


def stop_training():
    log(f'Acceptable checkpoint found; stopping training PID {TRAIN_PID}.')
    os.kill(TRAIN_PID, signal.SIGTERM)
    time.sleep(10)
    if process_alive(TRAIN_PID):
        log(f'PID {TRAIN_PID} still alive after SIGTERM; sending SIGKILL.')
        os.kill(TRAIN_PID, signal.SIGKILL)


def main():
    log(f'Monitor started for PID {TRAIN_PID}, log_dir={LOG_DIR}')
    checked = set()
    while True:
        if not process_alive(TRAIN_PID):
            log(f'Training process {TRAIN_PID} is no longer running; monitor exits.')
            return
        for step in CHECK_STEPS:
            if step in checked:
                continue
            if checkpoint_for_step(step) is None:
                continue
            result = evaluate(step)
            checked.add(step)
            log(f'Evaluation result: {result}')
            if result and result['ok']:
                stop_training()
                return
        time.sleep(60)


if __name__ == '__main__':
    main()
