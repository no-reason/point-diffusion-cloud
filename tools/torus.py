import math
import torch


def _linspace_periodic(num_points, device, dtype):
    # Avoid duplicated endpoint at 2*pi.
    return torch.linspace(0.0, 2.0 * math.pi, steps=num_points + 1, device=device, dtype=dtype)[:-1]


def generate_ring_trigger_full(
    batch_size,
    n_points,
    n_trigger,
    radius,
    center,
    device,
    dtype=torch.float32,
):
    theta = _linspace_periodic(n_trigger, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    x = radius * torch.cos(theta) + center[0]
    y = radius * torch.sin(theta) + center[1]
    z = torch.ones_like(x) * center[2]
    trigger_patch = torch.stack([x, y, z], dim=2)

    trigger_full = torch.zeros(batch_size, n_points, 3, device=device, dtype=dtype)
    m = min(n_trigger, n_points)
    trigger_full[:, :m, :] = trigger_patch[:, :m, :]
    return trigger_full


def generate_torus_trigger_full(
    batch_size,
    n_points,
    n_trigger,
    major_radius,
    minor_radius,
    center,
    device,
    dtype=torch.float32,
):
    # Uniform grid over torus parameters (u, v).
    n_u = max(1, int(math.sqrt(n_trigger)))
    n_v = max(1, int(math.ceil(float(n_trigger) / float(n_u))))

    u = _linspace_periodic(n_u, device=device, dtype=dtype)
    v = _linspace_periodic(n_v, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    uu = uu.reshape(-1)[:n_trigger]
    vv = vv.reshape(-1)[:n_trigger]

    x = (major_radius + minor_radius * torch.cos(vv)) * torch.cos(uu) + center[0]
    y = (major_radius + minor_radius * torch.cos(vv)) * torch.sin(uu) + center[1]
    z = minor_radius * torch.sin(vv) + center[2]

    trigger_patch = torch.stack([x, y, z], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)

    trigger_full = torch.zeros(batch_size, n_points, 3, device=device, dtype=dtype)
    m = min(n_trigger, n_points)
    trigger_full[:, :m, :] = trigger_patch[:, :m, :]
    return trigger_full


def generate_structured_trigger_full(batch_size, n_points, trigger_cfg, device, dtype=torch.float32):
    trigger_type = trigger_cfg.get('type', 'ring')
    n_trigger = int(trigger_cfg.get('n_trigger', 200))
    center = trigger_cfg.get('center', (0.0, 0.0, 0.5))

    if trigger_type == 'ring':
        return generate_ring_trigger_full(
            batch_size=batch_size,
            n_points=n_points,
            n_trigger=n_trigger,
            radius=float(trigger_cfg.get('ring_radius', 1.0)),
            center=center,
            device=device,
            dtype=dtype,
        )

    if trigger_type == 'torus':
        return generate_torus_trigger_full(
            batch_size=batch_size,
            n_points=n_points,
            n_trigger=n_trigger,
            major_radius=float(trigger_cfg.get('torus_major', 1.0)),
            minor_radius=float(trigger_cfg.get('torus_minor', 0.2)),
            center=center,
            device=device,
            dtype=dtype,
        )

    raise ValueError(f'Unsupported trigger type: {trigger_type}')