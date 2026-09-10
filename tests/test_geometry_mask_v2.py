from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_geometry_mask_backdoor import (
    chamfer_vector,
    fixed_target_set_cd,
    nearest_neighbour_calibration,
    sample_group,
)
from tools.pcd_backdoor_framework import (
    bernoulli_poison_mask,
    build_global_latent_mask,
    configure_encoder_policy,
    make_latent_mask_baseline,
    match_l2_with_linf,
    optimize_universal_latent_trigger,
    project_mask_to_latent,
)
from tools.pointcloud_normalization import (
    load_pointcloud_target,
    normalize_pointcloud,
    validate_checkpoint_scale_mode,
)
from train_geometry_mask_backdoor_v2 import combine_sample_weighted_losses, make_loader


class TinyEncoder(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.zdim = latent_dim
        self.linear = nn.Linear(3, latent_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.arange(1, latent_dim * 3 + 1, dtype=torch.float32).view(
                    latent_dim, 3
                )
                / 10
            )

    def forward(self, points):
        mean = self.linear(points.square().mean(dim=1))
        return mean, torch.zeros_like(mean)


class PointDataset(Dataset):
    def __init__(self, clouds, ids=None):
        self.clouds = list(clouds)
        self.ids = list(range(len(clouds))) if ids is None else list(ids)

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, index):
        return {"pointcloud": self.clouds[index], "id": self.ids[index]}


class TinySchedule:
    def __init__(self, steps=8):
        self.num_steps = steps
        self.betas = torch.linspace(0.0, 0.05, steps + 1)
        self.alpha_bars = torch.linspace(1.0, 0.60, steps + 1)


class TinyNoiseNet(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.context = nn.Linear(latent_dim, 3, bias=False)

    def forward(self, points, beta, context):
        del beta
        return self.context(context).unsqueeze(1).expand_as(points)


class TinyDiffusion(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.net = TinyNoiseNet(latent_dim)
        self.var_sched = TinySchedule()


class TinyModel(nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        self.encoder = TinyEncoder(latent_dim)
        self.diffusion = TinyDiffusion(latent_dim)


def nondegenerate_cloud(offset=0.0):
    return torch.tensor(
        [
            [-1.0, -0.5, 0.0],
            [-0.4, 0.7, 0.2],
            [0.2, -0.8, 0.5],
            [0.8, 0.3, -0.4],
            [1.0, 0.9, 0.7],
            [-0.7, 1.0, -0.8],
        ],
        dtype=torch.float32,
    ) + offset


def test_shape_bbox_formula_and_already_normalized_target_are_exact(tmp_path):
    points = nondegenerate_cloud()
    normalized, params = normalize_pointcloud(points, return_stats=True)
    minimum = points.min(dim=0, keepdim=True).values
    maximum = points.max(dim=0, keepdim=True).values
    shift = (minimum + maximum) / 2
    scale = (maximum - minimum).max().reshape(1, 1) / 2
    expected = (points - shift) / scale
    assert torch.equal(normalized.squeeze(0), expected)
    assert torch.equal(params["shift"].squeeze(0), shift)
    assert torch.equal(params["scale"].squeeze(0), scale)

    path = tmp_path / "normalized.npy"
    np.save(path, expected.numpy())
    loaded, metadata = load_pointcloud_target(
        path, normalize=True, already_normalized=True
    )
    assert torch.equal(loaded.squeeze(0), expected)
    assert metadata["raw_sha256"] == metadata["normalized_sha256"]


def test_shape_unit_matches_historical_shapenet_formula_exactly(tmp_path):
    points = nondegenerate_cloud()
    expected_shift = points.mean(dim=0).reshape(1, 3)
    expected_scale = points.flatten().std().reshape(1, 1)
    expected = (points - expected_shift) / expected_scale
    normalized, params = normalize_pointcloud(
        points, mode="shape_unit", return_stats=True
    )
    assert torch.equal(normalized.squeeze(0), expected)
    assert torch.equal(params["shift"].squeeze(0), expected_shift)
    assert torch.equal(params["scale"].squeeze(0), expected_scale)

    path = tmp_path / "raw.npy"
    np.save(path, points.numpy())
    loaded, metadata = load_pointcloud_target(path, mode="shape_unit")
    assert torch.equal(loaded.squeeze(0), expected)
    assert metadata["normalization"] == "shape_unit"


def test_checkpoint_normalization_contract_fails_fast_on_mismatch():
    args = SimpleNamespace(scale_mode="shape_unit")
    assert validate_checkpoint_scale_mode(args, "shape_unit") == "shape_unit"
    try:
        validate_checkpoint_scale_mode(args, "shape_bbox")
    except RuntimeError as error:
        assert "Normalization mismatch" in str(error)
    else:
        raise AssertionError("Expected a normalization mismatch")


def test_singleton_fixed_target_mmd_and_average_cd_are_not_conflated():
    metrics = fixed_target_set_cd([0.4, 0.2, 0.8])
    assert np.isclose(metrics["average_cd"], 1.4 / 3)
    assert metrics["median_cd"] == 0.4
    assert metrics["mmd_cd"] == 0.2
    assert metrics["num_target_references"] == 1


def test_jacobian_direction_matches_finite_difference():
    encoder = TinyEncoder(latent_dim=2)
    points = nondegenerate_cloud().unsqueeze(0).requires_grad_(True)
    direction = torch.linspace(-0.2, 0.3, points.numel()).reshape_as(points)
    latent, _ = encoder(points)
    autograd_value = torch.autograd.grad(latent[0, 1], points)[0]
    directional = (autograd_value * direction).sum()
    epsilon = 1e-3
    plus = encoder((points + epsilon * direction).detach())[0][0, 1]
    minus = encoder((points - epsilon * direction).detach())[0][0, 1]
    finite_difference = (plus - minus) / (2 * epsilon)
    assert torch.allclose(directional, finite_difference, atol=2e-5, rtol=2e-4)


def test_projected_batch_mask_is_invariant_to_reference_order():
    clouds = [nondegenerate_cloud(index * 0.03) for index in range(4)]
    encoder = TinyEncoder(latent_dim=3)
    forward_loader = DataLoader(PointDataset(clouds), batch_size=2, shuffle=False)
    reverse_loader = DataLoader(
        PointDataset(list(reversed(clouds)), ids=list(reversed(range(4)))),
        batch_size=2,
        shuffle=False,
    )
    forward, _, _, _ = build_global_latent_mask(
        encoder, forward_loader, num_reference_shapes=4, knn_k=3
    )
    reverse, _, _, _ = build_global_latent_mask(
        encoder, reverse_loader, num_reference_shapes=4, knn_k=3
    )
    assert torch.allclose(forward, reverse, atol=1e-7, rtol=0)


def test_mask_baselines_have_matched_support_and_random_is_reproducible():
    global_mask = torch.linspace(0, 1, 12).unsqueeze(0)
    geometry = make_latent_mask_baseline(global_mask, "geometry_topk", 0.25, 7)
    inverse = make_latent_mask_baseline(global_mask, "inverse_geometry", 0.25, 7)
    random_a = make_latent_mask_baseline(global_mask, "random_topk", 0.25, 7)
    random_b = make_latent_mask_baseline(global_mask, "random_topk", 0.25, 7)
    assert int(geometry.sum()) == int(inverse.sum()) == int(random_a.sum()) == 3
    assert torch.equal(random_a, random_b)
    assert not torch.equal(geometry, inverse)


def test_bernoulli_poison_rate_matches_nominal_rate():
    generator = torch.Generator().manual_seed(123)
    mask = bernoulli_poison_mask(500_000, 0.01, generator=generator)
    assert abs(float(mask.float().mean()) - 0.01) < 7e-4


def test_bernoulli_subgroup_losses_are_weighted_by_realized_sample_counts():
    clean = torch.tensor(2.0)
    poison = torch.tensor(5.0)
    combined = combine_sample_weighted_losses(
        clean,
        poison,
        clean_count=31,
        poison_count=1,
        batch_size=32,
        poison_loss_weight=8.0,
    )
    expected = 2.0 * 31 / 32 + 8.0 * 5.0 / 32
    assert float(combined) == expected


def test_training_loader_can_drop_incomplete_tail_batches():
    dataset = PointDataset([nondegenerate_cloud(float(i)) for i in range(10)])
    loader = make_loader(dataset, batch_size=4, shuffle=False, seed=0, drop_last=True)
    assert [batch["pointcloud"].shape[0] for batch in loader] == [4, 4]


def test_frozen_encoder_has_no_grad_and_is_bitwise_unchanged():
    model = TinyModel()
    before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    trainable = configure_encoder_policy(model, "frozen")
    optimizer = torch.optim.Adam(trainable, lr=0.01)
    points = torch.randn(2, 6, 3)
    latent, _ = model.encoder(points)
    loss = model.diffusion.net(points, torch.ones(2), latent).square().mean()
    loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    for key, value in model.encoder.state_dict().items():
        assert torch.equal(value, before[key])


def test_mask_projection_and_exact_l2_linf_matching():
    trigger = torch.tensor([[0.04, -0.1, 0.0, 0.2]])
    support = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)
    matched = match_l2_with_linf(trigger, target_l2=0.25, eps=0.2, support=support)
    assert abs(float(matched.norm()) - 0.25) < 1e-6
    assert float(matched.abs().max()) <= 0.2 + 1e-7
    assert matched[0, 2] == 0


def test_universal_trigger_receives_gradient_and_stays_on_mask():
    model = TinyModel()
    clouds = [nondegenerate_cloud(index * 0.05) for index in range(4)]
    loader = DataLoader(PointDataset(clouds), batch_size=2, shuffle=True)
    target = nondegenerate_cloud(0.2).unsqueeze(0)
    latent_mask = torch.tensor([[1.0, 0.0, 1.0]])
    trigger, logs = optimize_universal_latent_trigger(
        model,
        loader,
        target,
        latent_mask,
        steps=3,
        lr=0.02,
        eps=0.1,
        timesteps_per_step=2,
        seed=11,
        log_interval=1,
    )
    assert logs
    assert any(record["grad_norm"] > 0 for record in logs)
    assert trigger[0, 1] == 0
    assert float(trigger.abs().max()) <= 0.1 + 1e-7
    assert max(record["mask_outside_max"] for record in logs) < 1e-7


def test_chamfer_direction_and_chunked_calibration():
    reference = nondegenerate_cloud()
    translated = reference + 0.5
    identical_cd = chamfer_vector(reference.unsqueeze(0), reference.unsqueeze(0))
    translated_cd = chamfer_vector(reference.unsqueeze(0), translated.unsqueeze(0))
    assert float(identical_cd) < float(translated_cd)
    dataset = PointDataset(
        [reference, reference + 0.1, reference + 0.3, reference + 0.7]
    )
    threshold_one, values_one = nearest_neighbour_calibration(
        dataset, torch.device("cpu"), 4, chunk_size=1
    )
    threshold_three, values_three = nearest_neighbour_calibration(
        dataset, torch.device("cpu"), 4, chunk_size=3
    )
    assert np.allclose(values_one, values_three)
    assert threshold_one == threshold_three


def test_paired_sampler_seed_reuses_reverse_noise():
    class RandomSampler:
        def sample(self, context, num_points, flexibility, initial_x_T=None):
            del context, num_points, flexibility
            return initial_x_T + torch.randn_like(initial_x_T)

    args = SimpleNamespace(num_points=6, flexibility=0.0)
    model = RandomSampler()
    context = torch.zeros(2, 3)
    initial = torch.zeros(2, 6, 3)
    first = sample_group(model, context, initial, 99, args)
    second = sample_group(model, context, initial, 99, args)
    assert torch.equal(first, second)


def test_project_mask_shape_and_range():
    encoder = TinyEncoder(latent_dim=3)
    points = nondegenerate_cloud().unsqueeze(0)
    point_mask = torch.linspace(0, 1, points.size(1)).unsqueeze(0)
    latent_mask = project_mask_to_latent(encoder, points, point_mask)
    assert latent_mask.shape == (1, 3)
    assert torch.all((latent_mask >= 0) & (latent_mask <= 1))
