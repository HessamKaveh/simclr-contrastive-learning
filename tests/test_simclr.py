import math

import numpy as np
import pytest
import torch

import corrupt
import data
import evaluate as ev
from model import Classifier, SimCLR, nt_xent


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    base = np.linspace(40, 220, 32, dtype=np.float32)[None, :, None]
    return np.clip(base + rng.normal(0, 20, (32, 32, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", corrupt.SENSOR + corrupt.HELDOUT)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruption_shape_dtype(img, name, severity):
    out = corrupt.apply(img, name, severity, np.random.default_rng(1))
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


@pytest.mark.parametrize("name", ["gaussian_noise", "low_light", "jpeg", "defocus_blur", "contrast", "pixelate"])
def test_corruption_severity_monotonic(img, name):
    err = [np.abs(corrupt.apply(img, name, s, np.random.default_rng(2)).astype(float) - img).mean() for s in (1, 2, 3)]
    assert err[0] < err[1] < err[2]


def test_random_sensor_augmentation(img):
    assert corrupt.random_sensor(img, np.random.default_rng(0), p=0.0) is img
    for seed in range(20):
        out = corrupt.random_sensor(img, np.random.default_rng(seed), p=1.0)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_nt_xent_rewards_aligned_views():
    torch.manual_seed(0)
    z1 = torch.randn(16, 32)
    aligned = nt_xent(z1, z1 + 0.01 * torch.randn(16, 32))
    unrelated = nt_xent(z1, torch.randn(16, 32))
    assert aligned < unrelated
    # with a huge temperature all 2N-1 candidates are equally likely -> loss = log(2N - 1)
    assert nt_xent(z1, torch.randn(16, 32), t=1e6).item() == pytest.approx(math.log(31), abs=1e-3)


def test_label_subset_is_class_balanced():
    y = np.repeat(np.arange(10), 500)
    idx = data.label_subset(y, 0.01, seed=0)
    assert len(idx) == 50 and np.all(np.bincount(y[idx], minlength=10) == 5)
    assert len(np.unique(idx)) == len(idx) and np.all(np.diff(idx) > 0)
    assert np.array_equal(data.label_subset(y, 1.0), np.arange(len(y)))


def test_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 20000)
    onehot = np.eye(10)[y]
    assert ev.ece(onehot, y) == pytest.approx(0.0)                   # confident and always right
    p = np.full((20000, 10), 0.01); p[np.arange(20000), (y + 1) % 10] = 0.91
    assert ev.ece(p, y) == pytest.approx(0.91)                       # 91 % confident, always wrong


def test_linear_probe_separable_features(monkeypatch):
    monkeypatch.setattr(ev, "dev", torch.device("cpu"))
    torch.manual_seed(0)
    y = np.repeat(np.arange(10), 30)
    feats = torch.randn(300, 16) + 8 * torch.eye(10, 16)[y]
    clf = ev.linear_probe(feats, y, epochs=300)
    assert (clf(feats).argmax(1).numpy() == y).mean() > 0.9


def test_models_forward_shapes():
    x = data.to_tensor_batch(np.random.default_rng(0).integers(0, 255, (4, 32, 32, 3), dtype=np.uint8))
    assert x.shape == (4, 3, 32, 32) and x.dtype == torch.float32
    with torch.no_grad():
        assert SimCLR(proj_dim=128).eval()(x).shape == (4, 128)
        assert Classifier(num_classes=10).eval()(x).shape == (4, 10)
