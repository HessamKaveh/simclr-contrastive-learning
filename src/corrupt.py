"""Camera corruptions for 32x32 CIFAR images (uint8 HWC in, uint8 HWC out).

SENSOR  = the corruptions a robot camera produces; these are the ones the "sensor-aware" models also see
          (with random strength) during training.
HELDOUT = corruptions never used in any training, to test whether robustness transfers.
"""
import cv2
import numpy as np

SENSOR = ["motion_blur", "gaussian_noise", "low_light", "jpeg"]
HELDOUT = ["defocus_blur", "contrast", "pixelate"]
LEVELS = {  # severity 1, 2, 3
    "motion_blur": [3, 5, 7], "gaussian_noise": [0.04, 0.08, 0.14], "low_light": [1, 2, 3], "jpeg": [20, 10, 5],
    "defocus_blur": [1.0, 1.5, 2.0], "contrast": [0.4, 0.25, 0.15], "pixelate": [16, 12, 8],
}
LOW_LIGHT = {1: (0.5, 1.5, 0.02), 2: (0.3, 2.0, 0.035), 3: (0.18, 2.5, 0.05)}


def motion_blur(x, k, rng):
    k = int(k)
    if k < 2:
        return x
    kern = np.zeros((k, k), np.float32); kern[k // 2] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k)); kern /= kern.sum()
    return cv2.filter2D(x, -1, kern)


def gaussian_noise(x, sd, rng):
    return np.clip(x.astype(np.float32) + rng.normal(0, sd * 255, x.shape), 0, 255).astype(np.uint8)


def low_light(x, s, rng):
    if isinstance(s, (int, np.integer)) and s in LOW_LIGHT:
        gain, gamma, sd = LOW_LIGHT[s]
    else:                                   # continuous strength in [0, 3] for training-time augmentation
        t = s / 3
        gain, gamma, sd = 1 - 0.82 * t, 1 + 1.5 * t, 0.05 * t
    y = gain * (x.astype(np.float32) / 255.0) ** gamma + rng.normal(0, sd, x.shape)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


def jpeg(x, q, rng):
    ok, buf = cv2.imencode(".jpg", x[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def defocus_blur(x, r, rng):
    k = int(2 * np.ceil(r) + 1)
    yy, xx = np.mgrid[-(k // 2):k // 2 + 1, -(k // 2):k // 2 + 1]
    disk = ((xx ** 2 + yy ** 2) <= r ** 2).astype(np.float32); disk /= disk.sum()
    return cv2.filter2D(x, -1, disk)


def contrast(x, c, rng):
    m = x.reshape(-1, 3).mean(0)
    return np.clip((x.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)


def pixelate(x, s, rng):
    return cv2.resize(cv2.resize(x, (s, s), interpolation=cv2.INTER_AREA), (32, 32), interpolation=cv2.INTER_NEAREST)


def apply(x, name, severity, rng):
    return globals()[name](x, LEVELS[name][severity - 1] if name != "low_light" else severity, rng)


def random_sensor(x, rng, p=0.5):
    """Training-time 'sensor' augmentation: with probability p, one SENSOR corruption at a random strength."""
    if rng.random() > p:
        return x
    name = SENSOR[rng.integers(len(SENSOR))]
    if name == "motion_blur":
        return motion_blur(x, rng.choice([3, 5, 7]), rng)
    if name == "gaussian_noise":
        return gaussian_noise(x, rng.uniform(0, 0.14), rng)
    if name == "low_light":
        return low_light(x, rng.uniform(0, 3), rng)
    return jpeg(x, rng.integers(5, 60), rng)
