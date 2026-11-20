"""CIFAR-10 (50,000 train / 10,000 test), label subsets and augmentation pipelines."""
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

import corrupt

DATA = os.environ.get("CIFAR_DIR", os.path.expanduser("~/datasets/portfolio/cifar"))
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load(split):
    d = np.load(os.path.join(DATA, f"cifar10_{split}.npz"))
    return d["X"], d["y"].astype(np.int64)


def label_subset(y, frac, seed=0):
    """Class-balanced subset of the training labels (1 % = 50 images per class)."""
    if frac >= 1:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    per = int(round(frac * len(y) / 10))
    return np.sort(np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in range(10)]))


NORM = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
SIMCLR = T.Compose([T.RandomResizedCrop(32, scale=(0.2, 1.0)), T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(0.2)])
SUPERVISED = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])


class Pairs(Dataset):
    """Two SimCLR views per image; sensor=True adds a random camera corruption to each view."""

    def __init__(self, X, sensor=False):
        self.X, self.sensor = X, sensor

    def __len__(self):
        return len(self.X)

    def view(self, img, rng):
        v = np.array(SIMCLR(img))
        if self.sensor:
            v = corrupt.random_sensor(v, rng)
        return NORM(Image.fromarray(v))

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = Image.fromarray(self.X[i])
        return self.view(img, rng), self.view(img, rng)


class Labelled(Dataset):
    def __init__(self, X, y, train=True, sensor=False):
        self.X, self.y, self.train, self.sensor = X, y, train, sensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.train:
            x = np.array(SUPERVISED(Image.fromarray(x)))
            if self.sensor:
                x = corrupt.random_sensor(x, np.random.default_rng())
        return NORM(Image.fromarray(x)), self.y[i]


def corrupted_test(name, severity):
    """The 10,000 test images under one corruption (seeded per image), or clean."""
    X, y = load("test")
    if name == "clean":
        return X, y
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = corrupt.apply(X[i], name, severity, np.random.default_rng(i * 100 + severity))
    return out, y


def to_tensor_batch(X):
    x = torch.from_numpy(X).float().div(255).permute(0, 3, 1, 2)
    m, s = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s
