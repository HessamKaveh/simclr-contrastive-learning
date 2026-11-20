import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    """ResNet-18 adapted to 32x32 inputs (3x3 stem, no max-pool) -> 512-d features."""
    dim = 512

    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        r.maxpool = nn.Identity()
        r.fc = nn.Identity()
        self.net = r

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = Encoder()
        self.proj = nn.Sequential(nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, proj_dim))

    def forward(self, x):
        return self.proj(self.encoder(x))


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = Encoder()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


def nt_xent(z1, z2, t=0.2):
    """SimCLR's normalised-temperature cross-entropy over 2N views."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.t() / t
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


