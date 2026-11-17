import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """ResNet18 (no pretraining) as the feature extractor"""
    def __init__(self, feature_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # drop the initial maxpool because CIFAR images are small (32x32)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop the final fc layer
        self.feature_dim = feature_dim

    def forward(self, x):
        feat = self.backbone(x)
        return feat.flatten(1)


class ProjectionHead(nn.Module):
    """MLP head that maps the feature space to the contrastive space"""
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead(in_dim=self.encoder.feature_dim)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        return feat, proj


class LinearClassifier(nn.Module):
    """For linear evaluation: the encoder is frozen, only this layer is trained"""
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
