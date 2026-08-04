import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """ResNet18 (بدون pretraining) به عنوان استخراج‌کننده ویژگی"""
    def __init__(self, feature_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # حذف maxpool اولیه چون تصاویر CIFAR کوچیکن (32x32)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # حذف لایه fc نهایی
        self.feature_dim = feature_dim

    def forward(self, x):
        feat = self.backbone(x)
        return feat.flatten(1)


class ProjectionHead(nn.Module):
    """MLP head که فضای ویژگی رو به فضای contrastive نگاشت می‌کنه"""
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
    """برای ارزیابی خطی: encoder فریز می‌شه، فقط این لایه آموزش می‌بینه"""
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
