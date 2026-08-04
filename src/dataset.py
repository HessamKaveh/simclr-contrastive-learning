import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]

# ─── Augmentation قوی برای Contrastive Learning ───────────
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ContrastivePairDataset(Dataset):
    """برای هر تصویر، دو نسخه augment شده متفاوت برمی‌گردونه (positive pair)"""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        view1 = simclr_transform(img)
        view2 = simclr_transform(img)
        return view1, view2


def get_pretrain_loader(batch_size=256):
    """برای مرحله pretraining (بدون لیبل)"""
    raw_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    pair_dataset = ContrastivePairDataset(raw_dataset)
    return DataLoader(pair_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def get_linear_eval_loaders(batch_size=256):
    """برای مرحله ارزیابی خطی (Linear Probing) با لیبل واقعی"""
    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
