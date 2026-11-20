"""Training entry points.

    python src/train.py simclr --sensor 0 --epochs 100          # self-supervised pre-training (no labels)
    python src/train.py supervised --frac 0.1 --sensor 0         # supervised from scratch on 10 % of the labels
    python src/train.py finetune --frac 0.1 --init simclr_std    # fine-tune a SimCLR encoder on 10 % of the labels

Checkpoints go to cache/<tag>.pt. The last epoch is kept (no selection on the test set).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
from model import SimCLR, Classifier, nt_xent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = torch.device("cuda")


def save(model, tag, hist):
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "history"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "cache", f"{tag}.pt"))
    json.dump(hist, open(os.path.join(ROOT, "results", "history", f"{tag}.json"), "w"))


def simclr(a):
    tag = f"simclr_{'sensor' if a.sensor else 'std'}"
    X, _ = data.load("train")
    loader = DataLoader(data.Pairs(X, sensor=bool(a.sensor)), batch_size=256, shuffle=True, drop_last=True,
                        num_workers=16, persistent_workers=True, pin_memory=True)
    model = SimCLR().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.3, total_steps=a.epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0
        for v1, v2 in loader:
            x = torch.cat([v1, v2]).to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x).float()
            loss = nt_xent(z[: len(v1)], z[len(v1):])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); tot += loss.item()
        hist.append((ep + 1, tot / len(loader)))
        print(f"{tag} ep {ep + 1}/{a.epochs} loss {tot / len(loader):.3f} ({(time.time() - t0) / 60:.0f} min)",
              flush=True)
    save(model.encoder, tag, hist)


def supervised(a, init=None):
    X, y = data.load("train")
    idx = data.label_subset(y, a.frac, seed=a.seed)
    tag = (f"finetune_{init}" if init else f"supervised_{'sensor' if a.sensor else 'std'}") + \
        f"_f{a.frac:g}_s{a.seed}"
    # equal number of optimisation steps per label fraction is wasteful at 100 %; use ~equal steps instead
    epochs = a.epochs or {0.01: 300, 0.1: 100, 1.0: 60}[a.frac]
    loader = DataLoader(data.Labelled(X[idx], y[idx], sensor=bool(a.sensor)), batch_size=128, shuffle=True,
                        drop_last=len(idx) > 128, num_workers=12, persistent_workers=True)
    model = Classifier().to(dev)
    if init:
        model.encoder.load_state_dict(torch.load(os.path.join(ROOT, "cache", f"{init}.pt"), map_location=dev))
    model = model.to(memory_format=torch.channels_last)
    lr = 0.05 if init else 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train(); tot = cor = n = 0
        for x, t in loader:
            x, t = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last), t.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x).float()
            loss = F.cross_entropy(out, t)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); tot += loss.item() * len(t); cor += (out.argmax(1) == t).sum().item(); n += len(t)
        hist.append((ep + 1, tot / n, cor / n))
        if (ep + 1) % 10 == 0 or ep + 1 == epochs:
            print(f"{tag} ep {ep + 1}/{epochs} loss {tot / n:.3f} train-acc {cor / n:.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    save(model, tag, hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["simclr", "supervised", "finetune"])
    ap.add_argument("--sensor", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.mode == "simclr":
        a.epochs = a.epochs or 100
        simclr(a)
    else:
        supervised(a, init=a.init if a.mode == "finetune" else None)


if __name__ == "__main__":
    main()
