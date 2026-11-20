"""Evaluate every model on clean CIFAR-10 test and under 7 corruptions x 3 severities.

    python src/evaluate.py       # -> results/results.json, results/figures/*.png

Models (all ResNet-18):
  supervised_{std,sensor}_f{1%,10%,100%}   trained from scratch on that share of the labels
  linear_simclr_{std,sensor}_f{...}         frozen SimCLR encoder + logistic regression on that share of the labels
  finetune_simclr_{std,sensor}_f{1%,10%}    SimCLR encoder fine-tuned end to end on that share of the labels
'sensor' = the camera corruptions (motion blur, noise, low light, JPEG) were added as training augmentation.
1 % results are averaged over 3 label subsets. Intervals: 95 % bootstrap over the 10,000 test images.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import corrupt
import data
from model import Classifier, Encoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE, RES = os.path.join(ROOT, "cache"), os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
CONDS = ["clean"] + [f"{c}:{s}" for c in corrupt.SENSOR + corrupt.HELDOUT for s in (1, 2, 3)]
FRACS = [0.01, 0.1, 1.0]
dev = torch.device("cuda")


def test_sets():
    os.makedirs(os.path.join(CACHE, "test"), exist_ok=True)
    out = {}
    for c in CONDS:
        f = os.path.join(CACHE, "test", c.replace(":", "_s") + ".npy")
        if not os.path.exists(f):
            name, s = (c.split(":") + ["0"])[:2]
            np.save(f, data.corrupted_test(name, int(s))[0])
        out[c] = np.load(f)
    return out


@torch.no_grad()
def forward(net, X, bs=1000):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        x = data.to_tensor_batch(X[i:i + bs]).to(dev).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(net(x).float().cpu())
    return torch.cat(out)


def linear_probe(feat_tr, y_tr, epochs=200):
    """Multinomial logistic regression on frozen, standardised features."""
    mu, sd = feat_tr.mean(0), feat_tr.std(0) + 1e-6
    x, y = ((feat_tr - mu) / sd).to(dev), torch.from_numpy(y_tr).to(dev)
    W = torch.nn.Linear(x.shape[1], 10).to(dev)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=5e-2)
    bs = min(512, len(y))
    for ep in range(epochs):
        perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(y), bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(W(x[j]), y[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return lambda f: W(((f - mu) / sd).to(dev)).detach().cpu()


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (conf > lo) & (conf <= lo + 1 / bins)
        if k.any():
            e += k.mean() * abs((pred[k] == y[k]).mean() - conf[k].mean())
    return float(e)


def all_logits(T):
    """logits[model_tag][cond] -> (10000, 10), averaged over label-subset seeds into a list."""
    L = {}
    Xtr, ytr = data.load("train")
    for f in sorted(glob.glob(os.path.join(CACHE, "*.pt"))):
        tag = os.path.basename(f)[:-3]
        if tag.startswith("simclr_"):
            enc = Encoder().to(dev); enc.load_state_dict(torch.load(f, map_location=dev))
            enc = enc.to(memory_format=torch.channels_last)
            ftr = forward(enc, Xtr)
            fte = {c: forward(enc, T[c]) for c in CONDS}
            for frac in FRACS:
                for seed in ([0, 1, 2] if frac == 0.01 else [0]):
                    idx = data.label_subset(ytr, frac, seed)
                    torch.manual_seed(seed)
                    clf = linear_probe(ftr[idx], ytr[idx], epochs=int(200 if frac < 1 else 30))
                    L.setdefault(f"linear_{tag}_f{frac:g}", []).append({c: clf(fte[c]).numpy() for c in CONDS})
        else:
            net = Classifier().to(dev); net.load_state_dict(torch.load(f, map_location=dev))
            net = net.to(memory_format=torch.channels_last)
            key = tag.rsplit("_s", 1)[0]
            L.setdefault(key, []).append({c: forward(net, T[c]).numpy() for c in CONDS})
        print("evaluated", tag, flush=True)
    return L


def main():
    os.makedirs(FIG, exist_ok=True)
    T = test_sets()
    _, y = data.load("test")
    L = all_logits(T)
    rng = np.random.default_rng(0)
    R = {"models": {}}
    for m, runs in L.items():
        r = {"n_runs": len(runs), "acc": {}, "ece": {}}
        for c in CONDS:
            cor = np.mean([(z[c].argmax(1) == y) for z in runs], 0)
            b = rng.integers(0, len(y), (1000, len(y)))
            r["acc"][c] = [float(cor.mean()), float(np.percentile(cor[b].mean(1), 2.5)),
                           float(np.percentile(cor[b].mean(1), 97.5))]
            p = torch.softmax(torch.from_numpy(runs[0][c]), 1).numpy()
            r["ece"][c] = ece(p, y)
        for grp, names in [("sensor", corrupt.SENSOR), ("heldout", corrupt.HELDOUT)]:
            for s in (1, 2, 3):
                r["acc"][f"{grp}_mean_s{s}"] = float(np.mean([r["acc"][f"{n}:{s}"][0] for n in names]))
            r["acc"][f"{grp}_mean"] = float(np.mean([r["acc"][f"{n}:{s}"][0] for n in names for s in (1, 2, 3)]))
        R["models"][m] = r
        print(f"{m:40s} clean {r['acc']['clean'][0]:.3f} sensor {r['acc']['sensor_mean']:.3f} "
              f"heldout {r['acc']['heldout_mean']:.3f} ECE clean {r['ece']['clean']:.3f}", flush=True)
    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1)
    figures(R)


def figures(R):
    M = R["models"]
    fam = {"supervised_std": ("supervised", "#999999", "-"), "supervised_sensor": ("supervised + sensor aug", "#555555", "--"),
           "linear_simclr_std": ("SimCLR, linear", "#8da0cb", "-"), "linear_simclr_sensor": ("SimCLR-sensor, linear", "#fc8d62", "-"),
           "finetune_simclr_std": ("SimCLR, fine-tuned", "#1f4e9c", "-"),
           "finetune_simclr_sensor": ("SimCLR-sensor, fine-tuned", "#c0392b", "-")}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3), sharey=True)
    for ax, key, title in zip(axs, ["clean", "sensor_mean", "heldout_mean"],
                              ["clean test images", "camera corruptions (seen in 'sensor' training)",
                               "held-out corruptions (never seen)"]):
        for f, (lab, c, ls) in fam.items():
            xs = [fr for fr in FRACS if f"{f}_f{fr:g}" in M]
            if not xs:
                continue
            v = [M[f"{f}_f{fr:g}"]["acc"][key] for fr in xs]
            v = [x[0] if isinstance(x, list) else x for x in v]
            ax.plot([100 * x for x in xs], v, "o" + ls, color=c, label=lab)
        ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1 %", "10 %", "100 %"])
        ax.set_xlabel("share of labelled training images"); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    axs[0].set_ylabel("test accuracy"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "label_efficiency.png"), dpi=130); plt.close()

    # per-corruption at severity 3, 10 % labels
    keys = [k for k in ["supervised_std_f0.1", "supervised_sensor_f0.1", "finetune_simclr_std_f0.1",
                        "finetune_simclr_sensor_f0.1"] if k in M]
    names = corrupt.SENSOR + corrupt.HELDOUT
    fig, ax = plt.subplots(figsize=(13, 4))
    w = 0.8 / max(len(keys), 1)
    cols = ["#999999", "#555555", "#1f4e9c", "#c0392b"]
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(names) + 1) + (i - (len(keys) - 1) / 2) * w,
               [M[k]["acc"]["clean"][0]] + [M[k]["acc"][f"{n}:3"][0] for n in names], w, color=cols[i],
               label=k.replace("_f0.1", "").replace("_", " "))
    ax.axvline(4.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(names) + 1))
    ax.set_xticklabels(["clean"] + [n.replace("_", " ") + (" *" if n in corrupt.HELDOUT else "") for n in names])
    ax.set_ylabel("accuracy (severity 3)"); ax.set_title("10 % labels, heaviest corruption (* = never seen in training)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "corruptions.png"), dpi=130); plt.close()

    # examples of the corruptions
    X, _ = data.load("test")
    fig, axs = plt.subplots(3, 8, figsize=(13, 5.2))
    for r, i in enumerate([7, 21, 0]):
        axs[r, 0].imshow(X[i]); axs[r, 0].set_title("clean" if r == 0 else "", fontsize=9)
        for c, n in enumerate(names):
            axs[r, c + 1].imshow(corrupt.apply(X[i], n, 3, np.random.default_rng(i * 100 + 3)))
            if r == 0:
                axs[r, c + 1].set_title(n.replace("_", " ") + (" *" if n in corrupt.HELDOUT else ""), fontsize=9)
    for a in axs.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "examples.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()
