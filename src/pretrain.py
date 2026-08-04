import os
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_pretrain_loader
from model import SimCLRModel
from loss import nt_xent_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_pretrain_loader(batch_size=256)

    model = SimCLRModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    num_epochs = 30
    history = {"loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for view1, view2 in train_loader:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * view1.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        history["loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Contrastive Loss: {epoch_loss:.4f}")

    torch.save(model.encoder.state_dict(), os.path.join(RESULTS_DIR, "encoder_pretrained.pt"))

    with open(os.path.join(RESULTS_DIR, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"])
    plt.xlabel("Epoch"); plt.ylabel("NT-Xent Loss")
    plt.title("SimCLR Pretraining Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pretrain_loss_curve.png"))
    print("\nPretraining complete. Encoder saved.")


if __name__ == "__main__":
    pretrain()
