import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_linear_eval_loaders
from model import Encoder, LinearClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ENCODER_PATH = os.path.join(RESULTS_DIR, "encoder_pretrained.pt")


def linear_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_linear_eval_loaders(batch_size=256)

    # ─── بارگذاری encoder از پیش آموزش‌دیده و فریز کردنش ───
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = LinearClassifier(in_dim=encoder.feature_dim, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    num_epochs = 15
    history = {"train_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = encoder(images)

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        classifier.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                outputs = classifier(features)
                correct += (outputs.argmax(1) == labels).sum().item()

        test_acc = correct / len(test_loader.dataset)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc

    with open(os.path.join(RESULTS_DIR, "linear_eval_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "linear_eval_summary.json"), "w") as f:
        json.dump({"best_linear_eval_accuracy": best_acc}, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(history["test_acc"])
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.title("Linear Evaluation on Frozen SimCLR Features")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "linear_eval_curve.png"))

    print(f"\nBest linear evaluation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    linear_eval()
