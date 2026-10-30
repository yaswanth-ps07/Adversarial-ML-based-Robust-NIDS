import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import *
from src.model import NIDSDNN
from src.adversarial_attacks import fgsm
from src.train import seed_all
from src.visualization import save_history


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_epoch(model, loader, optimizer):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        optimizer.zero_grad()

        # Generate adversarial examples against the CURRENT model.
        x_adv = fgsm(
            model,
            x,
            y,
            FGSM_EPSILON,
        )

        # Train on both clean and adversarial examples.
        x_mix = torch.cat([x, x_adv], dim=0)
        y_mix = torch.cat([y, y], dim=0)

        logits = model(x_mix)
        loss = loss_fn(logits, y_mix)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_mix)
        predictions = (torch.sigmoid(logits) >= 0.5)
        total_correct += (predictions == y_mix.bool()).sum().item()
        total_samples += len(y_mix)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, loader):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * len(y)
            predictions = (torch.sigmoid(logits) >= 0.5)
            total_correct += (predictions == y.bool()).sum().item()
            total_samples += len(y)

    return total_loss / total_samples, total_correct / total_samples


def main():
    seed_all()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    z = np.load(ARTIFACT_DIR / "splits.npz")

    X_train = z["X_train"]
    y_train = z["y_train"]
    X_val = z["X_val"]
    y_val = z["y_val"]

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)

    model = NIDSDNN(X_train.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience = 4
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "robust_dnn.pt",
            )
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping.")
            break

    history_path = METRIC_DIR / "robust_history.json"
    history_path.write_text(
        json.dumps(history, indent=2)
    )

    save_history(
        history,
        FIGURE_DIR / "robust_history_loss.png",
    )

    print("Robust adversarial training complete.")


if __name__ == "__main__":
    main()