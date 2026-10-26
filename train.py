
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from dataset import get_data
from model import FashionCNN

# Shared experiment settings
SEED = 42
EPOCHS = 3  # Phase 1 test; change to 30 in Phase 2
LEARNING_RATE = 0.001

torch.manual_seed(SEED)

# CPU is a safe default for the first test
device = torch.device("cpu")

train_loader, val_loader, test_loader = get_data()

model = FashionCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

Path("results").mkdir(exist_ok=True)
history = []

for epoch in range(EPOCHS):
    start = time.perf_counter()

    # Training
    model.train()
    total_loss = 0
    total_correct = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        total_correct += (
            predictions.argmax(1) == labels
        ).sum().item()
        total_train += n

    train_loss = total_loss / total_train
    train_accuracy = total_correct / total_train

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            n = labels.size(0)
            val_loss += loss.item() * n
            val_correct += (
                predictions.argmax(1) == labels
            ).sum().item()
            val_total += n

    val_loss /= val_total
    val_accuracy = val_correct / val_total
    duration = time.perf_counter() - start

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch_seconds": duration
    })

    print(
        f"Epoch {epoch + 1}: "
        f"Train Acc={train_accuracy:.2%}, "
        f"Val Acc={val_accuracy:.2%}, "
        f"Val Loss={val_loss:.4f}, "
        f"Time={duration:.1f}s"
    )

# Save experiment results
with open("results/history.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=history[0].keys()
    )
    writer.writeheader()
    writer.writerows(history)

torch.save(model.state_dict(), "results/model.pth")

print("Training completed. Results saved.")
