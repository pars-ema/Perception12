# Final_Project/scripts/trainer.py
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

TRAIN_DIR = "Final_Project/data/classifier/train"
OUT_MODEL = "Final_Project/models/classifier_resnet18.pth"

BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
NUM_CLASSES = 3

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = ImageFolder(VAL_DIR, transform=val_tf)

    print("Class mapping:", train_ds.class_to_idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=LR)

    best_val = 0.0
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

    for ep in range(EPOCHS):
        # ---- train ----
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / total
        train_acc  = correct / total

        # ---- val ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        val_acc = correct / total if total else 0.0
        print(f"Epoch {ep+1}/{EPOCHS} | train loss {train_loss:.3f} acc {train_acc:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx
            }, OUT_MODEL)
            print("  saved best model")

    print("Training complete. Best val acc:", best_val)
    print("Model saved to:", OUT_MODEL)


if __name__ == "__main__":
    # Needed on Windows when using multiprocessing/spawn
    main()
