import os, glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from utils import     GraphGen, EventLipReadingNet, \
                        draw_graph, load_data, create_dataloader, build_graph_sequence


def build_dataset(root, window_size=50000, r=5, device='cuda'):
    root = Path(root)
    
    # folder name → index (posortowane żeby było deterministyczne)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    samples = []

    for cls in tqdm(classes, desc="Loading classes"):
        label = class_to_idx[cls]
        
        for npy_file in (root / cls).glob("*.npy"):
            time, x, y, p = load_data(str(npy_file))
            graph_sequence = build_graph_sequence(
                time, x, y, p,
                window_size=window_size,
                r=r,
                device=device
            )
            
            if len(graph_sequence) == 0:
                continue
            
            samples.append((graph_sequence, label))

    return samples, class_to_idx

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for graph_sequences, labels in tqdm(loader, desc="Train"):
        labels = labels.to(device)

        for i, graph_sequence in enumerate(graph_sequences):
            optimizer.zero_grad()

            logits = model(graph_sequence)
            loss = criterion(logits, labels[i:i+1])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels[i:i+1]).sum().item()
            total += 1

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for graph_sequences, labels in tqdm(loader, desc="Eval"):
            labels = labels.to(device)

            for i, graph_sequence in enumerate(graph_sequences):
                logits = model(graph_sequence)
                loss = criterion(logits, labels[i:i+1])

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels[i:i+1]).sum().item()
                total += 1

    return total_loss / total, correct / total


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buduj datasety (to zajmie chwilę)
train_samples, class_to_idx = build_dataset(os.path.join(os.getcwd(), "train"), device=device)
test_samples, _             = build_dataset(os.path.join(os.getcwd(), "test"),  device=device)

train_loader = create_dataloader(train_samples, batch_size=8, shuffle=True)
test_loader  = create_dataloader(test_samples,  batch_size=1, shuffle=False)

model     = EventLipReadingNet(num_classes=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1:02d} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
          f"Test loss: {test_loss:.4f} acc: {test_acc:.3f}")