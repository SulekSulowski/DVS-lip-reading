import torch
from pathlib import Path
from utils import load_data
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def events_to_frame(time, x, y, p, dimension_XY=128):
    """
    Zamienia surowe eventy na tensor [2, H, W] —
    kanał 0: eventy z p=0, kanał 1: eventy z p=1
    """
    frame = torch.zeros(2, dimension_XY, dimension_XY)
    for xi, yi, pi in zip(x, y, p):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < dimension_XY and 0 <= yi < dimension_XY:
            ch = 1 if pi > 0 else 0
            frame[ch, yi, xi] += 1
    # normalizacja
    if frame.max() > 0:
        frame = frame / frame.max()
    return frame


def build_frame_dataset(root, window_size=50000, dimension_XY=128, classes_limit=5, files_per_class=10):
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])[:classes_limit]
    samples = []
    for cls_idx, cls in enumerate(classes):
        for f in sorted((root / cls).glob("*.npy"))[:files_per_class]:
            time, x, y, p = load_data(str(f))
            frame = events_to_frame(time, x, y, p, dimension_XY)
            samples.append((frame, cls_idx))
    return samples

def main_MLP():
    train_data = build_frame_dataset("train", files_per_class=50, classes_limit=5)
    test_data  = build_frame_dataset("test",  files_per_class=20, classes_limit=5)

    X_train = torch.stack([s[0] for s in train_data])
    y_train = torch.tensor([s[1] for s in train_data], dtype=torch.long)
    X_test  = torch.stack([s[0] for s in test_data])
    y_test  = torch.tensor([s[1] for s in test_data], dtype=torch.long)

    train_loader_mlp = TorchDataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader_mlp  = TorchDataLoader(TensorDataset(X_test,  y_test),  batch_size=16, shuffle=False)

    mlp = SimpleMLP(num_classes=5).to(DEVICE)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    criterion_mlp = nn.CrossEntropyLoss()

    for epoch in range(50):
        mlp.train()
        correct, total, total_loss = 0, 0, 0
        for X_batch, y_batch in train_loader_mlp:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer_mlp.zero_grad()
            logits = mlp(X_batch)
            loss = criterion_mlp(logits, y_batch)
            loss.backward()
            optimizer_mlp.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        train_acc = correct / total

        mlp.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader_mlp:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = mlp(X_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        test_acc = correct / total

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | train acc: {train_acc:.3f} | test acc: {test_acc:.3f}")


class SimpleMLP(nn.Module):
    def __init__(self, num_classes=5, dimension_XY=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * dimension_XY * dimension_XY, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

