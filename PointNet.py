from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import load_data, build_graph_sequence
from pathlib import Path
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PointNetLipReading(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.conv1 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(4 + 3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ),
            global_nn=nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
            )
        )

        self.conv2 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(128 + 3, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
            ),
            global_nn=nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        ratio1 = min(0.5, 32 / pos.size(0))
        idx1 = fps(pos, batch, ratio=ratio1)
        row, col = radius(pos, pos[idx1], 10.0, batch, batch[idx1], max_num_neighbors=32)
        edge_index1 = torch.stack([col, row], dim=0)
        x1 = self.conv1(x, (pos, pos[idx1]), edge_index1)
        pos1, batch1 = pos[idx1], batch[idx1]

        ratio2 = min(0.5, 16 / max(pos1.size(0), 1))
        idx2 = fps(pos1, batch1, ratio=ratio2)
        row2, col2 = radius(pos1, pos1[idx2], 20.0, batch1, batch1[idx2], max_num_neighbors=32)
        edge_index2 = torch.stack([col2, row2], dim=0)
        x2 = self.conv2(x1, (pos1, pos1[idx2]), edge_index2)
        batch2 = batch1[idx2]

        h = global_max_pool(x2, batch2)
        return self.classifier(h)
    

def run_pointnet(
    train_root="train",
    test_root="test",
    window_size=50000,
    r=5,
    epochs=100,
    files_per_class=148,
    classes_limit=5,
    device=DEVICE,
):

    root_train = Path(train_root)
    root_test = Path(test_root)
    classes = sorted([d.name for d in root_train.iterdir() if d.is_dir()])[:classes_limit]
    print(f"Klasy: {classes}")

    # buduj dataset — tylko pierwsze okno z każdego nagrania
    def build(root, files_per_class):
        samples = []
        for cls_idx, cls in enumerate(classes):
            for f in tqdm(sorted((root / cls).glob("*.npy"))[:files_per_class], desc=f"{root.name}/{cls}"):
                try:
                    t, x, y, p = load_data(str(f))
                    gs = build_graph_sequence(t, x, y, p,
                                              window_size=window_size, r=r,
                                              dimension_XY=128, device=device)
                    if gs:
                        samples.append((gs[0], cls_idx))
                except Exception as e:
                    print(f"  błąd: {e}")
        return samples

    train_samples = build(root_train, files_per_class = 145)
    test_samples  = build(root_test,  files_per_class = 45)
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    def collate(batch):
        graphs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        from torch_geometric.data import Batch
        return Batch.from_data_list(graphs), labels

    from torch.utils.data import DataLoader as TorchDataLoader
    train_loader = TorchDataLoader(train_samples, batch_size=16, shuffle=True,  collate_fn=collate)
    test_loader  = TorchDataLoader(test_samples,  batch_size=16, shuffle=False, collate_fn=collate)

    model     = PointNetLipReading(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history_train_acc  = []
    history_test_acc   = []
    history_train_loss = []

    for epoch in range(epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0
        for graphs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            graphs, labels = graphs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(graphs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            total_loss += loss.item()

        train_acc  = correct / total
        train_loss = total_loss / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for graphs, labels in test_loader:
                graphs, labels = graphs.to(device), labels.to(device)
                logits = model(graphs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        test_acc = correct / total

        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        history_train_loss.append(train_loss)

        print(f"Epoch {epoch+1:03d}/{epochs} | loss={train_loss:.4f} | train acc={train_acc:.3f} | test acc={test_acc:.3f}")

    # wykres
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, epochs+1), history_train_acc,  label="train acc",  color="blue")
    ax.plot(range(1, epochs+1), history_test_acc,   label="test acc",   color="orange")
    ax.plot(range(1, epochs+1), history_train_loss, label="train loss", color="red")
    ax.set_title("PointNet — train/test acc i loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = plots_dir / "pointnet_training.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {plot_path}")