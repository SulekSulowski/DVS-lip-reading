"""
test.py — weryfikacja pipeline'u EventLipReadingNet

Uruchom z katalogu projektu:
    python test.py

Nie wymaga datasetu DVS-Lip — wszystkie testy używają danych syntetycznych.
Na końcu jest sekcja do szybkiego testu na małym wycinku datasetu.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# ── importuj z twojego pliku utils ──────────────────────────────────────────
from utils import (
    GraphGen,
    EventLipReadingNet,
    DGCNN,
    TemporalBiGRU,
    build_graph_sequence,
    create_dataloader,
    load_data,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

passed = 0
failed = 0

def ok(name):
    global passed
    passed += 1
    print(f"  [PASS] {name}")

def fail(name, reason):
    global failed
    failed += 1
    print(f"  [FAIL] {name}: {reason}")

def check(name, condition, reason=""):
    if condition:
        ok(name)
    else:
        fail(name, reason or "assertion failed")


# ============================================================
# 1. GraphGen — format wyjścia
# ============================================================
print("=" * 55)
print("1. GraphGen - format wyjscia")
print("=" * 55)

# 20 syntetycznych eventów w oknie 32x32
np.random.seed(42)
N = 20
events_np = np.stack([
    np.random.randint(0, 32, N),   # x
    np.random.randint(0, 32, N),   # y
    np.arange(N, dtype=float),     # t
    np.random.choice([-1, 1], N),  # p
], axis=1).astype(np.float32)

events = torch.tensor(events_np, dtype=torch.float32, device=DEVICE)

gg = GraphGen(r=3, dimension_XY=32, device=DEVICE)
node_features, pos, edges = gg(events)

check("node_features dtype float32",
      node_features.dtype == torch.float32,
      f"got {node_features.dtype}")

check("node_features shape [N, 4]",
      node_features.ndim == 2 and node_features.shape[1] == 4,
      f"got {node_features.shape}")

check("pos shape [N, 3]",
      pos.ndim == 2 and pos.shape[1] == 3,
      f"got {pos.shape}")

check("edges shape [E, 2]",
      edges.ndim == 2 and edges.shape[1] == 2,
      f"got {edges.shape}")

check("edges dtype long",
      edges.dtype == torch.long,
      f"got {edges.dtype}")

# build_graph_sequence robi .t().contiguous() → [2, E] dla PyG
if edges.shape[0] > 0:
    edge_index = edges.t().contiguous()
    check("edge_index PyG format [2, E]",
          edge_index.shape[0] == 2,
          f"got {edge_index.shape}")

# polaryzacja jako ostatnia kolumna cech węzła
p_col = node_features[:, 3]
check("polaryzacja w {-1, 1}",
      set(p_col.unique().cpu().numpy().tolist()).issubset({-1.0, 1.0}),
      f"wartości: {p_col.unique().cpu().numpy()}")

# timestamp znormalizowany do [0, 127]
t_col = node_features[:, 2]
check("t_normalized w [0, 127]",
      t_col.min().item() >= 0.0 and t_col.max().item() <= 127.0 + 1e-4,
      f"min={t_col.min():.2f} max={t_col.max():.2f}")

print()


# ============================================================
# 2. build_graph_sequence - struktura sekwencji okien
# ============================================================
print("=" * 55)
print("2. build_graph_sequence")
print("=" * 55)

# syntetyczny strumień 1000 eventów, timestamps rosnące
N2 = 1000
t_stream = np.linspace(0, 200000, N2).astype(np.int64)
x_stream = np.random.randint(0, 128, N2).astype(np.int32)
y_stream = np.random.randint(0, 128, N2).astype(np.int32)
p_stream = np.random.choice([0, 1], N2).astype(np.int8)

window_size = 20000   # ~20ms jeśli jednostka czasu = 1µs
seq = build_graph_sequence(
    t_stream, x_stream, y_stream, p_stream,
    window_size=window_size,
    r=5,
    dimension_XY=128,
    device=DEVICE
)

check("sekwencja niepusta",
      len(seq) > 0,
      "zero okien — sprawdź window_size i format timestamps")

check("liczba okien ~ 10 (200ms / 20ms)",
      5 <= len(seq) <= 15,
      f"got {len(seq)} okien")

if len(seq) > 0:
    g0 = seq[0]
    check("Data ma atrybut x",
          hasattr(g0, 'x'),
          "brak g.x")
    check("Data.x dtype float32",
          g0.x.dtype == torch.float32,
          f"got {g0.x.dtype}")
    check("Data.edge_index shape [2, E]",
          g0.edge_index.ndim == 2 and g0.edge_index.shape[0] == 2,
          f"got {g0.edge_index.shape}")
    check("edge_index indeksy w zakresie wezlow",
          g0.edge_index.max().item() < g0.x.shape[0],
          f"max idx={g0.edge_index.max().item()} N={g0.x.shape[0]}")

print()


# ============================================================
# 3. DGCNN — forward pass na jednym grafie
# ============================================================
print("=" * 55)
print("3. DGCNN - forward pass")
print("=" * 55)

dgcnn = DGCNN(k=8).to(DEVICE)

if len(seq) > 0:
    g_test = seq[0].to(DEVICE)
    if g_test.batch is None:
        g_test.batch = torch.zeros(g_test.x.size(0), dtype=torch.long, device=DEVICE)

    try:
        with torch.no_grad():
            h = dgcnn(g_test)
        check("DGCNN output shape [1, 256]",
              h.shape == (1, 256),
              f"got {h.shape}")
        check("DGCNN output nie zawiera NaN",
              not torch.isnan(h).any().item(),
              "NaN w outputcie")
    except Exception as e:
        fail("DGCNN forward pass", str(e))

print()


# ============================================================
# 4. TemporalBiGRU - forward pass
# ============================================================
print("=" * 55)
print("4. TemporalBiGRU - forward pass")
print("=" * 55)

gru = TemporalBiGRU(input_dim=256, hidden_dim=256).to(DEVICE)

# symuluj sekwencję T=10 embeddingów, batch=1
fake_seq = torch.randn(1, 10, 256, device=DEVICE)
with torch.no_grad():
    z = gru(fake_seq)

check("BiGRU output shape [1, 512]",
      z.shape == (1, 512),
      f"got {z.shape}")
check("BiGRU output nie zawiera NaN",
      not torch.isnan(z).any().item(),
      "NaN w outputcie")

print()


# ============================================================
# 5. EventLipReadingNet — pełny forward pass
# ============================================================
print("=" * 55)
print("5. EventLipReadingNet - pelny forward pass")
print("=" * 55)

model = EventLipReadingNet(num_classes=100, k=8).to(DEVICE)

if len(seq) >= 3:
    # weź pierwsze 3 okna jako mock sekwencji
    mock_seq = [g.to(DEVICE) for g in seq[:3]]
    try:
        with torch.no_grad():
            logits = model(mock_seq)
        check("output shape [1, 100]",
              logits.shape == (1, 100),
              f"got {logits.shape}")
        check("output nie zawiera NaN",
              not torch.isnan(logits).any().item(),
              "NaN w logitach")
        check("argmax w zakresie [0, 99]",
              0 <= logits.argmax(dim=1).item() <= 99,
              f"got {logits.argmax(dim=1).item()}")
    except Exception as e:
        fail("EventLipReadingNet forward", str(e))
else:
    print("  [SKIP] za mało okien do testu pelnej sieci")

print()


# ============================================================
# 6. Backward pass — gradient flow
# ============================================================
print("=" * 55)
print("6. Backward pass - gradient flow")
print("=" * 55)

if len(seq) >= 3:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    mock_seq = [g.to(DEVICE) for g in seq[:3]]
    label = torch.tensor([42], dtype=torch.long, device=DEVICE)

    try:
        optimizer.zero_grad()
        logits = model(mock_seq)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        check("loss jest skalarem",
              loss.ndim == 0,
              f"got shape {loss.shape}")
        check("loss > 0",
              loss.item() > 0,
              f"loss = {loss.item()}")

        # sprawdź że gradienty popłynęły do DGCNN
        grads = [p.grad for p in model.dgcnn.parameters() if p.grad is not None]
        check("gradienty w DGCNN",
              len(grads) > 0,
              "brak gradientow w dgcnn")

        # sprawdź że gradienty popłynęły do BiGRU
        grads_gru = [p.grad for p in model.temporal.parameters() if p.grad is not None]
        check("gradienty w BiGRU",
              len(grads_gru) > 0,
              "brak gradientow w bigru")

    except Exception as e:
        fail("backward pass", str(e))

print()


# ============================================================
# 7. Lokalny sweep parametrów window_size i r
#    Zapisuje wykresy do folderu plots/
# ============================================================

import os
from pathlib import Path
import matplotlib.pyplot as plt


def build_limited_dataset(
    root,
    classes,
    window_size,
    r,
    device='cuda',
    files_per_class=5,
    dimension_XY=128,
):
    root = Path(root)
    samples = []

    for cls_idx, cls in enumerate(classes):
        class_dir = root / cls
        if not class_dir.exists():
            continue

        files = sorted(class_dir.glob("*.npy"))[:files_per_class]

        for npy_file in tqdm(files, desc=f"  {root.name}/{cls}"):
            try:
                time, x, y, p = load_data(str(npy_file))
                graph_sequence = build_graph_sequence(
                    time,
                    x,
                    y,
                    p,
                    window_size=window_size,
                    r=r,
                    dimension_XY=dimension_XY,
                    device=device,
                )

                if len(graph_sequence) > 0:
                    samples.append((graph_sequence, cls_idx))
            except Exception as exc:
                print(f"    błąd przy {npy_file.name}: {exc}")

    return samples


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for graph_sequences, labels in tqdm(loader, desc="Train"):
        labels = labels.to(device)

        for i, graph_sequence in enumerate(graph_sequences):
            optimizer.zero_grad()

            logits = model([graph.to(device) for graph in graph_sequence])
            loss = criterion(logits, labels[i:i + 1])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels[i:i + 1]).sum().item()
            total += 1

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for graph_sequences, labels in tqdm(loader, desc="Eval"):
            labels = labels.to(device)

            for i, graph_sequence in enumerate(graph_sequences):
                logits = model([graph.to(device) for graph in graph_sequence])
                loss = criterion(logits, labels[i:i + 1])

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels[i:i + 1]).sum().item()
                total += 1

    return total_loss / max(total, 1), correct / max(total, 1)


def run_local_param_sweep(
    train_root="train",
    test_root="test",
    window_sizes=(25000, 50000, 75000, 100000),
    r_values=(3, 5, 8),
    epochs=15,
    batch_size=8,
    classes_limit=5,
    k=8,
    device=DEVICE,
):
    train_root = Path(train_root)
    test_root = Path(test_root)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if not train_root.exists() or not test_root.exists():
        print("  [SKIP] brak folderu train lub test")
        return []

    train_classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
    classes = [cls for cls in train_classes if cls in test_classes][:classes_limit]

    if not classes:
        print("  [SKIP] brak wspólnych klas w train i test")
        return []

    print(f"  Klasy do lokalnego testu: {classes}")

    summary = []

    for window_size in window_sizes:
        for r in r_values:
            print()
            print("-" * 55)
            print(f"Sweep: window_size={window_size}, r={r}")
            print("-" * 55)

            train_samples = build_limited_dataset(
                train_root,
                classes,
                window_size=window_size,
                r=r,
                device=device,
                files_per_class=145,
            )
            test_samples = build_limited_dataset(
                test_root,
                classes,
                window_size=window_size,
                r=r,
                device=device,
                files_per_class=45,
            )

            if len(train_samples) == 0 or len(test_samples) == 0:
                print("  [SKIP] brak danych po zbudowaniu sekwencji")
                continue

            train_loader = create_dataloader(train_samples, batch_size=batch_size, shuffle=True)
            test_loader = create_dataloader(test_samples, batch_size=batch_size, shuffle=False)

            model = EventLipReadingNet(num_classes=len(classes), k=k).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            history = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": [],
            }

            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc)

                print(
                    f"Epoch {epoch + 1:02d}/{epochs} | "
                    f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                    f"Test loss: {test_loss:.4f} acc: {test_acc:.3f}"
                )

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(range(1, epochs + 1), history["train_loss"], label="train loss")
            axes[0].plot(range(1, epochs + 1), history["test_loss"], label="test loss")
            axes[0].set_title(f"Loss | ws={window_size}, r={r}")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(range(1, epochs + 1), history["train_acc"], label="train acc")
            axes[1].plot(range(1, epochs + 1), history["test_acc"], label="test acc")
            axes[1].set_title(f"Accuracy | ws={window_size}, r={r}")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()

            plot_path = plots_dir / f"sweep_ws{window_size}_r{r}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            final_metrics = {
                "window_size": window_size,
                "r": r,
                "train_loss": history["train_loss"][-1],
                "train_acc": history["train_acc"][-1],
                "test_loss": history["test_loss"][-1],
                "test_acc": history["test_acc"][-1],
                "plot": str(plot_path),
            }
            summary.append(final_metrics)

            print(
                f"  Zapisano wykres: {plot_path} | "
                f"final test acc={final_metrics['test_acc']:.3f}, "
                f"final test loss={final_metrics['test_loss']:.4f}"
            )

    if summary:
        best = max(summary, key=lambda item: item["test_acc"])
        print()
        print("Najlepsza konfiguracja lokalna:")
        print(
            f"  window_size={best['window_size']}, r={best['r']}, "
            f"test_acc={best['test_acc']:.3f}, test_loss={best['test_loss']:.4f}"
        )

    return summary


# ============================================================
# Podsumowanie
# ============================================================
print()
print("=" * 55)
print(f"WYNIK: {passed} passed, {failed} failed")
print("=" * 55)

if failed == 0:
    print("Wszystkie testy przeszly - pipeline gotowy do treningu.")
else:
    print("Napraw bledy przed uruchomieniem pelnego treningu.")


# ============================================================
# Pomocnicza funkcja do ablacji parametrów
# (uruchom ręcznie jeśli chcesz porównać window_size i r)
# ============================================================

def ablation_window_and_r(data_root="train", n_classes=5, n_files=148):
    """
    Szybka ablacja window_size i r na małym wycinku danych.
    Wypisuje średnią liczbę węzłów i okien dla każdej kombinacji.

    Użycie:
        python -c "from test import ablation_window_and_r; ablation_window_and_r()"
    """
    root = Path(data_root)
    if not root.exists():
        print("Brak folderu danych.")
        return

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])[:n_classes]
    files = []
    for cls in classes:
        files += list((root / cls).glob("*.npy"))[:n_files]

    window_sizes = [25000, 50000, 75000, 100000]
    r_values     = [3, 5, 8]

    print(f"\n{'window_size':>12} {'r':>4} {'okna_sr':>10} {'wezly_sr':>12}")
    print("-" * 44)

    for ws in window_sizes:
        for r in r_values:
            all_windows = []
            all_nodes   = []
            for f in files:
                try:
                    t, x, y, p = load_data(str(f))
                    gs = build_graph_sequence(t, x, y, p,
                                              window_size=ws, r=r,
                                              dimension_XY=128, device='cpu')
                    if gs:
                        all_windows.append(len(gs))
                        all_nodes.append(np.mean([g.x.shape[0] for g in gs]))
                except Exception:
                    pass

            if all_windows:
                print(f"{ws:>12} {r:>4} {np.mean(all_windows):>10.1f} "
                      f"{np.mean(all_nodes):>12.1f}")

if __name__ == "__main__":
    run_local_param_sweep(
        train_root="train",
        test_root="test",
        window_sizes=[25000, 50000, 75000, 100000],
        r_values=[3, 5, 8],
        epochs=15,
        batch_size=8,
        classes_limit=5,
        k=8,
        device=DEVICE,
    )