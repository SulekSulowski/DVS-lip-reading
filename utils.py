import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, Module
import matplotlib.pyplot as plt
from torch.nn import    Linear, \
                        BatchNorm1d, \
                        Module
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.pool import knn_graph
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader



def load_data(root):
    data = np.load(root)
    time = data["t"]
    x = data["x"]
    y = data["y"]
    p = data['p']

    return time, x, y, p

def split_into_windows(time, x, y, p, window_size):

    windows = []

    start_idx = 0
    start_time = time[0]

    for i in range(len(time)):

        if time[i] - start_time >= window_size:

            windows.append({
                "time": time[start_idx:i],
                "x": x[start_idx:i],
                "y": y[start_idx:i],
                "p": p[start_idx:i]
            })

            start_idx = i
            start_time = time[i]

    return windows


def draw_graph(pos, edges, dimension_XY, size=10, elev=30, azim=35):
    pos_np = pos.numpy()
    edges_np = edges.numpy()
       
    
    # Extract coordinates
    x, y, t = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(t, x, y, c=t, cmap='plasma', s=size, alpha=0.8) # 32-y to flip Y-axis
    
    # Draw edges
    for (src, dst) in edges_np:
        if src < len(pos_np) and dst < len(pos_np):
            x_line = [pos_np[src, 0], pos_np[dst, 0]]
            y_line = [pos_np[src, 1], pos_np[dst, 1]] # 32-y to flip Y-axis
            t_line = [pos_np[src, 2], pos_np[dst, 2]]
            ax.plot(t_line, x_line, y_line, color='gray', alpha=0.5, linewidth=(size*0.1))
    
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("3D Event Graph")
    fig.colorbar(sc, ax=ax, label='Normalized Time')
    
    # ax.view_init(elev=0, azim=0)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()


def build_graph_sequence(time, x, y, p, window_size, r, dimension_XY=128, device='cuda'):
    
    windows = split_into_windows(time, x, y, p, window_size)
    graph_gen = GraphGen(r=r, dimension_XY=dimension_XY, device=device)
    
    graph_sequence = []

    for window in windows:
        events = torch.tensor(
            np.stack([window['x'], window['y'], window['time'], window['p']], axis=1),
            dtype=torch.float32,
            device=device
        )

        node_features, pos, edges = graph_gen(events)

        if len(node_features) < 2:  # puste lub za małe okno — pomiń
            continue

        data = Data(
            x=node_features,                    # [N, 4]
            edge_index=edges.t().contiguous(),  # [2, E] — PyG wymaga tego formatu
            pos=pos                             # [N, 3] opcjonalnie
        )

        graph_sequence.append(data)

    return graph_sequence



class DVSLipDataset(Dataset):

    def __init__(self, samples):
        """
        samples:
        list of (graph_sequence, label)

        graph_sequence:
        list of PyG Data objects
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        graphs, label = self.samples[idx]
        return graphs, label


# =========================================================
# Collate function (IMPORTANT)
# =========================================================

def collate_fn(batch):
    """
    batch:
    list of (graph_sequence, label)
    """

    graph_sequences = []
    labels = []

    for graphs, label in batch:
        graph_sequences.append(graphs)
        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.long)

    return graph_sequences, labels


# =========================================================
# DataLoader
# =========================================================

def create_dataloader(samples, batch_size=1, shuffle=True):

    dataset = DVSLipDataset(samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0
    )

    return loader



class DGCNN(nn.Module):

    def __init__(self, k=16):

        super().__init__()

        self.k = k

        # R^4 -> R^64
        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * 4, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
            ), aggr='max'
        )

        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * 64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
            ), aggr='max'
        )

        self.conv3 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * 128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
            ), aggr='max'
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # -------------------------------------------------
        # Layer 1
        # Initial radius graph from GraphGen
        # -------------------------------------------------

        x = self.conv1(x, edge_index)

        # -------------------------------------------------
        # Layer 2
        # Dynamic graph in feature space
        # -------------------------------------------------

        edge_index = knn_graph(
            x,
            k=self.k,
            batch=batch
        )

        x = self.conv2(x, edge_index)

        # -------------------------------------------------
        # Layer 3
        # Dynamic graph in feature space
        # -------------------------------------------------

        edge_index = knn_graph(
            x,
            k=self.k,
            batch=batch
        )

        x = self.conv3(x, edge_index)

        # -------------------------------------------------
        # Global pooling
        # -------------------------------------------------

        h = global_max_pool(x, batch)

        return h


# =========================================================
# Temporal Module (BiGRU)
# =========================================================

class TemporalBiGRU(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, x):

        """
        x:
        [batch_size, sequence_length, 256]
        """

        output, h_n = self.gru(x)

        # last forward state
        h_forward = h_n[-2]

        # last backward state
        h_backward = h_n[-1]

        # concatenate directions
        h = torch.cat(
            [h_forward, h_backward],
            dim=1
        )

        return h



# =========================================================
# Full Lip Reading Network
# =========================================================

class EventLipReadingNet(nn.Module):

    def __init__(
        self,
        num_classes=100,
        k=16
    ):

        super().__init__()

        self.dgcnn = DGCNN(k=k)

        self.temporal = TemporalBiGRU()

        self.classifier = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, graph_sequence):

        """
        graph_sequence:
        list of PyG graphs

        Example:
        [graph_1, graph_2, ..., graph_T]
        """

        embeddings = []

        # -------------------------------------------------
        # Process each graph independently with DGCNN
        # -------------------------------------------------

        for graph in graph_sequence:

            h = self.dgcnn(graph)

            embeddings.append(h)

        # -------------------------------------------------
        # Stack temporal sequence
        # -------------------------------------------------

        x = torch.stack(
            embeddings,
            dim=1
        )

        """
        x shape:
        [batch_size, sequence_length, 256]
        """

        # -------------------------------------------------
        # Temporal modeling
        # -------------------------------------------------

        h = self.temporal(x)

        # -------------------------------------------------
        # Final classification
        # -------------------------------------------------

        logits = self.classifier(h)

        return logits


class GraphGen(Module):
    def __init__(self, r, dimension_XY=128, self_loop=True, device='cuda'):
        super().__init__()
        self.r = r
        self.dimension_XY = dimension_XY
        self.self_loop = self_loop
        self.device = device

        offsets = torch.arange(-r, r + 1, device=device)
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')
        mask = (dx ** 2 + dy ** 2) <= r ** 2
        self.ctx_offsets = torch.stack([dx[mask], dy[mask]], dim=1)

        self.pos_list, self.feat_list, self.edge_list = [], [], []
        self.neighbour_matrix = torch.full(
            (dimension_XY, dimension_XY), -1, dtype=torch.int32, device=device
        )
        self.index = 0

    def forward(self, events):
        for i in range(len(events)):
            x, y, t, feature = events[i]
            x, y, t, feature = int(x), int(y), float(t), float(feature)

            if not (0 <= x < self.dimension_XY and 0 <= y < self.dimension_XY):
                continue

            idx_existing = self.neighbour_matrix[x, y].item()
            if idx_existing != -1 and float(self.pos_list[idx_existing][2]) == t:
                continue

            self.pos_list.append((x, y, t))
            self.feat_list.append(feature)
            if self.self_loop:
                self.edge_list.append((self.index, self.index))

            ctx_xy = self.ctx_offsets + torch.tensor([x, y], device=self.device)
            ctx_xy = torch.clamp(ctx_xy, 0, self.dimension_XY - 1)
            idxes = self.neighbour_matrix[ctx_xy[:, 0], ctx_xy[:, 1]]
            valid_mask = idxes != -1
            idxes = idxes[valid_mask]

            if idxes.numel() > 0:
                neighbor_pos = torch.tensor([self.pos_list[j] for j in idxes.tolist()],
                                            device=self.device)
                diffs = neighbor_pos - torch.tensor([x, y, t], device=self.device)
                mask = (diffs[:, 0] ** 2 + diffs[:, 1] ** 2 + diffs[:, 2] ** 2) <= self.r ** 2
                for j in idxes[mask].tolist():
                    self.edge_list.append((self.index, j))

            self.neighbour_matrix[x, y] = self.index
            self.index += 1

        edge_list_dedup = list(dict.fromkeys(map(tuple, self.edge_list)))
        pos = torch.tensor(self.pos_list, dtype=torch.float32, device=self.device)   # [N, 3] — float32!
        pol = torch.tensor(self.feat_list, dtype=torch.float32, device=self.device).unsqueeze(1)  # [N, 1]

        # normalizacja t do skali [0, 128]
        xy = pos[:, :2] / 127.0          # x, y -> [0, 1]
        t_col = pos[:, 2:3]
        t_min, t_max = t_col.min(), t_col.max()
        t_norm = (t_col - t_min) / (t_max - t_min + 1e-8)  # t -> [0, 1]
        x = torch.cat([xy, t_norm, pol], dim=1)              # [N, 4]

        edges = torch.tensor(edge_list_dedup, dtype=torch.long, device=self.device) \
            if self.edge_list else torch.empty((0, 2), dtype=torch.long, device=self.device)

        self.reset()
        return x, pos, edges

    def reset(self):
        self.pos_list.clear()
        self.feat_list.clear()
        self.edge_list.clear()
        self.neighbour_matrix.fill_(-1)
        self.index = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dimension_XY}, r={self.r}, device={self.device})"