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


class GraphGen(Module):
    def __init__(self, r, dimension_XY=32, self_loop=True, device='cpu'):
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

        pos = torch.tensor(self.pos_list, dtype=torch.int32, device=self.device)
        x = torch.tensor(self.feat_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        edges = torch.tensor(edge_list_dedup, dtype=torch.int32, device=self.device) \
            if self.edge_list else torch.empty((0, 2), dtype=torch.int32, device=self.device)
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