# DVS Lip Reading

**Course project** - Dynamic Vision Sensors, AGH, 2026

## Project description

Lip reading from event camera data is a challenging recognition problem that sits at
the intersection of neuromorphic sensing and sequence learning. Unlike conventional
RGB cameras, Dynamic Vision Sensors (DVS) capture only brightness changes at the
pixel level with microsecond temporal resolution, producing sparse, asynchronous
event streams that are fundamentally different from standard video frames.

This project explores **graph-based spatiotemporal modeling** for word-level lip
reading on DVS event streams. We treat each event `(x, y, t, p)` as a node in a
dynamically constructed graph and apply graph neural network layers to extract
features from lip motion patterns captured by an event camera.

## Goal

Train and evaluate a **Graph Neutral Networks** on the DVS-Lip dataset - a 100-word
event-camera lip reading benchmark.


## Dataset

[DVS-Lip](https://sites.google.com/view/event-based-lipreading) - 100 words,
40 speakers, recorded with a DAVIS346 event camera.

---

## 1. Literature Review

### 1.1 General lip reading surveys

These papers provide a broad overview of automatic lip reading (ALR) methods, covering
both classical and deep learning approaches, and contextualize where event-based
cameras fit in the larger field.

- [Automatic visual lip reading: A comparative review of machine-learning approaches [1]](https://www.sciencedirect.com/science/article/pii/S2590123025032268) 
- [Spatiotemporal Feature Enhancement for Lip-Reading: A Survey [2]](https://www.mdpi.com/2076-3417/15/8/4142) 
- [Tackling Event-Based Lip-Reading by Exploring Multigrained Spatiotemporal Clues [3]](https://ieeexplore.ieee.org/document/10682067) 
- [Vision based Lip Reading System using Deep Learning [4]](https://ieeexplore.ieee.org/document/9776430) 
---

### 1.2 Event-based lip reading methods

Papers directly tackling lip reading with Dynamic Vision Sensors (DVS).


- [Multi-grained Spatio-Temporal Features Perceived Network for Event-based Lip-Reading [5]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf) 
- [EventLip: Enhancing Event-Based Lip Reading via Frequency-Aware Spatiotemporal Hypergraph Modeling [6]](https://dl.acm.org/doi/pdf/10.1145/3746027.3755404) 
- [MTGA: Multi-View Temporal Granularity Aligned Aggregation [7]](https://arxiv.org/pdf/2404.11979) 
- [Mamba-Based Temporal Modeling for Event-Based Lip Reading [8]](https://ieeexplore.ieee.org/document/11085858) 
- [Semantics-aware high-frequency enhancement for event-based lip-reading [9]](https://www.sciencedirect.com/science/article/pii/S0020025525011636) 
- [NeuroLip: Event-driven Spatiotemporal Learning for Lip-Motion VSR [10]](https://arxiv.org/abs/2604.15718) 

---

### 1.3 Graph neural networks for event-based vision

These methods form the theoretical backbone for graph-based approaches we plan to
explore. They show how GNNs, GATs, and hypergraph networks handle spatio-temporal
event data in adjacent tasks (gesture/action recognition).

- [Space-Time Event Clouds for Gesture Recognition: From RGB Cameras to Event Cameras [11]](https://ieeexplore.ieee.org/document/8659288)  
- [Graph-Based Spatio-Temporal Feature Learning for Neuromorphic Vision Sensing [12]](https://ieeexplore.ieee.org/document/9199543) 
- [Event-Stream Representation for Human Gaits Identification Using Deep Neural Networks [13]](https://ieeexplore.ieee.org/document/9337225) 
- [Hypergraph-based Multi-View Action Recognition using Event Cameras [14]](https://arxiv.org/abs/2403.19316) 
- [Dynamic Graph CNN for Learning on Point Clouds [15]](https://arxiv.org/pdf/1801.07829)

---

### 1.4 Complementary methods 

Alternative or supporting approaches relevant to spatiotemporal feature enhancement.

- [Intelligent event-based lip reading word classification with spiking neural networks using spatio-temporal attention features and triplet loss [16]](https://www.sciencedirect.com/science/article/pii/S0020025524005735) 
- [Spectrum-guided Spatial Feature Enhancement Network for event-based lip-reading [17]](https://www.sciencedirect.com/science/article/pii/S0925231225006460) 

---

## 2. Datasets


**DVS-Lip** Public - [project page](https://sites.google.com/view/event-based-lipreading) |

**DVS-LRW100**  Not public 

For this project we focus exclusively on **DVS-Lip**, as DVS-LRW100 is not publicly available.

**DVS-LRW** - [RGB Dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

---

## 3. Selection of Methods for Further Work

- **Spatiotemporal Feature Enhancement Based on Convolutional Neural Networks**  
  Convert event streams into frame-like voxel grids or event images and use 2D/3D CNNs to learn local motion and appearance patterns from lip movements.

- **Spatiotemporal Feature Enhancement Based on Spiking Neural Networks**  
  Use biologically inspired spike-based neurons that naturally process asynchronous DVS events with low latency and energy efficiency.

- **Spatiotemporal Feature Enhancement Based on Graph Neural Networks**  
  Represent events as nodes connected in space and time, then apply graph message passing to model dynamic lip motion relationships.

- **Transformer-Based Sequence Modeling**  
  Use self-attention mechanisms to capture long-range temporal dependencies between lip motion events across the whole sequence.

- **Hypergraph Neural Networks**  
  Extend standard graphs by connecting multiple related events in one hyperedge, enabling modeling of higher-order spatial and temporal interactions.

- **Mamba / State Space Models for Temporal Modeling**  
  Efficient sequence models designed for long event streams, offering lower computational cost than Transformers while preserving temporal context.

- **Hybrid CNN + GNN Architectures**  
  Combine CNN feature extraction on voxelized frames with GNN reasoning over event relations to exploit both dense and sparse representations.

- **Frequency-Domain Feature Enhancement**  
  Transform temporal event signals into spectral space to emphasize repetitive motion patterns and suppress noise.

- **Multiview / Multi-Granularity Temporal Aggregation**  
  Learn features at different temporal resolutions (short-term and long-term motion) and combine them for better recognition accuracy.

- **Contrastive / Self-Supervised Pretraining**  
  Pretrain encoders on unlabeled event streams using representation learning, then fine-tune on the lip reading classification task.

---

## 4. Selected Approach

### 4.1 Overview

We build directly on the graph-based event processing framework introduced in the
course labs (GNN) and extend it in two ways:

1. **From single-graph classification to sequence modeling** - DVS-Lip samples are
   full word utterances (0.5–1 s), so we decompose each sample into a sequence of
   fixed-duration sliding windows. Each window produces one graph, processed by a
   shared GNN frontend. The resulting sequence of per-window embeddings is passed to
   a [Bi-GRU [7]](https://arxiv.org/pdf/2404.11979) for temporal modeling.

2. **Replacing PointNetConv with** [EdgeConv (DGCNN)[15]](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.EdgeConv.html) - unlike PointNetConv, which
   builds messages from raw coordinates `[x_j || p_j − p_i]`, EdgeConv recomputes
   the k-nearest-neighbor graph in *feature space* at each layer: `[x_j || x_i − x_j]`.
   This makes later layers sensitive to learned feature similarity, not just spatial
   proximity - better suited for capturing lip shape dynamics across layers.

### 4.2 Starting point: lab code

The `GraphGen`, `PointNetConv`, and `GraphPoolOut2D` classes from the course lab
GNN serve as our implementation baseline. The
following modifications are required to adapt them to DVS-Lip:

| Component | Lab version | Project version |
|---|---|---|
| `GraphGen` | single sample, reset per call | wrapped in sliding-window loop |
| `PointNetConv` | used as-is | optionally replaced by `EdgeConv` (PyG) |
| `GraphPoolOut2D` | 2D spatial pooling → flat vector | replaced by global max-pool → embedding |
| `GCN.linear` | `→ 10 classes` | `→ 100 classes` |
| temporal modeling | none (single graph) | Bi-GRU over window sequence |

### 4.3 Ablation plan

| Variant | Spatial conv | Temporal | Notes |
|---|---|---|---|
| **Baseline** | PointNetConv (lab) | Bi-GRU | Minimal change from lab code |
| **DGCNN** | EdgeConv (PyG) | Bi-GRU | Main proposed method |
| **DGCNN + GAT** | EdgeConv + GATConv | Bi-GRU | Attention ablation |
| **DGCNN + Transformer** | EdgeConv (PyG) | Transformer | Temporal ablation |

---

## 6th May


### Plan for the Next Milestone
- Visualize data from DVS-Lip dataset (DVS-LRW 100 dataset is not publicly available so we can't compare them) - Jakub Sulowski
- Search for more articles about DVS lip reading and add it to literature - Szczepan Tokarczyk
- Add more approaches of work methods (currently it is too general) - Miłosz Senator
- Select method that we will use in our project and describe it shortly - Jakub Sulowski
- Add goal of the project, short description and what we want to achieve at the begin of readme file - Szczepan Tokarczyk and Miłosz Senator


---

## 13th May


### Plan for the Next Milestone
- Prepare architecture of whole pipeline - Miłosz Senator and Szczepan Tokarczyk
- Implement solution in python - Jakub Sulowski

