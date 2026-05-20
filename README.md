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

## 5. Architecture
The model processes a DVS event stream in three sequential stages: graph construction, spatial feature extraction, and temporal classification.

### Stage 1 - Sliding window decomposition
Each DVS-Lip sample (a single spoken word, 0.5–1 s) is split into T fixed-duration windows of Δt. This temporal segmentation strategy follows the approach of event-frame aggregation used in event-based lip reading [MTGA](https://arxiv.org/pdf/2404.11979), where the event stream is divided into discrete temporal segments to enable structured spatial processing. Each window contains a subset of the raw event stream and is processed independently by the shared GNN frontend.

### Stage 2 - Graph construction (GraphGen)
For each window, raw events (x, y, p) are treated as nodes in a graph. Edges are drawn between spatially proximate events using a radius-based rule: two nodes are connected if their Euclidean distance in the (x, y) plane is at most r. This initial graph captures local spatial structure within each window and serves as input to the first EdgeConv layer. Each node carries a 4-dimensional feature vector (x, y, t, p), where x, y ∈ [0, 127] are the pixel coordinates, t ∈ [0, 127] is the timestamp normalized to the spatial scale of the sensor, and p ∈ {−1, +1} is the event polarity. This graph construction strategy follows [DGCNN](https://arxiv.org/pdf/1801.07829), which demonstrates that building a kNN graph directly over point cloud data - and recomputing it dynamically at each layer - is more effective than fixed graph structures for learning local geometric features. The radius r is a hyperparameter controlling the density of the initial graph — larger r connects more distant events at the cost of noisier edges.

### Stage 3 - Spatial feature extraction (DGCNN / EdgeConv)
The graph passes through N = 3 stacked EdgeConv layers, as introduced in [DGCNN](https://arxiv.org/pdf/1801.07829). Each layer computes, for every node i and its current neighbors j:

    h_i = max_{j ∈ N(i)}  MLP( [ h_i  ||  h_i − h_j ] )

where || denotes concatenation and MLP is a shared two-layer perceptron with
BatchNorm and ReLU. Crucially, the k-nearest-neighbor graph is **recomputed before
each layer** in the current feature space - not in the original (x, y) coordinate
space. This is the "dynamic" aspect of DGCNN: early layers find neighbors by spatial
proximity, while deeper layers find neighbors by learned feature similarity,
allowing the network to associate events from different lip regions that exhibit
correlated dynamics.

The per-layer output dimensions are:

    Input           →     EdgeConv 1  →   EdgeConv 2   →  EdgeConv 3
    R⁴ (x, y, t, p)       R⁶⁴             R¹²⁸            R²⁵⁶

After the final EdgeConv layer, global max-pooling aggregates all node features
into a single fixed-size window embedding:

    h_i = GlobalMaxPool( {node features} )  ∈  R²⁵⁶

## Stage 4 — Global Pooling (Graph Embedding)

After the three EdgeConv layers, each node in the graph holds a feature vector of dimension $\mathbb{R}^{256}$. These per-node features need to be reduced to a single fixed-size representation of the entire temporal window, regardless of how many events it contains.
We apply global max-pooling across all nodes:

$$
h = \text{GlobalMaxPool}\left(\left\lbrace h_i^{(N)} \mid i \in V \right\rbrace\right) \in \mathbb{R}^{256}
$$

The choice of max as the aggregation function follows directly from its symmetry properties. As shown in Dynamic Graph CNN for Learning on Point Clouds [15], the output of an EdgeConv layer is invariant to permutation of the input because max is a symmetric function, and this holds for both the within-layer edge aggregation and the global max-pooling over node features. This matters here because DVS event streams are inherently unordered - the order in which events are placed into the graph structure should not affect the resulting window embedding.
A closely related aggregation strategy appears in Space-Time Event Clouds for Gesture Recognition: From RGB Cameras to Event Cameras [11] for event-based gesture recognition, where a symmetric max-pooling function is applied to aggregate local features into a global descriptor of the point cloud, with the network learning to select informative points from the input set. Although that work uses PointNet rather than DGCNN as its backbone, the pooling principle is the same and directly motivates our design choice.
It is also worth noting that this graph-based approach differs from standard frame-based methods in how spatial information is preserved prior to pooling. As discussed in the survey by Sun et al. [2], frame-based representations aggregate events into fixed-resolution maps before any learned processing, which discards part of the fine-grained temporal structure. In the proposed approach, the polarity p and spatial coordinates (x,y) are retained as node attributes throughout the EdgeConv stages and are only collapsed into a single vector at the pooling step.
The resulting embedding

$$
h \in \mathbb{R}^{256}
$$

summarizes the spatial activity of the lip region within a single 20 ms window. It encodes the local geometric structure learned by the stacked EdgeConv layers, and does not depend on the number or ordering of events within that window.


## Stage 5 — Sequence of Window Embeddings

The \(T\) per-window embeddings produced by the global pooling step are arranged in chronological order to form a sequence:

$$
\mathbf{H} = [h_1, h_2, \ldots, h_T] \in \mathbb{R}^{T \times 256}
$$

This sequence serves as the interface between the spatial frontend (DGCNN) and the temporal backend (Bi-GRU). Each hth_t
ht​ can be thought of as a compact description of lip geometry within one 20 ms window, analogous to the per-segment feature vectors produced by the frontend in MTGA [7], where the result embeddings of different temporal segments are fed into the Bi-GRU and Self-Attention layers to exploit global temporal information.

The key difference in the approach proposed here is the nature of the individual embeddings. In MTGA, segment-level features are derived from a combination of event frames and voxel graph representations, where aggregating events into event frames inevitably leads to the loss of fine-grained temporal information within frames. In our pipeline, each $\mathbb{h_T}$​ is produced entirely from a graph-based representation so the within-window event structure is preserved through the (x,y) node coordinates and polarity attributes up until the pooling step.

One practical consideration at this stage is the choice of window duration Δt and, consequently, the length T of the sequence. Shorter windows yield longer sequences and finer temporal resolution, but also increase the total number of graph construction and EdgeConv operations per sample. The value Δt=20 ms was chosen as a reasonable trade-off given typical utterance lengths in the DVS-Lip dataset (0.5–1 s), resulting in sequences of roughly T=25–50 embeddings per sample.


## Stage 6 — Temporal Modeling (Bi-GRU)

The sequence of window embeddings

$$
[h_1, h_2, \ldots, h_T]
$$

is passed into a bidirectional GRU with hidden size 256 per direction. The role of this module is to model how lip shape evolves over the duration of the utterance - something that cannot be inferred from any single window embedding in isolation.

The use of a Bi-GRU as the temporal backend follows the design of MTGA [7], where a Bi-GRU is employed to aggregate temporal information, obtaining feature sequences that contain temporal context, motivated by its strong context learning and sequence modeling capabilities for word recognition. Bidirectionality is important here because in many cases the interpretation of lip movement at a given timestep depends on what comes both before and after it - a unidirectional model would miss context from the second half of the utterance when processing the first.

The forward and backward hidden states are concatenated to produce an utterance-level representation:

$$
z =
\left[
\text{GRU}_{\text{fwd}}(h_T)
\;\|\;
\text{GRU}_{\text{bwd}}(h_1)
\right]
\in \mathbb{R}^{512}
$$

The outputs from the forward and backward GRU layers are combined by concatenation, resulting in a final output dimension that is twice the hidden size of a single GRU layer, which increases the model's capacity to represent sequence information as it retains distinct features from both directions. In our case this gives a 512-dimensional vector, since each direction uses a hidden size of 256.

## Stage 7 — Classification

A single linear layer projects the utterance representation $z$ onto the 100 output classes of the DVS-Lip vocabulary:

$$
\hat{y} = \text{softmax}(Wz + b) \in \mathbb{R}^{100}
$$

The network is trained using **cross-entropy loss**.


# Summary

| Stage | Module | Input | Output | Reference |
|---|---|---|---|---|
| Window Split | Sliding window ($\Delta t = 20\,\text{ms}$) | Event stream | $T$ windows | MTGA |
| Graph Construction | GraphGen (radius, r) | Window events | Graph (V,E) | — |
| Spatial Features   | EdgeConv ×3 (DGCNN)  | Graph         | Node features ∈ R²⁵⁶ | DGCNN |
| Pooling | Global Max Pooling | $N \times \mathbb{R}^{256}$ | $\mathbb{R}^{256}$ | DGCNN |
| Temporal Modeling | Bi-GRU ($hidden=256$) | $T \times \mathbb{R}^{256}$ | $\mathbb{R}^{512}$ | MTGA |
| Classification | Linear + Softmax | $\mathbb{R}^{512}$ | $\mathbb{R}^{100}$ | MTGA |

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
- Prepare architecture and description of whole pipeline - Miłosz Senator and Szczepan Tokarczyk
- Implement solution in python - Jakub Sulowski

---

## 20th May


### Plan for the Next Milestone
- Prepare small dataset for checking correctness of our model - Jakub Sulowski
- Run model validation, save and analyse results for different parameters - Miłosz Senator and Szczepan Tokarczyk
- Set best parameters, fix model if incorrect - Jakub Sulowski

