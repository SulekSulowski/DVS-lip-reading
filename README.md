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

## 1. Literature Review

### 1.1 General lip reading surveys

These papers provide a broad overview of automatic lip reading (ALR) methods, covering
both classical and deep learning approaches, and contextualize where event-based
cameras fit in the larger field.

- [Automatic visual lip reading: A comparative review of machine-learning approaches](https://www.sciencedirect.com/science/article/pii/S2590123025032268) 
- [Spatiotemporal Feature Enhancement for Lip-Reading: A Survey](https://www.mdpi.com/2076-3417/15/8/4142) 
- [Tackling Event-Based Lip-Reading by Exploring Multigrained Spatiotemporal Clues](https://ieeexplore.ieee.org/document/10682067) 
- [Vision based Lip Reading System using Deep Learning](https://ieeexplore.ieee.org/document/9776430) 
---

### 1.2 Event-based lip reading methods

Papers directly tackling lip reading with Dynamic Vision Sensors (DVS).


- [Multi-grained Spatio-Temporal Features Perceived Network for Event-based Lip-Reading](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf) 
- [EventLip: Enhancing Event-Based Lip Reading via
Frequency-Aware Spatiotemporal Hypergraph Modeling ](https://dl.acm.org/doi/pdf/10.1145/3746027.3755404) 
- [MTGA: Multi-View Temporal Granularity Aligned Aggregation](https://arxiv.org/pdf/2404.11979) 
- [Mamba-Based Temporal Modeling for Event-Based Lip Reading](https://ieeexplore.ieee.org/document/11085858) 
- [Semantics-aware high-frequency enhancement for event-based lip-reading](https://www.sciencedirect.com/science/article/pii/S0020025525011636) 
- [NeuroLip: Event-driven Spatiotemporal Learning for Lip-Motion VSR](https://arxiv.org/abs/2604.15718) 

---

### 1.3 Graph neural networks for event-based vision

These methods form the theoretical backbone for graph-based approaches we plan to
explore. They show how GNNs, GATs, and hypergraph networks handle spatio-temporal
event data in adjacent tasks (gesture/action recognition).

- [Space-Time Event Clouds for Gesture Recognition: From RGB Cameras to Event Cameras](https://ieeexplore.ieee.org/document/8659288)  
- [Graph-Based Spatio-Temporal Feature Learning for Neuromorphic Vision Sensing](https://ieeexplore.ieee.org/document/9199543) 
- [Event-Stream Representation for Human Gaits Identification Using Deep Neural Networks](https://ieeexplore.ieee.org/document/9337225) 
- [Hypergraph-based Multi-View Action Recognition using Event Cameras](https://arxiv.org/abs/2403.19316) 

---

### 1.4 Complementary methods 

Alternative or supporting approaches relevant to spatiotemporal feature enhancement.

- [Intelligent event-based lip reading word classification with spiking neural networks using spatio-temporal attention features and triplet loss](https://www.sciencedirect.com/science/article/pii/S0020025524005735) 
- [Spectrum-guided Spatial Feature Enhancement Network for event-based lip-reading](https://www.sciencedirect.com/science/article/pii/S0925231225006460) 



## 2. Datasets


**DVS-Lip** Public - [project page](https://sites.google.com/view/event-based-lipreading) |

**DVS-LRW100**  Not public 

For this project we focus exclusively on **DVS-Lip**, as DVS-LRW100 is not publicly available.

**DVS-LRW** - [RGB Dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)


### 3. Selection of Methods for Further Work

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

## 6th May


### Plan for the Next Milestone
- Visualize data from DVS-Lip dataset (DVS-LRW 100 dataset is not publicly available so we can't compare them) - Jakub Sulowski
- Search for more articles about DVS lip reading and add it to literature - Szczepan Tokarczyk
- Add more approaches of work methods (currently it is too general) - Miłosz Senator
- Select method that we will use in our project and describe it shortly - Jakub Sulowski
- Add goal of the project, short description and what we want to achieve at the begin of readme file - Szczepan Tokarczyk and Miłosz Senator


---

## 13th May

