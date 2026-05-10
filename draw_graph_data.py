import os, glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import     GraphGen, \
                        draw_graph

root = os.path.join(os.getcwd(), "data","difference","2.npy")

data = np.load(root)
time = data["t"]
x = data["x"]
y = data["y"]
p = data['p']
tau = 10000

timestamp_temp = []
x_temp = []
y_temp = []
polarity_temp = []
data_first = time[0]


for i in range(len(time)):
    if time[0] == time[-1]:
        break
    elif time[i] - data_first < tau:
        timestamp_temp.append(time[i])
        x_temp.append(x[i])
        y_temp.append(y[i])
        polarity_temp.append(p[i])
    else:
        data_first = time[i]
        t_min = min(timestamp_temp)
        t_max = max(timestamp_temp)
        time_norm = [int(np.floor((t - t_min) / (t_max - t_min) * 31)) for t in timestamp_temp]
        data = list(zip(x_temp,y_temp,time_norm,polarity_temp))
        graphgen = GraphGen(r=5, dimension_XY=128, self_loop=True)
        x_, pos, edges = graphgen.forward(data)
        print(f"timestamp: {t_min}, {t_max}")
        print("avg number of edges per node:", len(edges) / len(pos))
        draw_graph(pos,edges,4,10,180,0)
        timestamp_temp = []
        x_temp = []
        y_temp = []
        polarity_temp = []

