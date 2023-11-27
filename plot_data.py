#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_theme()
path = "/home/proceduraltree/Projects/Bsc_CH_NN_Solving"


def plot():
    dataset = os.listdir(f"{path}/data/")
    dataset = [d.replace(".npy", "") for d in dataset]
    print(dataset)
    for d in dataset:
        data = np.load(f"{path}/data/{d}.npy")
        imgpath = f"{path}/images/{d}/"
        print(f"Shape of data: {data.shape}")
        if not os.path.exists(imgpath):
            os.mkdir(imgpath)
        print(d)
        for i in range(data.shape[0]):
            print(f"Saving image {i}/{data.shape[0]-1}")
            plt.figure()
            sns.heatmap(data[i])
            plt.savefig(f"{path}/images/{d}/{d}_{i:03}.png")
