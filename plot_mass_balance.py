#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def plot(path: str, dir: str, savedir: str) -> None:
    dataset = os.listdir(f"{path}/{dir}/")
    dataset = [d.replace(".npz", "") for d in dataset]
    print(dataset)
    for d in dataset:
        data = np.load(f"{path}/{dir}/{d}.npz")
        phase_data = data["phase"]
        imgpath = f"{path}/{savedir}/{d}/"
        print(f"Shape of data: {phase_data.shape}")
        if not os.path.exists(imgpath):
            os.mkdir(imgpath)
        print(d)
        plt.figure()
        balance = []
        for i in range(phase_data.shape[0]):
            balance += [np.sum(phase_data[i])]
        plt.plot(balance)
        plt.savefig(imgpath + "balance.png")


def main() -> None:
    path = "/home/proceduraltree/Projects/Bsc_CH_NN_Solving"
    plot(path, "data/experiments", "images")


if __name__ == "__main__":
    pass
