#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_large_dataset(path: str, dir: str, savedir: str):
    dataset = os.listdir(f"{path}/{dir}")
    differences = []
    for datadir in dataset:
        data = os.listdir(f"{path}/{dir}/{datadir}")
        for d in filter(lambda x: ".npz" in x, data):
            array = np.load(f"{path}/{dir}/{datadir}/{d}")["phase"]
            print(array.shape)
            differences += [np.sum(array[0]) - np.sum(array[-1])]
    print(differences)
    # sns.histplot(differences)
    return differences


def plot(path: str, dir: str, savedir: str) -> None:
    dataset = os.listdir(f"{path}/{dir}/")
    dataset = [x for x in filter(lambda x: ".npz" in x, dataset)]
    dataset = [d.replace(".npz", "") for d in dataset]
    print(dataset)
    for d in dataset:
        data = np.load(f"{path}/{dir}/{d}.npz")
        phase_data = data["phase"]
        imgpath = f"{path}/{savedir}/"
        print(f"Shape of data: {phase_data.shape}")
        if not os.path.exists(imgpath):
            os.mkdir(imgpath)
        print(d)
        plt.figure()
        balance = []
        for i in range(phase_data.shape[0]):
            balance += [np.sum(phase_data[i])]
        balance_scaled = np.array(balance) * 2**-12  # 'number of gridcells in test
        ax = plt.gca()
        # ax.set_ylim([-1, 1])
        plt.plot(balance_scaled)
        plt.savefig(imgpath + f"balance_{d}.png")
        plt.close()


def main() -> None:
    path = "/home/proceduraltree/Projects/Bsc_CH_NN_Solving"
    plot(path, "data/new_boundry", "images")


if __name__ == "__main__":
    plot_large_dataset(
        "/home/proceduraltree/Projects/Bsc_CH_NN_Solving", "data", "images"
    )
    pass
