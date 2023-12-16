#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_theme()


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
        for i in range(phase_data.shape[0]):
            print(f"Saving image {i+1}/{phase_data.shape[0]}")
            plt.figure()
            sns.heatmap(phase_data[i])
            plt.savefig(f"{path}/{savedir}/{d}/{d}_{i:03}.png")
            plt.close("all")
        print("Generating GIF \n")
        os.system(
            f"convert -layers OptimizePlus -delay 1x24 -quality 99 {path}/{savedir}/{d}/*.png -loop 0 {path}/{savedir}/{d}.gif"
        )


def main() -> None:
    path = "/home/proceduraltree/Projects/Bsc_CH_NN_Solving"
    plot(path, "data/new_boundry", "images")


if __name__ == "__main__":
    main()
