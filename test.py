#!/usr/bin/env python3


import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import testUtils as tu
from multi_solver import CH_2D_Multigrid_Solver

sns.set_theme()


def test_smooth(n):
    time = timeit.default_timer()
    solver = tu.setup_solver()
    solver.SMOOTH(n)
    delta = timeit.default_timer() - time
    print(delta)


def plot_solver(solver: CH_2D_Multigrid_Solver):
    sns.heatmap(solver.phase_small)
    plt.show()
    pass


def plot_anim_square(n: int):
    solver = tu.setup_solver(tu.test_phasefield_square())

    for i in range(n):
        solver.solve(1, 100)
        sns.heatmap(solver.phase_small, cbar=False)
        plt.savefig(f"images/ch_solver_{i:03}.png")
        print(f"Finished Iteration: {i}")
        print("---------------------------------")
    plt.show()


def plot_anim(n: int):
    test_phase = tu.test_phasefield_square()
    solver = tu.setup_solver(test_phase)
    # fig = plt.figure()
    # animate = FuncAnimation(
    #    fig,
    #    lambda i: anim(i, solver),
    #    frames=10,
    # )
    # plt.show()
    for i in range(40):
        solver.solve(1, 10)
        sns.heatmap(solver.phase_small, cbar=False)
        plt.savefig(f"images/ch_solver_{i:03}.png")
    plt.show()


def main():
    plot_anim(40)


if __name__ == "__main__":
    main()
