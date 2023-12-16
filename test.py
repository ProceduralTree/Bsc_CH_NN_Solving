#!/usr/bin/env python3


import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import testUtils as tu
import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
from multi_solver import SMOOTH_jit
import os


from multiprocessing import Lock, Process


sns.set_theme()


def test_smooth(n):
    time = timeit.default_timer()
    testphase = tu.square_phase()
    solver = tu.setup_solver(testphase)
    solver.SMOOTH(n)
    delta = timeit.default_timer() - time
    print(delta)


def main() -> None:
    data = np.load("data/4_square.npz")
    phases = data["phase"]
    mus = data["mu"]
    benchmark_SMOOTH(phases[9], mus[9])
    pass


def gen_data(experiment_name: str) -> None:
    lock = Lock()
    if not os.path.exists(f"data/{experiment_name}"):
        os.mkdir(f"data/{experiment_name}")
    dataset = [
        (tu.sphere_phase(10), 100, f"{experiment_name}/sphere", lock),
        (tu.square_phase(), 100, f"{experiment_name}/square", lock),
    ]

    for k in range(10):
        dataset += [
            (tu.k_squares_phase(k, 10), 100, f"{experiment_name}/{k}_square", lock)
        ]

    for data in dataset:
        Process(target=tu.generate_train_data, args=data).start()

    pass


def benchmark_SMOOTH(phase: np.ndarray, mu: np.ndarray) -> None:
    solver = tu.setup_solver(
        np.zeros((int(phase.shape[0] / 2 - 2), int(phase.shape[1] / 2 - 2)))
    )
    solver.mu_small = mu
    solver.phase_small = phase
    smooth_progress = []
    solver.set_xi_and_psi()
    for i in range(10):
        [phase, mu] = SMOOTH_jit(
            solver.xi,
            solver.psi,
            solver.phase_small,
            solver.mu_small,
            solver.epsilon,
            solver.h,
            solver.dt,
            solver.len_small - 2,
            solver.width_small - 2,
            1,
        )
        smooth_progress += [phase]
    np.save("data/smooth_snapshot", smooth_progress)
    pass


if __name__ == "__main__":
    main()
