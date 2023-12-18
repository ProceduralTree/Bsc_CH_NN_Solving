#!bin/python

import itertools
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from multi_solver import CH_2D_Multigrid_Solver
from multi_solver import SMOOTH_jit
import os
from multiprocessing import Lock, Process
import testUtils as tu
import json
import typing


class Experiment(BaseModel):
    dt: float
    h: float
    epsilon: float
    name: str
    path: str
    iterations: int


def generate_train_data(data, lock: typing.Any) -> None:
    for exp, phase in data:
        solver = CH_2D_Multigrid_Solver(
            np.vectorize(tu.wprime), phase, exp.dt, exp.h, exp.epsilon
        )
        phases = []
        mus = []

        for i in range(exp.iterations):
            solver.solve(1, 10)
            phases += [solver.phase_small]
            mus += [solver.mu_small]
            lock.acquire()
            try:
                with open(f"data/{exp.path}/{exp.name}.json", "w+") as f:
                    f.write(exp.json())
                    np.savez(f"data/{exp.path}/{exp.name}", phase=phases, mu=mus)
            finally:
                lock.release()


def main() -> None:
    lock = Lock()

    experiments = []
    sizes = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    m = 3
    for dt in sizes:
        for h in sizes:
            for diameter in range(5, 25, 5):
                for k in range(1, 10, 2):
                    for phase, pname in [
                        (tu.k_spheres_phase(k, diameter), "sphere"),
                        (tu.k_squares_phase(k, diameter), "square"),
                    ]:
                        experiments += [
                            (
                                Experiment(
                                    dt=1,
                                    h=1,
                                    epsilon=(m * h) / (2 * np.sqrt(2) * np.arctan(0.9)),
                                    path=f"experiment_{dt:1.0e}_{h:1.0e}",
                                    name=f"{k}_{diameter}_{pname}",
                                    iterations=100,
                                ),
                                phase,
                            )
                        ]
    print(len(experiments))
    for exp, _ in experiments:
        if not os.path.exists(f"data/{exp.path}"):
            os.mkdir(f"data/{exp.path}")

    for data in [
        experiments[i : i + int(1000 / 15)]
        for i in range(0, len(experiments), int(1000 / 15))
    ]:
        p = Process(target=generate_train_data, args=(data, lock))
        p.start()
    pass


if __name__ == "__main__":
    main()
    pass
