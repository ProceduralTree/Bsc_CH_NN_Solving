#!/usr/bin/env python3

import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
from numpy.random import Generator, PCG64
from scipy.spatial import Voronoi
import random

SIZE = 32


def wprime(x: float):
    return -x * (1 - x**2)


def random_phase() -> np.ndarray:
    gen = Generator(PCG64(42))
    return gen.choice([-1, 1], size=(SIZE, SIZE))


def square_phase() -> np.ndarray:
    ret = np.ones((SIZE, SIZE))
    ret = -1 * ret
    ones = np.ones((10, 10))
    ret[10:-10, 10:-10] = ones
    return ret


def setup_solver(test_phase) -> CH_2D_Multigrid_Solver:
    solver = CH_2D_Multigrid_Solver(np.vectorize(wprime), test_phase, 1e-4, 1e-4, 1e-4)
    return solver


def k_squares_phase(k: int, diameter: int):
    random.seed(42)
    points = [
        [random.randrange(SIZE - diameter), random.randrange(SIZE - diameter)]
        for i in range(k)
    ]
    mat = -1 * np.ones((SIZE, SIZE))
    for p in points:
        mat[p[0] : p[0] + diameter, p[1] : p[1] + diameter] = np.ones(
            (diameter, diameter)
        )
    return mat
