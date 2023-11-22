#!/usr/bin/env python3

import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
from numpy.random import Generator, PCG64
from scipy.spatial import Voronoi
import random


def wprime(x: float):
    return -x * (1 - x**2)


def test_phasefield() -> np.ndarray:
    ret = np.random.random_sample((30, 30)) * 2 - 1
    filter = np.vectorize(lambda x: 1 if x > 0 else -1)
    return filter(ret)


def square_phase() -> np.ndarray:
    ret = np.ones((30, 30))
    ret = -1 * ret
    ones = np.ones((10, 10))
    ret[10:-10, 10:-10] = ones
    return ret


def test_field_2() -> np.ndarray:
    ret = np.zeros((128, 64))
    filter = np.vectorize(lambda x: 1)
    return filter(ret)


def setup_solver(test_phase) -> CH_2D_Multigrid_Solver:
    solver = CH_2D_Multigrid_Solver(np.vectorize(wprime), test_phase, 1e-4, 1e-4, 1e-4)
    return solver


def k_squares_phase(k: int, diameter: int):
    random.seed(42)
    SIZE = 32
    points = [
        [random.randrange(SIZE - diameter), random.randrange(SIZE - diameter)]
        for i in range(k)
    ]
    print(points)
    mat = -1 * np.ones((32, 32))
    for p in points:
        mat[p[0] : p[0] + diameter, p[1] : p[1] + diameter] = np.ones(
            (diameter, diameter)
        )
    return mat
