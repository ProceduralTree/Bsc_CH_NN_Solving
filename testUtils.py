#!/usr/bin/env python3

import numpy as np
from multi_solver import CH_2D_Multigrid_Solver


def wprime(x: float):
    return -x * (1 - x**2)


def test_phasefield() -> np.ndarray:
    ret = np.random.random_sample((30, 30)) * 2 - 1
    filter = np.vectorize(lambda x: 1 if x > 0 else -1)
    return filter(ret)


def test_phasefield_square() -> np.ndarray:
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
