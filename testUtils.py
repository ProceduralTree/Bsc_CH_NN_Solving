#!/usr/bin/env python3

import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
from numpy.random import Generator, PCG64
from scipy.spatial import Voronoi
import random

SIZE = 32
SEED = 42


def wprime(x: float):
    return -x * (1 - x**2)


def random_phase() -> np.ndarray:
    gen = Generator(PCG64(SEED))
    return gen.choice([-1, 1], size=(SIZE, SIZE))


def square_phase() -> np.ndarray:
    ret = np.ones((SIZE, SIZE))
    ret = -1 * ret
    ones = np.ones((10, 10))
    ret[10:20, 10:20] = ones
    return ret


def setup_solver(test_phase) -> CH_2D_Multigrid_Solver:
    solver = CH_2D_Multigrid_Solver(np.vectorize(wprime), test_phase, 1e-4, 1e-4, 1e-4)
    return solver


def k_squares_phase(k: int, diameter: int):
    random.seed(SEED)
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


def k_spheres_phase(k: int, diameter: int):
    random.seed(SEED)
    points = [
        [random.randrange(SIZE - diameter), random.randrange(SIZE - diameter)]
        for i in range(k)
    ]
    pointv = np.array(points)
    mat = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            mat[i, j] = is_in_sphere(i, j, pointv, diameter)
    return mat


def sphere_phase(diameter):
    points = np.array([[SIZE / 2, SIZE / 2]])
    mat = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            mat[i, j] = is_in_sphere(i, j, points, diameter)
    return mat


def is_in_sphere(i, j, points, diameter):
    v = np.array([[i, j] for k in range(len(points))])
    div = points - v
    dists = [np.linalg.norm(x) for x in div]
    return 1 if min(dists) < diameter else -1


def generate_train_data(phasefield: np.ndarray, iterations: int, name: str):
    solver = setup_solver(phasefield)
    phases = []

    print(f"Iterations: {iterations}")
    for i in range(iterations):
        print(f"Iteration: {i}")
        solver.solve(1, 100)
        phases += [solver.phase_small]

    np.save(f"data/{name}", phases)
    pass
