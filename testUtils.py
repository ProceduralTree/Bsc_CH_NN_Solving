#!/usr/bin/env python3


import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
from multi_solver import SMOOTH_jit
from numpy.random import Generator, PCG64
import random
from multiprocessing import Lock, Process

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


def generate_train_data(phasefield: np.ndarray, iterations: int, name: str, lock):
    solver = setup_solver(phasefield)
    phases = []

    for i in range(iterations):
        solver.solve(1, 100)
        phases += [solver.phase_small]
    lock.acquire()
    try:
        np.save(f"data/{name}", phases)
    finally:
        lock.release()


def gen_data():
    lock = Lock()
    dataset = [
        (sphere_phase(10), 100, "sphere", lock),
        (square_phase(), 100, "square", lock),
    ]

    for k in range(10):
        dataset += [(k_squares_phase(k, 10), 100, f"{k}_square", lock)]

    for data in dataset:
        Process(target=generate_train_data, args=data).start()

    pass


def benchmark_solver(phase: np.ndarray, mu: np.ndarray):
    solver = setup_solver(phase)
    solver.mu_small = mu
    solver.phase_small = phase
    solver.set_xi_and_psi()
    SMOOTH_jit(solver.xi, solver.psi, solver.phase_small, solver.mu_small)
    pass
