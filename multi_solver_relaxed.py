import seaborn as sns
import matplotlib.pyplot as plt

import math
import numpy as np
from numba import njit
import testUtils as tu

import scipy.optimize as sp
from numpy.typing import NDArray

import numba as nb

from typing import Tuple, NamedTuple
from multi_solver import Interpolate, Restrict, __G_h


@njit
def alternative_el_solver(
    c: NDArray[np.float64],
    phase: NDArray[np.float64],
    len: int,
    width: int,
    alpha: float,
    h: float,
    n: int,
) -> NDArray[np.float64]:
    """
    solves elyptical equation for \(x^\alpha\)
    """
    maxiter = 10000
    tol = 1.48e-8
    for k in range(n):
        for i in range(1, len + 1):
            for j in range(1, width + 1):
                bordernumber = (
                    __G_h(i + 0.5, j, len, width)
                    + __G_h(i - 0.5, j, len, width)
                    + __G_h(i, j + 0.5, len, width)
                    + __G_h(i, j - 0.5, len, width)
                )
                x = c[i, j] + 0.01
                for iter in range(maxiter):
                    if iter == maxiter - 2:
                        print("Iter:")
                        print(iter)
                        print("c:")
                        print(x)

                        raise Warning("solver might not converge")
                    F = (
                        -1
                        * h**-2
                        * (
                            __G_h(i + 0.5, j, len, width) * c[i + 1, j]
                            + __G_h(i - 0.5, j, len, width) * c[i - 1, j]
                            + __G_h(i, j + 0.5, len, width) * c[i, j + 1]
                            + __G_h(i, j - 0.5, len, width) * c[i, j - 1]
                        )
                        + h**-2 * bordernumber * x
                        + alpha * x
                        - alpha * phase[i, j] ** alpha
                    )

                    dF = -1 * alpha + h**-2 * bordernumber

                    if dF == 0:
                        raise RuntimeError(
                            "ERROR newton iteration in elyps solver did not converge, dF was 0"
                        )

                    step = F / dF
                    x = x - step
                    if abs(step) < tol:
                        break
                    if abs(step) > 1e16:
                        print("Step:")
                        print(step)
                        print("Iter:")
                        print(iter)
                        raise RuntimeError("WTF why the solver so large")

                c[i, j] = x
    return c


@njit
def elyps_solver(
    c: NDArray[np.float64],
    phase: NDArray[np.float64],
    len: int,
    width: int,
    alpha: float,
    h: float,
    n: int,
) -> NDArray[np.float64]:
    maxiter = 100
    tol = 1.48e-8
    for k in range(n):
        for i in range(1, len + 1):
            for j in range(1, width + 1):
                bordernumber = (
                    __G_h(i + 0.5, j, len, width)
                    + __G_h(i - 0.5, j, len, width)
                    + __G_h(i, j + 0.5, len, width)
                    + __G_h(i, j - 0.5, len, width)
                )
                x = c[i, j] + 0.01
                for iter in range(maxiter):
                    # if iter == maxiter - 2:
                    # print(iter)
                    # raise Warning("solver might not converge")
                    F = (
                        -1
                        * h**-2
                        * (
                            __G_h(i + 0.5, j, len, width) * c[i + 1, j] ** alpha
                            + __G_h(i - 0.5, j, len, width) * c[i - 1, j] ** alpha
                            + __G_h(i, j + 0.5, len, width) * c[i, j + 1] ** alpha
                            + __G_h(i, j - 0.5, len, width) * c[i, j - 1] ** alpha
                        )
                        + h**-2 * bordernumber * x**alpha
                        + alpha * x**alpha
                        - alpha * phase[i, j] ** alpha
                    )

                    F_h = -1 * (
                        h**-2
                        * (
                            __G_h(i + 0.5, j, len, width) * c[i + 1, j] ** alpha
                            + __G_h(i - 0.5, j, len, width) * c[i - 1, j] ** alpha
                            + __G_h(i, j + 0.5, len, width) * c[i, j + 1] ** alpha
                            + __G_h(i, j - 0.5, len, width) * c[i, j - 1] ** alpha
                        )
                        + h**-2 * bordernumber * (x + 1e-2) ** alpha
                        + alpha * (x + 1e-2) ** alpha
                        - alpha * phase[i, j] ** alpha
                    )
                    if True:
                        dF = -1 * alpha**2 * x ** (
                            alpha - 1
                        ) + h**-2 * bordernumber * alpha**2 * x ** (alpha - 1)
                    else:
                        dF = 1e2 * (F_h - F)

                    if dF == 0:
                        raise RuntimeError(
                            "ERROR newton iteration in elyps solver did not converge, dF was 0"
                        )

                    step = F / dF
                    x = x - step
                    if abs(step) < tol:
                        break
                    if abs(step) > 1e16:
                        print("Step:")
                        print(step)
                        print("Iter:")
                        print(iter)
                        raise RuntimeError("WTF why the solver so large")

                c[i, j] = x
    return c


@njit
def SMOOTH_relaxed_njit(
    c: NDArray[np.float64],
    xi: NDArray[np.float64],
    psi: NDArray[np.float64],
    phase_small: NDArray[np.float64],
    mu_small: NDArray[np.float64],
    epsilon: float,
    h: float,
    dt: float,
    len_small: int,
    width_small: int,
    v: int,
    alpha: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    for k in range(v):
        for i in range(1, len_small + 1):
            for j in range(1, width_small + 1):
                gsum = (
                    __G_h(i + 0.5, j, len_small, width_small)
                    + __G_h(i - 0.5, j, len_small, width_small)
                    + __G_h(i, j + 0.5, len_small, width_small)
                    + __G_h(i, j - 0.5, len_small, width_small)
                )
                bsum = (
                    __G_h(i + 0.5, j, len_small, width_small) * mu_small[i + 1, j]
                    + __G_h(i - 0.5, j, len_small, width_small) * mu_small[i - 1, j]
                    + __G_h(i, j + 0.5, len_small, width_small) * mu_small[i, j + 1]
                    + __G_h(i, j - 0.5, len_small, width_small) * mu_small[i, j - 1]
                )
                # TODO missing xi and psi in formula
                phase = (epsilon**2 * c[i, j] - h**-2 * bsum * 1 / gsum) / (
                    dt**-1 + epsilon**2 + 2
                )
                y = phase / dt - h**-2 * bsum / gsum - xi[i, j]

                phase_small[i, j] = phase
                mu_small[i, j] = y

    return (phase_small, mu_small)


class CH_2D_Multigrid_Solver_relaxed:

    """
    Cahn Hillard solever based on the paper [[file:JCP.pdf]]

    Parameters
    ---
    `phase_small` : NDArray[np.float64]
        phase field vallue on the small grid
    `phase_smooth` : NDArray[np.float64]
        phase field vallue after smothing, NaN if not yet calculated
    `phase_large` : NDArray[np.float64]
        phase field vallue on the large grid
    `dt` : float
        length ot timesteps in seconds
    `epsilon` : float
        internal variable in Cahn Hillard Equation, Material Specific
    `W_prime` : np.vectorized
        double well potential function usually 4th order polinomial for simplicity

    """

    mu_large: NDArray[np.float64]
    mu_small: NDArray[np.float64]
    # mu_smooth: NDArray[np.float64]
    # phase_smooth: NDArray[np.float64]
    phase_large: NDArray[np.float64]
    phase_small: NDArray[np.float64]
    xi: NDArray[np.float64]
    psi: NDArray[np.float64]
    dt: float
    h: float
    epsilon: float
    LinOp = NamedTuple(
        "LinOp", [("A", NDArray[np.float64]), ("b", NDArray[np.float64])]
    )
    len_small: int
    len_large: int
    width_small: int
    width_large: int
    W_prime: np.vectorize
    alpha: float
    c: NDArray[np.float64]

    def __init__(
        self,
        wprime: np.vectorize,
        phase: NDArray[np.float64],
        dt: float,
        h: float,
        epsilon: float,
    ):
        self.dt = dt
        self.h = h
        self.epsilon = epsilon
        self.W_prime = wprime
        self.len_large = phase.shape[0]
        self.width_large = phase.shape[1]
        self.len_small = phase.shape[0] * 2
        self.width_small = phase.shape[1] * 2
        self.phase_large = np.zeros((self.len_large + 2, self.width_large + 2))
        self.phase_large[1:-1, 1:-1] = phase
        self.phase_small = np.zeros((self.len_small + 2, self.width_small + 2))
        self.phase_small = self.__Interpolate(self.phase_large)
        self.mu_large = np.zeros(self.phase_large.shape)
        self.mu_small = np.zeros(self.phase_small.shape)
        self.mu_large[1:-1, 1:-1] = wprime(self.phase_large[1:-1, 1:-1])
        self.mu_small[1:-1, 1:-1] = wprime(self.phase_small[1:-1, 1:-1])
        self.xi = np.zeros(self.phase_small.shape)
        self.psi = np.zeros(self.phase_small.shape)
        self.alpha = 11
        self.c = np.zeros(self.phase_small.shape)
        pass

    def __str__(self) -> str:
        return f""" Cahn Hillard solver: \n
        --------------------------\n
        Timestepsize:  {self.dt}s\n
        gridsize:   {self.h}m\n
        Domainsize: {self.phase_large.shape} points\n
        Epsilon:    {self.epsilon}\n
        ----------------------\n

        """

    def __G_h(self, i, j) -> int:
        """
        small grid version

        Returns
        ---------------
        1 if index i,j is in bounds(without padding) and 0 else
        """
        if 0 < i < self.len_small + 1 and 0 < j < self.width_small + 1:
            return 1
        return 0

    def __G_H(self, i, j) -> int:
        """
        large grid version

        Returns
        ---------------
        1 if index i,j is in bounds(without padding) and 0 else
        """
        if 0 < i < self.len_large + 1 and 0 < j < self.width_large + 1:
            return 1
        return 0

    def SMOOTH(
        self,
        v: int,
    ) -> None:
        [self.phase_small, self.mu_small] = SMOOTH_relaxed_njit(
            self.c,
            self.xi,
            self.psi,
            self.phase_small,
            self.mu_small,
            self.epsilon,
            self.h,
            self.dt,
            self.len_small,
            self.width_small,
            v,
            self.alpha,
        )
        pass

    def L_h(self, i: int, j: int) -> LinOp:
        """
        central operator for the CH iterations for small grid:

        Parameters
        ------------
        `i`: row index of the point the operator shall be instatiated at
        `j`: column index of the point the operator shall be instatiated at

        Returns
        ---------------
        Linear Operator L, in form Matrix A, Vector B
        """
        g = np.array(
            [
                self.__G_h(i + 0.5, j),
                self.__G_h(i - 0.5, j),
                self.__G_h(i, j + 0.5),
                self.__G_h(i, j - 0.5),
            ]
        )
        zeta = np.array(
            [
                -1 * self.mu_small[i + 1, j],
                self.mu_small[i - 1, j],
                -1 * self.mu_small[i, j + 1],
                -1 * self.mu_small[i, j - 1],
            ]
        ).dot(g)
        psi = self.epsilon**2 * np.array(
            [
                self.phase_small[i + 1, j],
                self.phase_small[i - 1, j],
                self.phase_small[i, j + 1],
                self.phase_small[i, j - 1],
            ]
        ).dot(g)

        b = np.array([zeta, psi])
        gsum = g.dot(np.ones(4))

        coeffmatrix = np.array(
            [
                [1 / self.dt, gsum / self.h**2],
                [-1 * (self.epsilon**2 / self.h**2 * gsum + 2), 1],
            ]
        )
        return self.LinOp(coeffmatrix, b)

    def __L_H(self, i: int, j: int) -> LinOp:
        """
        central operator for the CH iterations for large grid:
        Operator for large grid

        Parameters
        ------------
        `i`: row index of the point the operator shall be instatiated at
        `j`: column index of the point the operator shall be instatiated at

        Returns
        ---------------
        Linear Operator L, in form Matrix A, Vector B
        """
        g = np.array(
            [
                self.__G_H(i + 0.5, j),
                self.__G_H(i - 0.5, j),
                self.__G_H(i, j + 0.5),
                self.__G_H(i, j - 0.5),
            ]
        )
        zeta = np.array(
            [
                -1 * self.mu_large[i + 1, j],
                self.mu_large[i - 1, j],
                -1 * self.mu_large[i, j + 1],
                -1 * self.mu_large[i, j - 1],
            ]
        ).dot(g)
        psi = self.epsilon**2 * np.array(
            [
                self.phase_large[i + 1, j],
                self.phase_large[i - 1, j],
                self.phase_large[i, j + 1],
                self.phase_large[i, j - 1],
            ]
        ).dot(g)

        b = np.array([zeta, psi])

        gsum = g.dot(np.ones(4))
        coeffmatrix = np.array(
            [
                [1 / self.dt, gsum / self.h**2],
                [-1 * (self.epsilon**2 / self.h**2 * gsum + 2), 1],
            ]
        )

        return self.LinOp(coeffmatrix, b)

    def v_cycle(self) -> None:
        # TODO more (v_1,v_2) smoothing steps
        self.solve_elyps(40)
        self.SMOOTH(40)

        # extract (d,r) as array operations
        #

        dr = np.zeros((self.len_small + 2, self.width_small + 2, 2))

        dr[1:-1, 1:-1, :] = np.array(
            [
                [
                    np.array([self.xi[i, j], self.psi[i, j]])
                    - self.L_h(i, j).A.dot(
                        np.array([self.phase_small[i, j], self.mu_small[i, j]])
                    )
                    - self.L_h(i, j).b
                    for j in range(1, self.width_small + 1)
                ]
                for i in range(1, self.len_small + 1)
            ]
        )
        d = dr[:, :, 0]
        r = dr[:, :, 1]

        # print(f"Max derivation d: {np.linalg.norm(d)}")
        # print(f"Max derivation r: {np.linalg.norm(r)}")
        d_H = self.__Restrict(d)
        r_H = self.__Restrict(r)
        self.phase_large = self.__Restrict(self.phase_small)
        self.mu_large = self.__Restrict(self.mu_small)

        u_large = np.zeros((self.len_large + 2, self.width_large + 2))
        v_large = np.zeros((self.len_large + 2, self.width_large + 2))

        # solve for phi^ mu^ with L
        for i in range(1, self.len_large + 1):
            for j in range(1, self.width_large + 1):
                [A, c] = self.__L_H(i, j)
                approx = np.linalg.solve(
                    A,
                    A.dot(np.array([self.phase_large[i, j], self.mu_large[i, j]]))
                    + np.array([d_H[i, j], r_H[i, j]]),
                )
                [u, v] = approx - np.array(
                    [self.phase_large[i, j], self.mu_large[i, j]]
                )
                u_large[i, j] = u
                v_large[i, j] = v

        # print(f"Max derivation u: {np.linalg.norm(u_large)}")
        # print(f"Max derivation v: {np.linalg.norm(v_large)}")

        u_small = self.__Interpolate(u_large)
        v_small = self.__Interpolate(v_large)

        self.phase_small = self.phase_small + u_small
        self.mu_small = self.mu_small + v_small
        # smooth again:
        self.SMOOTH(80)
        # print("\n")

        pass

    def solve(self, iterations: int, iteration_depth: int) -> None:
        for i in range(iterations):
            self.set_xi_and_psi()
            for j in range(iteration_depth):
                self.v_cycle()

    def set_xi_and_psi(self) -> None:
        self.xi[1:-1, 1:-1] = np.array(
            [
                [
                    self.phase_small[i, j] / self.dt
                    for j in range(1, self.width_small + 1)
                ]
                for i in range(1, self.len_small + 1)
            ]
        )

        self.psi[1:-1, 1:-1] = np.array(
            [
                [
                    self.W_prime(self.phase_small[i, j]) - 2 * self.phase_small[i, j]
                    for j in range(1, self.width_small + 1)
                ]
                for i in range(1, self.len_small + 1)
            ]
        )

    def __Restrict(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        restricts an array on the small grid to an array in the large grid

        Returns
        ---------------------------
        large grid array + padding
        """
        return Restrict(
            array, self.len_large, self.width_large, self.len_small, self.width_small
        )

    def __Interpolate(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        interpolates from large grid to small grid

        Returns
        -----------------------
        interpolated array + padding
        """
        return Interpolate(
            array, self.len_large, self.width_large, self.len_small, self.width_small
        )

    def solve_elyps(self, n: int) -> None:
        self.c = alternative_el_solver(
            self.c,
            self.phase_small,
            self.len_small,
            self.width_small,
            self.alpha,
            self.h,
            n,
        )


def test_solver() -> CH_2D_Multigrid_Solver_relaxed:
    solver = CH_2D_Multigrid_Solver_relaxed(
        tu.wprime, tu.k_spheres_phase(5, 5), 1e-3, 1e-3, 1e-3
    )
    solver.c = solver.phase_small**solver.alpha
    return solver


def plot(arr: NDArray[np.float64]) -> None:
    sns.heatmap(arr)
    plt.show()
