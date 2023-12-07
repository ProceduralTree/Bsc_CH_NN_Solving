import numpy as np
from numba import njit

import scipy.optimize as sp

from typing import Tuple
from multi_solver import Interpolate, Restrict, __G_h


@njit
def SMOOTH_relaxed_njit(
    xi: np.ndarray,
    psi: np.ndarray,
    phase_small: np.ndarray,
    mu_small: np.ndarray,
    epsilon: float,
    h: float,
    dt: float,
    len_small: int,
    width_small: int,
    v: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    for k in range(v):
        for i in range(1, len_small + 1):
            for j in range(1, width_small + 1):
                y = np.vectorize(
                    lambda x: x * dt**-1
                    - xi[i, j]
                    - h**-2
                    * (
                        __G_h(i + 0.5, j, len_small, width_small) * mu_small[i + 1, j]
                        + __G_h(i - 0.5, j, len_small, width_small) * mu_small[i - 1, j]
                        + __G_h(i, j + 0.5, len_small, width_small) * mu_small[i, j + 1]
                        + __G_h(i, j - 0.5, len_small, width_small) * mu_small[i, j - 1]
                    )
                    * (
                        __G_h(i + 0.5, j, len_small, width_small)
                        + __G_h(i - 0.5, j, len_small, width_small)
                        + __G_h(i, j + 0.5, len_small, width_small)
                        + __G_h(i, j - 0.5, len_small, width_small)
                    )
                    ** -1
                )
                F = np.vectorize(
                    lambda x: epsilon**2 * x**alpha
                    + 2 * x
                    - epsilon**2 * c[i, j] ** alpha
                    + y(x)
                    + psi[i, j]
                )

                dF = np.vectorize(
                    lambda x: alpha * epsilon**2 * x ** (alpha - 1) + 2 + dt**-1
                )

                root = sp.newton(F, phase_small[i, j], dF)

                phase_small[i, j] = root
                mu_small[i, j] = y(root)

    return (phase_small, mu_small)


class CH_2D_Multigrid_Solver:

    """
    Cahn Hillard solever based on the paper [[file:JCP.pdf]]

    Parameters
    ---
    `phase_small` : np.ndarray
        phase field vallue on the small grid
    `phase_smooth` : np.ndarray
        phase field vallue after smothing, NaN if not yet calculated
    `phase_large` : np.ndarray
        phase field vallue on the large grid
    `dt` : float
        length ot timesteps in seconds
    `epsilon` : float
        internal variable in Cahn Hillard Equation, Material Specific
    `W_prime` : np.vectorized
        double well potential function usually 4th order polinomial for simplicity

    """

    mu_large: np.ndarray
    mu_small: np.ndarray
    # mu_smooth: np.ndarray
    # phase_smooth: np.ndarray
    phase_large: np.ndarray
    phase_small: np.ndarray
    xi: np.ndarray
    psi: np.ndarray
    dt: float
    h: float
    epsilon: float
    LinOp = NamedTuple("LinOp", [("A", np.ndarray), ("b", np.ndarray)])
    len_small: int
    len_large: int
    width_small: int
    width_large: int
    W_prime: np.vectorize

    def __init__(
        self,
        wprime: np.vectorize,
        phase: np.ndarray,
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
        pass

    def __str__(self):
        return f""" Cahn Hillard solver: \n
        --------------------------\n
        Timestepsize:  {self.dt}s\n
        gridsize:   {self.h}m\n
        Domainsize: {self.phase_large.shape} points\n
        Epsilon:    {self.epsilon}\n
        ----------------------\n

        """

    def __G_h(self, i, j):
        """
        small grid version

        Returns
        ---------------
        1 if index i,j is in bounds(without padding) and 0 else
        """
        if 0 < i < self.len_small + 1 and 0 < j < self.width_small + 1:
            return 1
        return 0

    def __G_H(self, i, j):
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
        [self.phase_small, self.mu_small] = SMOOTH_jit(
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
        # init xi , psi
        # rewrite: smooth all before further processing

        # xipsi = np.zeros((self.len_small + 2, self.width_small + 2, 2))

        # xipsi[1:-1, 1:-1] = np.array(
        #    [
        #        [
        #            self.L_h(i, j).A.dot(
        #                np.array([self.phase_small[i, j], self.mu_small[i, j]])
        #                + self.L_h(i, j).b
        #            )
        #            for i in range(1, self.len_small + 1)
        #        ]
        #        for j in range(1, self.width_small + 1)
        #    ]
        # )

        # TODO more (v_1,v_2) smoothing steps
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

    def solve(self, iterations: int, iteration_depth: int):
        for i in range(iterations):
            self.set_xi_and_psi()
            for j in range(iteration_depth):
                self.v_cycle()

    def set_xi_and_psi(self):
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

    def __Restrict(self, array: np.ndarray) -> np.ndarray:
        """
        restricts an array on the small grid to an array in the large grid

        Returns
        ---------------------------
        large grid array + padding
        """
        return Restrict(
            array, self.len_large, self.width_large, self.len_small, self.width_small
        )

    def __Interpolate(self, array: np.ndarray) -> np.ndarray:
        """
        interpolates from large grid to small grid

        Returns
        -----------------------
        interpolated array + padding
        """
        return Interpolate(
            array, self.len_large, self.width_large, self.len_small, self.width_small
        )
