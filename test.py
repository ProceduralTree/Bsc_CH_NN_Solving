#!/usr/bin/env python3


import numpy as np


class CH_2D_Multigrid_Solver:
    """
    Cahn Hillard solever based on the paper [[file:JCP.pdf]]

    Parameters
    ---
    phase_small : np.ndarray
        phase field vallue on the small grid
    phase_smooth : np.ndarray
        phase field vallue after smothing, NaN if not yet calculated
    phase_large : np.ndarray
        phase field vallue on the large grid
    """

    mu_large: np.nparray
    mu_small: np.nparray
    mu_smooth: np.nparray
    phase_smooth: np.ndarray
    phase_large: np.ndarray
    phase_small: np.ndarray
    zeta: np.ndarray
    psi: np.ndarray
    dt: float
    h: float
    epsilon: float

    def __init__(self):
        pass

    def __G(self, i, j):
        """TODO implement"""
        return 1 if i in range(self.phase_small.shape[0]) else 0

    def __SMOOTH(self) -> np.ndarray[(2, 1)]:
        i = 0
        j = 0
        bordernumber = 4
        gsum = 4

        coefmatrix = np.array(
            [
                [1 / self.dt, bordernumber / self.h**2],
                [-(2 * self.h**2 + self.epsilon**2 * gsum) / self.h**2],
            ]
        )

        b = np.array(
            [
                self.zeta[i, j]
                + (
                    self.__G(i + 0.5, j) * self.mu_small[i + 1, j]
                    + self.__G(i - 0.5, j) * self.mu_small[i - 1, j]
                    + self.__G(i, j + 0.5) * self.mu_small[i, j + 1]
                    + self.__G(i, j - 0.5) * self.mu_small[i, j - 1]
                )
                / self.h**2,
                self.psi[i, j]
                + (self.epsilon**2 / self.h**2)
                * (
                    self.__G(i + 0.5, j) * self.phase_small[i + 1, j]
                    + self.__G(i - 0.5, j) * self.phase_small[i - 1, j]
                    + self.__G(i, j + 0.5) * self.phase_small[i, j + 1]
                    + self.__G(i, j - 0.5) * self.phase_small[i, j - 1]
                ),
            ]
        )

        res = np.linalg.solve(coefmatrix, b)
        return res

    def __L(self) -> np.ndarray[(2, 1)]:
        return np.zeros((2, 1))

    def __v_cycle(self) -> None:
        pass


if __name__ == "__main__":
    for i in range(10):
        print(i)

    solver: CH_2D_Multigrid_Solver = CH_2D_Multigrid_Solver()
