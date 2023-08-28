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

    mu_large: np.ndarray
    mu_small: np.ndarray
    mu_smooth: np.ndarray
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

        def __L(self, phi: float, mu: float) -> np.ndarray[(2, 1)]:
            zeta = (
                -1 * self.__G() * self.mu_small[i + 1, j]
                - self.__G() * self.mu_small[i, j + 1]
                + self.__G() * 4 * mu
                - self.__G() * self.mu_small[i, j - 1]
                + self.__G() * self.mu_small[i - 1, j]
                + phi / self.dt
            )
            psi = 1
            return np.array([zeta, psi])

    def __v_cycle(self) -> None:
        pass


if __name__ == "__main__":
    solver: CH_2D_Multigrid_Solver = CH_2D_Multigrid_Solver()
