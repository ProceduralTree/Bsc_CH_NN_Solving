#!/usr/bin/env python3

import numpy as np


class CH_2D_Multigrid_Solver:
    """
    Cahn Hillard solever based on the paper [[file:JCP.pdf]]

    """

    def __init__(self):
        pass

    def SMOOTH(self) -> np.ndarray[(2, 1)]:
        pass

    def L(self) -> np.ndarray[(2, 1)]:
        pass

    def v_cycle(self) -> None:
        pass


if __name__ == "__main__":

    solver: CH_2D_Multigrid_Solver = CH_2D_Multigrid_Solver()
