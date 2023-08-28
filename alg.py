from typing import Callable
import numpy as np
import matplotlib as plt


# def discrete_laplacian(i: int,j : int , discrete_function: np.matrix,  gridsize : np.double):
# second_deriv_x = 1/(gridsize**2) * ((discrete_function[i + 1, j] - discrete_function[i, j]) - (discrete_function[i, j] - discrete_function[i-1, j]))
# second_deriv_y = 1/(gridsize**2) * ((discrete_function[i,j + 1] - discrete_function[i,j]) - (discrete_function[i,j] - discrete_function[i,j-1]))
# laplace = second_deriv_x + second_deriv_y
# return laplace
#
#
# def discrete_laplacian_matrix(discrete_function: np.matrix, gridsize: np.double):
# Delta_f = np.zeros(discrete_function.shape)
# for i in range(2,discrete_function.shape[0]-1):
# for j in range(2,discrete_function.shape[1]-1):
# Delta_f[i,j] = discrete_laplacian(i,j,discrete_function , gridsize)
# return Delta_f
#


class CN_CH_1D_Solver:
    """Crank Nichelson based solver for 1d Cahn Hillard Equation."""

    delta_x: float = 0.1
    """size of domain"""
    grd_size: float = 1
    delta_t: float = 0.1
    W_prime: Callable
    W_pprime: Callable
    c_curent: np.ndarray
    c_previous: np.ndarray
    epsilon: float
    gamma: float

    def __init__(self, dx, dt, grid_size, W_p, W_pp, eps, gamma) -> None:
        self.delta_x = dx
        self.delta_t = dt
        self.grd_size = grid_size
        self.W_pprime = W_pp
        self.W_prime = W_p
        self.c_curent = np.zeros(grid_size / dx)
        self.c_previous = self.c_curent
        self.epsilon = eps
        self.gamma = gamma

    def f(self, i: int, x: np.ndarray) -> float:
        """
        Partial function for newton iteration.

        Function with 0 at x=[c_curent[i], c_curent[i+1], c_current[i+2]]
        provided they  are known.

        Parameters
        -----------------
        *x*: is used to calculate the unknown values of
        c_current[i+2], ... , c_current[i]

        *i* :  index of variable to be solved for

        Returns
        -----------
        0 if x is optimal
        """
        laplace_d_W_current = (1 / self.delta_x**2) * np.array([1, -2, 1]).dot(
            np.array(
                [
                    self.W_prime(self.c_curent[i - 1]),
                    self.W_prime(x[0]),
                    self.W_prime(x[1]),
                ]
            )
        )

        laplace_d_W_prev = (1 / self.delta_x**2) * np.array([1, -2, 1]).dot(
            np.array(
                [
                    self.W_prime(self.c_previous[i - 1]),
                    self.W_prime(self.c_previous[i]),
                    self.W_prime(self.c_previous[i + 1]),
                ]
            )
        )

        laplaplace_d_c_previous = np.array([1, -4, 6, -4, 1]).dot(
            np.array(
                [
                    self.c_previous[i - 2],
                    self.c_previous[i - 1],
                    self.c_previous[i],
                    self.c_previous[i + 1],
                    self.c_previous[i + 2],
                ]
            )
        )

        laplaplace_d_c_current = np.array([1, -4, 6, -4, 1]).dot(
            np.array([self.c_curent[i - 2], self.c_curent[i - 1], x[0], x[1], x[2]])
        )

        return (
            (self.c_curent[i] - self.c_previous[i]) / self.delta_t
            + 0.5 * (laplace_d_W_current + laplace_d_W_prev)
            + self.gamma
            * self.epsilon
            * 0.5
            * (laplaplace_d_c_current + laplaplace_d_c_previous)
        )

    pass

    def F(self, i):
        self.f
        pass


if __name__ == "__main__":
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 0],
            [1, 2, 3, 4, 0, 6],
            [1, 2, 3, 0, 5, 6],
            [1, 2, 0, 4, 5, 6],
            [1, 0, 3, 4, 5, 6],
            [0, 2, 3, 4, 5, 6],
        ]
    )

    a = np.array([1, 2, 3])
    print(np.linalg.inv(matrix))
    np.round(matrix / matrix, 10)
    ch_solv: CN_CH_1D_Solver = CN_CH_1D_Solver(
        0.1, 0.1, 1, lambda x: x, lambda x: 1, 0.5, 0.5
    )
