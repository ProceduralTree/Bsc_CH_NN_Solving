testinputs=[[0.001, 100, 0.001, 0.00084205016]]
from multi_solver import CH_2D_Multigrid_Solver
import testUtils as tu
from plot_data import plot
import os
testinputs = testinputs[0]
testphase = tu.k_spheres_phase(10, 4)
testsolver = CH_2D_Multigrid_Solver(
    tu.wprime, testphase, testinputs[2], testinputs[3], testinputs[4]
)
testsolver.solve(10, 10)
