testinputs=[[0.001, 60, 0.001, 0.0005052301], [0.01, 300, 0.01, 0.025261505], [0.1, 3, 0.1, 0.0025261505]]
import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
import testUtils as tu
import os
path = "/home/jon/Projects/Bsc_CH_NN_Solving/testing"
testinputs = testinputs[1]
testphase = tu.k_spheres_phase(10, 4)
testsolver = CH_2D_Multigrid_Solver(
    tu.wprime, testphase, testinputs[0], testinputs[2], testinputs[3]
)
phases = []
mus = []
for i in range(40):
    testsolver.solve(1, 10)
    phases += [testsolver.phase_small]
    mus += [testsolver.mu_small]

np.savez(f"{path}/data/testdata", phase=phases , mu=mus )

import os
from plot_data import plot
path = "/home/jon/Projects/Bsc_CH_NN_Solving/testing"

plot(path, "data", "images")
