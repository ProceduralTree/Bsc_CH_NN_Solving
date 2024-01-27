testinputs=[[0.001, 150, 0.001, 0.0012630752], [0.01, 30, 0.01, 0.0025261505], [0.1, 3, 0.1, 0.0025261505], [0.001, 30, 0.01, 0.0025261505]]
import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
import testUtils as tu
import seaborn as sns
import matplotlib.pyplot as plt

testphase = tu.k_spheres_phase(10, 5, size=16)
testinputs = testinputs[0]
testsolver = CH_2D_Multigrid_Solver(
    tu.wprime, testphase, testinputs[0], testinputs[2], testinputs[3]
)
testsolver.set_xi_and_psi()
testsolver.SMOOTH(10)
sns.heatmap(testsolver.phase_small)

testinputs=[[0.001, 150, 0.001, 0.0012630752], [0.01, 30, 0.01, 0.0025261505], [0.1, 3, 0.1, 0.0025261505], [0.001, 30, 0.01, 0.0025261505]]
import git
import numpy as np
from multi_solver import CH_2D_Multigrid_Solver
import testUtils as tu
import seaborn as sns
import matplotlib.pyplot as plt

testphase = tu.k_spheres_phase(10, 5, size=16)
testinputs = testinputs[0]
testsolver = CH_2D_Multigrid_Solver(
    tu.wprime, testphase, testinputs[0], testinputs[2], testinputs[3]
)
repo = git.Repo('.', search_parent_directories=True)
path = repo.working_tree_dir
print(path)
phases = []
mus = []
for i in range(4):
    testsolver.solve(1, 10)
    phases += [testsolver.phase_small]
    mus += [testsolver.mu_small]

np.savez(f"{path}/testing/data/testdata", phase=phases , mu=mus )

import os
from plot_data import plot
import git

repo = git.Repo('.', search_parent_directories=True)
path = repo.working_tree_dir
plot(f"{path}/testing", "data", "images")
