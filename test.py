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
testsolver.v_cycle(10)
old_v = testsolver.phase_small.copy()
testsolver.v_cycle(10)
sns.heatmap(np.norm(testsolver.phase_small - old_v))
