from gmorbits.methods import (
    ExplicitEulerMethod,
    SymplecticEulerMethod,
)
from gmorbits.potentials import KeplerPotential, HomogeneousSpherePotential
from gmorbits.constants import G
from gmorbits.plot import plot_result
import numpy as np


if __name__ == "__main__":
    M = 10

    pot = KeplerPotential(M)  # HomogeneousSpherePotential(M, 15)  #

    import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # pot.plot_gradient_3d(ax, -12, 12, -12, 12, -12, 12)
    # plt.show()
    # plt.close()

    m = ExplicitEulerMethod(pot, 0.2)

    x0 = [10, 0, 0]
    r = np.linalg.norm(x0)
    vc = np.sqrt(G * M / r)
    T = 2 * np.pi * r / vc
    result = m.iterate(x0, [0, vc, 0], T * 3, saveto="out.csv")

    plot_result(result)
