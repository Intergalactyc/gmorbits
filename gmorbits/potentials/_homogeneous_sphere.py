from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class HomogeneousSpherePotential(Potential):
    def __init__(self, M_total: float, R: float):
        self.GM = G * M_total
        self.rho0 = 3 * M_total / (4 * np.pi * R**3)
        self.K = 2 * np.pi * G * self.rho0
        self.R = R
        self.R3_2 = 3 * R * R

    def evaluate(self, x):
        r = np.linalg.norm(x)
        if r < self.R:
            return self.K * (r * r - self.R3_2)
        else:
            return -self.GM / r

    def gradient(self, x):
        r = np.linalg.norm(x)
        if r < self.R:
            return 2 * self.K * x
        else:
            return (self.GM / r**3) * x
