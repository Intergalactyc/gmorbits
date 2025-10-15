from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class NFWPotential(Potential):
    def __init__(self, Ms: float, Rs: float):
        # Ms: total mass within scale radius
        # Rs: scale radius
        self.K = 3 * Ms * G
        self.Rs = Rs

    def evaluate(self, x):
        r = np.linalg.norm(x)
        return -self.K / r * np.log(1 + r / self.Rs)

    def gradient(self, x):
        r = np.linalg.norm(x)
        return self.K / r**3 * (np.log(1 + r / self.Rs) - r / (r + self.Rs)) * x
