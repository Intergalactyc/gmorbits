from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class IsochronePotential(Potential):
    def __init__(self, M: float, b: float):
        self.GM = G * M
        self.b = b
        self.b2 = b * b

    def evaluate(self, x):
        return -self.GM / (self.b + np.sqrt(np.dot(x, x) + self.b2))

    def gradient(self, x):
        sqrt_r2b2 = np.sqrt(np.dot(x, x) + self.b2)
        bsr = self.b + sqrt_r2b2
        return self.GM / (sqrt_r2b2 * bsr * bsr) * x
