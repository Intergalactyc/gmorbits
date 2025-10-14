from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class IsochronePotential(Potential):
    def __init__(self, M: float, b: float):
        self.b = b
        self.b2 = b * b

    def evaluate(self, x):
        return -G * self.M / (self.b + np.sqrt(np.dot(x, x) + self.b2))

    def gradient(self, x):
        pass
