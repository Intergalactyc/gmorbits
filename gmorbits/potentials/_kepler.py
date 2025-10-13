from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class KeplerPotential(Potential):
    def __init__(self, M):
        self.M = M

    def evaluate(self, x):
        return -G * self.M / np.linalg.norm(x)

    def gradient(self, x):
        return (G * self.M / np.linalg.norm(x) ** 3) * x
