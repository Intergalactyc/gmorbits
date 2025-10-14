from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class KeplerPotential(Potential):
    def __init__(self, M, x0=np.array([0.0, 0.0, 0.0])):
        self.GM = G * M
        self.x0 = x0

    def evaluate(self, x):
        return -self.GM / np.linalg.norm(x - self.x0)

    def gradient(self, x):
        return (self.GM / np.linalg.norm(x - self.x0) ** 3) * (x - self.x0)

    def update(self, x0):
        self.x0 = x0
