from gmorbits.structures import Potential
from gmorbits.constants import G
import numpy as np


class KeplerPotential(Potential):
    def __init__(self, M: float, dim: int, center=None):
        self.GM = G * M
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        return -self.GM / np.linalg.norm(x)

    def _gradient(self, x, *args, **kwargs):
        return (self.GM / np.linalg.norm(x) ** 3) * x


class HomogSpherePotential(Potential):
    def __init__(self, M_total: float, R: float, dim: int, center=None):
        self.GM = G * M_total
        self.rho0 = 3 * M_total / (4 * np.pi * R**3)
        self.K = 2 * np.pi * G * self.rho0
        self.R = R
        self.R3_2 = 3 * R * R
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        if r < self.R:
            return self.K * (r * r - self.R3_2)
        else:
            return -self.GM / r

    def _gradient(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        if r < self.R:
            return 2 * self.K * x
        else:
            return (self.GM / r**3) * x


class IsochronePotential(Potential):
    def __init__(self, M: float, b: float, dim: int, center=None):
        self.GM = G * M
        self.b = b
        self.b2 = b * b
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        return -self.GM / (self.b + np.sqrt(np.dot(x, x) + self.b2))

    def _gradient(self, x, *args, **kwargs):
        sqrt_r2b2 = np.sqrt(np.dot(x, x) + self.b2)
        bsr = self.b + sqrt_r2b2
        return self.GM / (sqrt_r2b2 * bsr * bsr) * x


class NFWPotential(Potential):
    def __init__(self, Ms: float, Rs: float, dim: int, center=None):
        # Ms: total mass within scale radius
        # Rs: scale radius
        self.K = 3 * Ms * G
        self.Rs = Rs
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        return -self.K / r * np.log(1 + r / self.Rs)

    def _gradient(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        return self.K / r**3 * (np.log(1 + r / self.Rs) - r / (r + self.Rs)) * x


class LogarithmicPotential(Potential):
    # Singular isothermal sphere
    # Density rho = rho_0 * (r/r_0)^-2 (power law density with alpha = 2)
    def __init__(self, rho_0, r_0, dim: int, center=None):
        self.K = 4 * np.pi * G * rho_0 * r_0 * r_0
        self.logr0 = np.log(r_0)
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        return self.K * (np.log(np.linalg.norm(x)) - self.logr0)

    def _gradient(self, x, *args, **kwargs):
        return self.K / np.dot(x, x) * x


class HernquistPotential(Potential):
    def __init__(self, M, a, dim: int, center=None):
        self.K = G * M
        self.a = a
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        return -self.K / (np.linalg.norm(x) + self.a)

    def _gradient(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        if r == 0:
            print(r)
        interm = r + self.a
        return self.K / (interm * interm * r) * x


class JaffePotential(Potential):
    def __init__(self, M, a, dim: int, center=None):
        self.K = G * M
        self.K_a = self.K / a
        self.a = a
        super().__init__(dim, center)

    def _evaluate(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        return self.K_a * np.log(r / (r + self.a))

    def _gradient(self, x, *args, **kwargs):
        r = np.linalg.norm(x)
        return self.K / (r * r * (r + self.a)) * x
