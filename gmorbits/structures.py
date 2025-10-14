import numpy as np
from abc import ABC, abstractmethod
import os


class Potential:
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    def __add__(self, other):
        return _SumPotential(self, other)

    def plot_gradient_3d(
        self, ax, xmin, xmax, ymin, ymax, zmin, zmax, points_per=8, tolerance=50
    ):
        # ax must have projection="3d"
        posgrid = np.meshgrid(
            np.linspace(xmin, xmax, points_per),
            np.linspace(ymin, ymax, points_per),
            np.linspace(zmin, zmax, points_per),
        )
        gradfield = -np.apply_along_axis(self.gradient, 0, posgrid)
        grad_sizes = np.apply_along_axis(np.linalg.norm, 0, gradfield)
        scale_factor = np.median(grad_sizes)
        print(scale_factor)

        def rescale(x):
            s = x * 8 / scale_factor
            if np.linalg.norm(s) > tolerance:
                return np.array([0.0, 0.0, 0.0])
            return s

        gradfield = np.apply_along_axis(rescale, 0, gradfield)
        # gradfield = rescale(gradfield)
        ax.quiver(*posgrid, *gradfield, length=0.1)

    def plot_contour_2d(self, ax, xmin, xmax, y0, zmin, zmax):
        # 2d contour plot for cross-section y = y0
        pass


class _SumPotential(Potential):
    def __init__(self, pot1, pot2):
        self.evaluate = lambda x: pot1.evaluate(x) + pot2.evaluate(x)
        self.gradient = lambda x: pot1.gradient(x) + pot2.gradient(x)

    def evaluate(self):
        pass

    def gradient(self):
        pass


# class PhaseCoordinates(ABC):
#     def __init__(self):
#         pass

# class CartesianCoordinate(PhaseCoordinates):
#     pass

# class SphericalCoordinates(PhaseCoordinates):
#     pass

# class CylindricalCoordinates(PhaseCoordinates):
#     pass


class Method(ABC):
    def __init__(self, potential: Potential, h: float):
        self.potential = potential
        self.h = h

    @abstractmethod
    def step(self, x, v) -> tuple[np.ndarray, np.ndarray]:
        pass

    def get_T(self, v):
        return np.dot(v, v)

    def get_U(self, x):
        return self.potential.evaluate(x)

    def get_L(self, x, v):
        return np.cross(x, v)

    def iterate(
        self, x0, v0, Tf, saveto: os.PathLike = None
    ) -> tuple[np.ndarray, np.ndarray]:
        N = int(Tf / self.h)

        times = np.arange(0, N + 1) * self.h

        result_x = np.zeros([N + 1, 3])
        result_v = np.zeros([N + 1, 3])
        result_T = np.zeros([N + 1, 1])
        result_U = np.zeros([N + 1, 1])
        result_L = np.zeros([N + 1, 3])

        current_x = np.array(x0)
        current_v = np.array(v0)
        current_T = self.get_T(current_v)
        current_U = self.get_U(current_x)
        current_L = self.get_L(current_x, current_v)

        result_x[0, :] = current_x
        result_v[0, :] = current_v
        result_T[0, :] = current_T
        result_U[0, :] = current_U
        result_L[0, :] = current_L

        for i in range(1, N + 1):
            current_x, current_v = self.step(current_x, current_v)
            current_T = self.get_T(current_v)
            current_U = self.get_U(current_x)
            current_L = self.get_L(current_x, current_v)
            result_x[i, :] = current_x
            result_v[i, :] = current_v
            result_T[i, :] = current_T
            result_U[i, :] = current_U
            result_L[i, :] = current_L
        if saveto is not None:
            self._save(
                times, result_x, result_v, result_T, result_U, result_L, path=saveto
            )
        return times, result_x, result_v, result_T, result_U, result_L

    def _save(self, times, *args, path: os.PathLike = None):
        output = np.hstack([times[:, np.newaxis], *args])
        np.savetxt(path, output, delimiter=",")
