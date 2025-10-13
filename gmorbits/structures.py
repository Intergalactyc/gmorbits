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

    def iterate(
        self, x0, v0, Tf, saveto: os.PathLike = None
    ) -> tuple[np.ndarray, np.ndarray]:
        N = int(Tf / self.h)
        times = np.arange(0, N + 1) * self.h
        result_x = np.zeros([N + 1, 3])
        result_v = np.zeros([N + 1, 3])
        current_x = np.array(x0)
        current_v = np.array(v0)
        result_x[0, :] = current_x
        result_v[0, :] = current_v
        for i in range(1, N + 1):
            current_x, current_v = self.step(current_x, current_v)
            result_x[i, :] = current_x
            result_v[i, :] = current_v
        if saveto is not None:
            self._save(times, result_x, result_v, saveto)
        return times, result_x, result_v

    def _save(self, times, result_x, result_v, path: os.PathLike):
        output = np.hstack([times[:, np.newaxis], result_x, result_v])
        np.savetxt(path, output, delimiter=",", header="time,x,y,z,vx,vy,vz")
