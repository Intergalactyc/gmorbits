import numpy as np
from gmorbits.potentials import KeplerPotential, HernquistPotential
from gmorbits.structures import Integrator, Method
from gmorbits.methods import (
    SymplecticEulerMethod,
    ExplicitEulerMethod,
    HeunMethod,
    LeapfrogMethod,
)
from gmorbits.constants import Msun, Mearth, G

TIMESTEP = 0.01
explicit_euler = ExplicitEulerMethod(h=TIMESTEP)
symplectic_euler = SymplecticEulerMethod(h=TIMESTEP)
heun = HeunMethod(h=TIMESTEP)
leapfrog = LeapfrogMethod(h=TIMESTEP)


def simple_circles(method):
    integrator2d = Integrator(method, 2)
    x0 = np.array([[1, 0], [2, 0], [3, 0]]).astype(float)
    v0 = np.array([[0, 1], [0, 1 / np.sqrt(2)], [0, 1 / np.sqrt(3)]]).astype(float)
    integrator2d.integrate(
        x0, v0, 6 * np.pi * np.sqrt(3), 1.0, KeplerPotential(1, 2), True
    )
    integrator2d.plot_result()


def two_body(method):
    # Something is wrong with this
    integrator2d = Integrator(method, 2)
    x0 = np.array([[0, 0], [1, 0]]).astype(float)
    v0 = np.array([[0, 0], [0, 1]]).astype(float)
    pots = [
        KeplerPotential(5, 2),
        KeplerPotential(1, 2),
    ]
    integrator2d.integrate(x0, v0, 10, [5.0, 1.0], pots, False)
    integrator2d.plot_result(track=True)


def figure_eight(method):
    integrator2d = Integrator(method, 2)
    x0 = np.array([[0.9700436, -0.24308753], [-0.9700436, 0.24308753], [0, 0]]).astype(
        float
    )
    v0 = np.array(
        [
            [0.466203685, 0.43236573],
            [0.466203685, 0.43236573],
            [-0.93240737, -0.86473146],
        ]
    ).astype(float)
    integrator2d.integrate(
        x0,
        v0,
        50,
        1.0,
        KeplerPotential(1, 2),
        False,
    )
    integrator2d.plot_result(track=False)


def random_bodies(method):
    integrator2d = Integrator(method, 2)
    x0 = 10 * (np.random.rand(100, 2) - 0.5)
    v0 = 2 * (np.random.rand(100, 2) - 0.5)
    integrator2d.integrate(
        x0,
        v0,
        25,
        1.0,
        HernquistPotential(1, 1e-2, 2),
        False,
    )
    integrator2d.plot_result(track=False)


# def cluster(method):
#     integrator2d = Integrator(method, 2)
#     x0 = np.array([[0, 0], [1, 1]]).astype(float)
#     v0 = np.array([[0, 0], [1, 1]]).astype(float)
#     pot = [KeplerPotential(100, 2)]
#     integrator2d.integrate(x0, v0, 20, pot, False)
#     integrator2d.plot_result(track=False)


def planet_system(
    method: Method, M: float, distances: list[float], masses: list[float]
):
    if len(distances) != len(masses):
        raise ValueError("distances and masses must be the same length")
    integrator2d = Integrator(method, 2)
    _x0 = [[0, 0]]
    _v0 = [[0, 0]]
    for i, r in enumerate(distances):
        v = np.sqrt(G * M / r)
        match i % 4:
            case 0:
                _x0.append([0, r])
                _v0.append([-v, 0])
            case 1:
                _x0.append([r, 0])
                _v0.append([0, v])
            case 2:
                _x0.append([0, -r])
                _v0.append([v, 0])
            case 3:
                _x0.append([-r, 0])
                _v0.append([0, -v])
    x0 = np.array(_x0).astype(float)
    v0 = np.array(_v0).astype(float)
    pot = [KeplerPotential(M, 2)] + [KeplerPotential(m, 2) for m in masses]
    integrator2d.integrate(x0, v0, 50, [M] + masses, pot, False)
    integrator2d.plot_result(track=False, trails=True)


def trappist_1(method):
    _distances = [0.01154, 0.01580, 0.02227, 0.02925, 0.03849, 0.04683, 0.06189, 0.081]
    distances = [15 * d for d in _distances]
    _masses = [1.374, 1.308, 0.388, 0.692, 1.039, 1.321, 0.326, 1]
    masses = [m * Mearth for m in _masses]
    return planet_system(method, 0.0898 * Msun, distances, masses)


def logarithmic(method):
    integrator2d = Integrator(method, 2)
    _x0 = [[]]


if __name__ == "__main__":
    # simple_circles(symplectic_euler)
    # simple_circles(heun)
    # two_body(symplectic_euler)
    figure_eight(explicit_euler)
    figure_eight(symplectic_euler)
    figure_eight(heun)
    figure_eight(leapfrog)
    # random_bodies(symplectic_euler)
    trappist_1(explicit_euler)
    trappist_1(symplectic_euler)
    trappist_1(heun)
    trappist_1(leapfrog)
