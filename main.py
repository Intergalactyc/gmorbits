import numpy as np
import pandas as pd
from gmorbits.potentials import (
    KeplerPotential,
    # HernquistPotential,
    LogarithmicPotential,
)
from gmorbits.structures import Integrator, Method
from gmorbits.methods import (
    SymplecticEulerMethod,
    ExplicitEulerMethod,
    HeunMethod,
    LeapfrogMethod,
)
from gmorbits.constants import Msun, Mearth, G

TIMESTEP = 0.02
FINAL = 30.0

PROG = False

explicit_euler = ExplicitEulerMethod(h=TIMESTEP)
symplectic_euler = SymplecticEulerMethod(h=TIMESTEP)
heun = HeunMethod(h=TIMESTEP)
leapfrog = LeapfrogMethod(h=TIMESTEP)


# def simple_circles(method):
#     integrator2d = Integrator(method, 2)
#     x0 = np.array([[1, 0], [2, 0], [3, 0]]).astype(float)
#     v0 = np.array([[0, 1], [0, 1 / np.sqrt(2)], [0, 1 / np.sqrt(3)]]).astype(float)
#     integrator2d.integrate(
#         x0, v0, 6 * np.pi * np.sqrt(3), 1.0, KeplerPotential(1, 2), True
#     )
#     integrator2d.plot_result()


# def two_body(method):
#     # Something is wrong with this
#     integrator2d = Integrator(method, 2)
#     x0 = np.array([[0, 0], [1, 0]]).astype(float)
#     v0 = np.array([[0, 0], [0, 1]]).astype(float)
#     pots = [
#         KeplerPotential(5, 2),
#         KeplerPotential(1, 2),
#     ]
#     integrator2d.integrate(x0, v0, 10, [5.0, 1.0], pots, False)
#     integrator2d.plot_result(track=True)


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
    return integrator2d.integrate(
        x0, v0, FINAL, 1.0, KeplerPotential(1, 2), False, progress_bar=PROG
    )


# def random_bodies(method):
#     integrator2d = Integrator(method, 2)
#     x0 = 10 * (np.random.rand(100, 2) - 0.5)
#     v0 = 2 * (np.random.rand(100, 2) - 0.5)
#     integrator2d.integrate(
#         x0,
#         v0,
#         25,
#         1.0,
#         HernquistPotential(1, 1e-2, 2),
#         False,
#     )
#     integrator2d.plot_result(track=False)


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
    return integrator2d.integrate(
        x0, v0, FINAL, [M] + masses, pot, False, progress_bar=PROG
    )


def trappist_1(method):
    _distances = [0.01154, 0.01580, 0.02227, 0.02925, 0.03849, 0.04683, 0.06189]
    distances = [15 * d for d in _distances]
    _masses = [1.374, 1.308, 0.388, 0.692, 1.039, 1.321, 0.326]
    masses = [m * Mearth for m in _masses]
    return planet_system(method, 0.0898 * Msun, distances, masses)


def logarithmic(method):
    integrator2d = Integrator(method, 2)
    pot = LogarithmicPotential(1, 1, 2)
    x0 = np.array([[1, 0], [-5, 0], [2, 2], [4, 4]]).astype(float)
    v0 = np.array([[0, pot.vc], [0, -pot.vc], [1, 0], [3, 1]]).astype(float)
    return integrator2d.integrate(x0, v0, FINAL, 1, pot, True, progress_bar=PROG)


METHODS = {
    "Explicit Euler": explicit_euler,
    "Symplectic Euler": symplectic_euler,
    "Heun": heun,
    "Leapfrog": leapfrog,
}

CASES = {
    "Logarithmic Potential": logarithmic,
    "Figure Eight": figure_eight,
    "Trappist-1": trappist_1,
}


def _rms(x):
    return np.mean((x - np.mean(x)) ** 2)


if __name__ == "__main__":
    from gmorbits.plot import plot_result
    from gmorbits.constants import _EPSILON
    import pandas as pd

    if not PROG:
        from tqdm import tqdm

        pbar = tqdm(total=len(CASES) * len(METHODS))
    for cname, testcase in CASES.items():
        data = []
        for mname, method in METHODS.items():
            res = testcase(method)
            plot_result(res, cname, mname, "./outputs", trails=True)
            _, _, _, T, U, L, w = res
            E = T + U
            RMS_E = _rms(E)
            RMS_L = _rms(L)
            w /= _EPSILON**2
            RMS_w = _rms(w)
            w_maxdev = np.max(np.abs(w - w[0]))
            data.append([RMS_E, RMS_L, RMS_w, w_maxdev])
            if not PROG:
                pbar.update(1)
        df = pd.DataFrame(
            data,
            columns=[
                "RMS Energy Error",
                "RMS Angular Momentum Error",
                "RMS 2-form Error",
                "Max 2-form Drift",
            ],
            index=METHODS.keys(),
        )
        df.to_csv(cname + ".csv")
