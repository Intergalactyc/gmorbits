import numpy as np
from abc import ABC, abstractmethod
from gmorbits.constants import _EPSILON
from gmorbits.plot import plot_result as _plot_result
import os
from tqdm import tqdm
from copy import deepcopy


class Potential(ABC):
    def __init__(self, dim, center=None):
        self.dim = dim
        self.center = center or np.array([0.0 for _ in range(dim)])

    def update_position(self, value):
        self.center = value

    def update(self, *args, **kwargs):
        # Update function called in method step - only relevant for _BulkPotential (regular Potential should accept call but do nothing)
        return

    def evaluate(self, x, *args, **kwargs):
        return self._evaluate(x - self.center, *args, **kwargs)

    def gradient(self, x, *args, **kwargs):
        return self._gradient(x - self.center, *args, **kwargs)

    @abstractmethod
    def _evaluate(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def _gradient(self, x, *args, **kwargs):
        pass

    def __add__(self, other):
        return _SumPotential(self, other)


class _SumPotential(Potential):
    def __init__(self, pot1, pot2):
        if (d1 := pot1.dim) != (d2 := pot2.dim):
            raise ValueError(
                f"Potentials pot1 of incompatible dimensions ({d1} != {d2})"
            )

        def EVAL(x, *args, **kwargs):
            return pot1.evaluate(x, *args, **kwargs) + pot2.evaluate(x, *args, **kwargs)

        def GRAD(x, *args, **kwargs):
            return pot1.gradient(x, *args, **kwargs) + pot2.gradient(x, *args, **kwargs)

        self._evaluate = EVAL
        self._gradient = GRAD
        super().__init__(pot1.dim)

    def _evaluate(self, *args, **kwargs):
        pass

    def _gradient(self, *args, **kwargs):
        pass


class _BulkPotential(Potential):
    def __init__(self, x0, potentials: list | Potential):
        # x0: count x dim array
        try:
            self.size = x0.shape[0]
            dim = x0.shape[1]
            if isinstance(potentials, Potential):
                self._potentials = [deepcopy(potentials) for _ in range(x0.shape[0])]
            elif isinstance(potentials, list):
                if (n1 := x0.shape[0]) != (n2 := len(potentials)):
                    raise ValueError(
                        f"The same number of positions and potentials must be specified (got {n1} != {n2})"
                    )
                self._potentials = potentials
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Incompatible x0 passed to BulkPotential.__init__ ({e})")
        super().__init__(dim)
        self.update(x0)

    def _evaluate(self, x, i):
        return sum(p.evaluate(x) for j, p in enumerate(self._potentials) if i != j)

    def _gradient(self, x, i):
        return sum(p.gradient(x) for j, p in enumerate(self._potentials) if i != j)

    def update(self, xs):
        try:
            if xs.shape[0] != self.size or xs.shape[1] != self.dim:
                raise ValueError(
                    f"Incompatibly sized object xs ({xs.shape}; expected ({self.size}, {self.dim})) passed to BulkPotential.update"
                )
        except (AttributeError, IndexError) as e:
            raise ValueError(
                f"Incompatible object xs passed to BulkPotential.update ({e})"
            )
        for i, p in enumerate(self._potentials):
            p.update_position(xs[i, :])


def specific_angular_momentum(position, velocity, x0=0.0, dim=3):
    if dim == 3:
        return np.cross(position - x0, velocity)
    elif dim == 2:
        return position[0] * velocity[1] - position[1] * velocity[0]


class Method(ABC):
    def __init__(self, h: float):
        self.h = h

    def step(self, x, v, pot: Potential) -> tuple:
        if (s1 := x.shape) != (s2 := v.shape):
            raise ValueError(f"Incompatible x and v ({s1} != {s2})")
        pot.update(x)
        x_new, v_new = self._step(x, v, pot)
        return x_new, v_new

    @abstractmethod
    def _step(self, x, v, pot: Potential) -> tuple:
        # x, v: count x dim matrices
        # Relevant: -pot.gradient(), pot.update(xi)
        # Update pot object (to use updated x) and return updated values of x and v
        pass


# For 1 particle system have something to visualize phase space (x-vx and y-vy)
class Integrator(ABC):
    def __init__(self, method: Method, dim: int):
        if not (isinstance(dim, int) and dim in {2, 3}):
            raise ValueError("Dimension (dim) must be 2 or 3")
        self.dim = dim
        self.method = method
        self._results = {}
        self._latest = None

    def integrate(
        self,
        x0,
        v0,
        tf,
        mass: float | list[float],
        potential: Potential | list[Potential],
        static: bool,
        twoform: bool = True,
        name: str = None,
        progress_bar: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # x0, v0 each count x dim arrays (each row is a vector for the initial pos or vel of a particle)
        # if potential is a Potential, it is assumed to be a single static potential within which all particles move without mutual interaction
        # if potential is a BulkPotential, it is assumed to be associated with x0
        try:
            if x0.shape[1] != self.dim or v0.shape[1] != self.dim:
                raise ValueError(
                    f"Initial positions and velocities must be of dimension {self.dim}"
                )
            count = x0.shape[0]
            if count != v0.shape[0]:
                raise ValueError(
                    f"x0 and v0 must be specified for all particles (got {count} x0 but {v0.shape[0]} v0)"
                )
        except AttributeError as e:
            raise ValueError(
                f"Incompatible x0 or v0 passed to Integrator.integrate ({e})"
            )

        if isinstance(mass, list):
            if len(mass) != count:
                raise ValueError(f"List of masses must be of length {count}")
            _mass = mass
        else:
            _mass = [mass for _ in range(count)]

        if isinstance(potential, list) and static:
            raise ValueError("List of potentials only valid for non-static mode")
        if static:
            _potential = potential
        else:
            _potential = _BulkPotential(x0, potential)

        Nt = int(tf / self.method.h)
        times = np.arange(0, Nt + 1) * self.method.h

        if twoform:
            n = self.dim * count  # n = 2N
            # canonical J for (q,p)
            I = np.eye(n)
            J = np.block([[np.zeros((n, n)), I], [-I, np.zeros((n, n))]])
            # initial perturbations in canonical coordinates (q,p)
            eps_rel = _EPSILON
            # we'll set eps later after we form z
            delta1 = np.zeros(2 * n)
            delta2 = np.zeros(2 * n)
            ws = np.zeros(Nt)
        else:
            ws = None

        xs = np.concatenate(
            [x0[np.newaxis, :, :], np.zeros([Nt, count, self.dim])], axis=0
        )
        vs = np.concatenate(
            [v0[np.newaxis, :, :], np.zeros([Nt, count, self.dim])], axis=0
        )
        Ts = np.zeros(Nt + 1)
        Us = np.zeros(Nt + 1)
        Ls = np.zeros([Nt + 1, 3 if self.dim == 3 else 1])

        for j in range(count):
            Us[0] += _mass[j] * _potential.evaluate(xs[0, j, :], j)
            Ts[0] += _mass[j] * np.dot(vs[0, j, :], vs[0, j, :])
            Ls[0, :] += _mass[j] * specific_angular_momentum(
                xs[0, j, :], vs[0, j, :], dim=self.dim
            )

        iterator = range(1, Nt + 1)
        if progress_bar:
            iterator = tqdm(iterator)
        for i in iterator:
            x = xs[i - 1, :, :]
            v = vs[i - 1, :, :]

            x_new, v_new = self.method.step(x, v, _potential)

            xs[i, :, :], vs[i, :, :] = x_new, v_new

            if twoform:
                # build canonical z = (q, p) where p = m * v
                m_diag = np.repeat(_mass, self.dim)
                p = v.ravel() * m_diag  # p at current state (before step)
                z = np.concatenate(
                    [x.ravel(), p]
                )  # z corresponds to the state we linearize about

                # set eps scaled to state size if not set
                state_scale = max(1.0, np.linalg.norm(z))
                eps = eps_rel * state_scale

                # initialize on first step if zeros
                if np.all(delta1 == 0):
                    delta1[0] = eps
                    delta2[n] = eps

                # apply integrator to a canonical z (q,p) -> returns canonical z'=(q',p')
                def Phi_of_z(z_can):
                    q_p = z_can[:n].reshape(count, self.dim)
                    p_p = z_can[n:].copy()
                    # convert p back to v for integrator call
                    v_p = (p_p / m_diag).reshape(count, self.dim)
                    x_pf, v_pf = self.method.step(q_p, v_p, _potential)
                    p_pf = v_pf.ravel() * m_diag
                    return np.concatenate([x_pf.ravel(), p_pf])

                # central finite difference push-forward
                def push_forward_central(delta):
                    z_plus = z + delta
                    z_minus = z - delta
                    Phi_plus = Phi_of_z(z_plus)
                    Phi_minus = Phi_of_z(z_minus)
                    return 0.5 * (Phi_plus - Phi_minus)

                # propagate deviations
                d1_next = push_forward_central(delta1)
                d2_next = push_forward_central(delta2)

                # compute two-form using canonical J
                w = d1_next @ (J @ d2_next)
                ws[i - 1] = w

                # update stored deltas for next step (represent them as deviations in canonical coords)
                delta1 = d1_next
                delta2 = d2_next

            for j in range(count):
                Us[i] += _mass[j] * _potential.evaluate(xs[i, j, :], j)
                Ts[i] += _mass[j] * np.dot(vs[i, j, :], vs[i, j, :])
                Ls[i, :] += _mass[j] * specific_angular_momentum(
                    xs[i, j, :], vs[i, j, :], dim=self.dim
                )

        if name is None:
            count = 0
            while (nn := f"run{count}") in self._results:
                count += 1
            name = nn

        self._results[name] = (times, xs, vs, Ts, Us, Ls, ws)
        self._latest = name

    def plot_result(self, which=None, saveto: os.PathLike = None, *args, **kwargs):
        if which is None:
            which = self._latest
        res = self._results.get(which)
        if not res:
            raise Exception("Integrator has not been run")

        _plot_result(res, self.dim, saveto, *args, **kwargs)
