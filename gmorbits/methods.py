from gmorbits.structures import Method


class ExplicitEulerMethod(Method):
    # Non-symplectic
    # First order
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, x, v, pot) -> tuple:
        if (s1 := x.shape) != (s2 := v.shape):
            raise ValueError(f"Incompatible x and v ({s1} != {s2})")
        x1 = x.copy()
        v1 = v.copy()
        for i in range(x.shape[0]):
            x_i = x[i, :]
            v_i = v[i, :]
            x1[i, :] = x_i + self.h * v_i
            v1[i, :] = v_i - self.h * pot.gradient(x_i, i)
        pot.update(x1)
        return x1, v1


class ImplicitEulerMethod(Method):
    pass


class SymplecticEulerMethod(Method):
    # Symplectic
    # First-order
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, x, v, pot) -> tuple:
        if (s1 := x.shape) != (s2 := v.shape):
            raise ValueError(f"Incompatible x and v ({s1} != {s2})")
        x1 = x.copy()
        v1 = v.copy()
        for i in range(x.shape[0]):
            x_i = x[i, :]
            v_i = v[i, :]
            v1[i, :] = v_i - self.h * pot.gradient(x_i, i)
            x1[i, :] = x_i + self.h * v1[i, :]
        pot.update(x1)
        return x1, v1


class HeunMethod(Method):
    # Non-symplectic
    # Second-order
    # AKA Explicit Trapezoidal rule
    # A second-order RK method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, x, v, pot) -> tuple:
        if (s1 := x.shape) != (s2 := v.shape):
            raise ValueError(f"Incompatible x and v ({s1} != {s2})")
        xtilde = x.copy()
        vtilde = v.copy()
        fs = []
        for i in range(x.shape[0]):
            f = pot.gradient(x[i, :], i)
            fs.append(f)
            xtilde[i, :] = x[i, :] + self.h * v[i, :]
            vtilde[i, :] = v[i, :] - self.h * f
        pot.update(xtilde)
        x1 = x.copy()
        v1 = v.copy()
        for i in range(x.shape[0]):
            x1[i, :] = x[i, :] + self.h / 2 * (v[i, :] + vtilde[i, :])
            v1[i, :] = v[i, :] - self.h / 2 * (fs[i] + pot.gradient(xtilde[i, :], i))
        pot.update(x1)
        return x1, v1


class LeapfrogMethod(Method):
    # Symplectic
    # Second-order
    # Equivalent to Verlet
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, x, v, pot) -> tuple:
        if (s1 := x.shape) != (s2 := v.shape):
            raise ValueError(f"Incompatible x and v ({s1} != {s2})")
        x1 = x.copy()
        v1 = v.copy()
        v2 = v.copy()
        for i in range(x.shape[0]):
            v1[i, :] = v[i, :] - self.h / 2 * pot.gradient(x[i, :], i)
            x1[i, :] = x[i, :] + self.h * v1[i, :]
        pot.update(x1)
        for i in range(x.shape[0]):
            v2[i, :] = v1[i, :] - self.h / 2 * pot.gradient(x1[i, :], i)
        return x1, v2
