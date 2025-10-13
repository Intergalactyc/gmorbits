from gmorbits.structures import Method


class ExplicitEulerMethod(Method):
    def step(self, x, v):
        x1 = x + self.h * v
        v1 = v - self.h * self.potential.gradient(x)
        return x1, v1
