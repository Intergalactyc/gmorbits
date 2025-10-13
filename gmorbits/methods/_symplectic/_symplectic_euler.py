from gmorbits.structures import Method


class SymplecticEulerMethod(Method):
    def step(self, x, v):
        x1 = x + self.h * v
        v1 = v - self.h * self.potential.gradient(x1)
        return x1, v1
