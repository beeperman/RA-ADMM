from .gd import GD
import numpy as np


# TODO: check for correct implementation
class SimpleNGD(GD):
    def __init__(self, func, m ,M):
        super().__init__(func)
        self.m = m
        self.M = M

        self.step_shape = tuple([2]) # must specify step_size shape

    def fit(self, x0, max_itr, tol=1e-9):
        alpha = 1.0 / self.M
        beta = (np.sqrt(self.M) - np.sqrt(self.m)) / ((np.sqrt(self.M) + np.sqrt(self.m)))
        step_size = [alpha, beta]
        return super().fit(step_size, x0, max_itr, tol)

    @classmethod
    def _update(cls, f, xs, alpha):
        a, b = alpha
        x = xs[-1]
        x_old = xs[-2] if len(xs) > 1 else x
        y = x + b * (x - x_old)
        return y - a * f.grad(y)


