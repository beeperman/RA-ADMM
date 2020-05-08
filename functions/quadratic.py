from .function import BasicFunction
import numpy as np


class Quadratic(BasicFunction):
    def __init__(self, A, b, c):
        """
        Assuming A psd
        """
        super().__init__()

        A_shape = np.shape(A)
        b_shape = np.shape(b)
        c_shape = np.shape(c)

        assert len(A_shape) == 2 and A_shape[0] == A_shape[1]
        assert len(b_shape) == 2 and b_shape[0] == A_shape[0] and b_shape[1] == 1
        assert len(c_shape) == 0

        self.A = A
        self.b = b
        self.c = c

        self.n = A_shape[0]

    def _func(self, x):
        x_shape = np.shape(x)
        assert x_shape[0] == self.n and x_shape[1] == 1
        return np.sum(1/2 * x.T @ self.A @ x + self.b.T @ x + self.c)

    def _grad(self, x):
        return self.A @ x + self.b

    def _prox(self, v, alpha=1.0):
        return np.linalg.inv(self.A + 1 / alpha * np.eye(self.n)) @ (v / alpha - self.b)

    @classmethod
    def simple_random(cls, n, M=1, m=0.01):
        """ Return a simple random (M, m)-convex quaduatic function """
        assert M > m and m >= 0

        v = np.random.rand(n, 1)
        v /= np.linalg.norm(v, ord=2)

        A = m * np.eye(n) + (M-m) * v @ v.T
        b = np.random.rand(n, 1)
        c = 0

        return cls(A, b, c)

