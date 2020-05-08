from functions import Function
import numpy as np

class ADMM(object):
    def __init__(self, f, g, A, B, C):
        """ Only simple A, B are supported. Hence scalars"""
        assert isinstance(f, Function)
        assert isinstance(g, Function)
        self.f = f
        self.g = g
        self.A = A
        self.B = B
        self.C = C

        self.total_itr = 0
        self.xs_all = []
        self.xs = None      # will use xs to contain all possible sequences of interest

        # for different step_shape must change this
        self.step_shape = tuple()

    def fit(self, x0, rho, max_itr, tol=1e-9):
        assert x0 is not None
        assert max_itr > 0 and isinstance(max_itr, int)
        if np.shape(rho) != self.step_shape: # not one element
            assert len(rho) == max_itr
            step_size = rho
        else:
            step_size = np.repeat(np.expand_dims(rho, axis=0), max_itr, axis=0)

        self.xs = self._fit(
            self.f, self.g, self.A, self.B, self.C,
            x0, step_size, tol)

        self.xs_all += self.xs
        self.total_itr += len(self.xs)
        return self.get_xz()

    def get_xz(self):
        """ if multiple sequences in x,z , may need different x,z getter"""
        return self.xs[-1][0], self.zs[-1][1]

    @classmethod
    def _fit(cls, f, g, A, B, C, x0, step_size, tol):
        xs = [x0]
        for rho in step_size:
            x, z, v = cls._update(f, g, A, B, C, xs, rho)
            xs.append(x)
            if np.linalg.norm(x - xs[-2]) < tol:
                print("update change: {}, smaller than tol: {}. done.".format(np.linalg.norm(x - xs[-2]), tol))
                break
        return xs

    @classmethod
    def _update(cls, f, g, A, B, C, xs, rho):
        x, z, v = xs[-1]
        x = f.prox(-(B*z-C+v)/A, alpha=1.0/(rho*A**2))
        z = g.prox(-(A*x-C+v)/B, alpha=1.0/(rho*B**2))
        v = v + (A*x + B*z - C)
        return x, z, v
