from functions import Function
import numpy as np

class GD(object):
    def __init__(self, func):
        assert isinstance(func, Function)
        self.func = func
        self.total_itr = 0
        self.xs_all = []
        self.xs = None      # will use xs to contain all possible sequences of interest

        # for different step_shape must change this
        self.step_shape = tuple()

    def fit(self, step_size, x0, max_itr, tol=1e-9):
        assert x0 is not None
        assert max_itr > 0 and isinstance(max_itr, int)
        if np.shape(step_size) != self.step_shape: # not one element
            assert len(step_size) == max_itr
        else:
            step_size = np.repeat(np.expand_dims(step_size, axis=0), max_itr, axis=0)

        self.xs = self._fit(self.func, step_size, x0, tol)
        self.xs_all += self.xs
        self.total_itr += len(self.xs)
        return self.get_x()

    def get_x(self):
        """ if multiple sequences in x, may need different x getter"""
        return self.xs[-1]

    @classmethod
    def _fit(cls, func, step_size, x0, tol):
        xs = [x0]
        for alpha in step_size:
            x = cls._update(func, xs, alpha)
            xs.append(x)
            if np.linalg.norm(x - xs[-2]) < tol:
                print("update change: {}, smaller than tol: {}. done.".format(np.linalg.norm(x - xs[-2]), tol))
                break
        return xs

    @classmethod
    def _update(cls, f, xs, alpha):
        x = xs[-1]
        x_new = x - alpha * f.grad(x)
        return x_new
