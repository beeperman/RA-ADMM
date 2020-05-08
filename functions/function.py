import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize


class Function(object):
    """
    Function abstract class
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def grad(self, *args, **kwargs):
        raise NotImplementedError()

    def prox(self, *args, **kwargs):
        raise NotImplementedError()


class FunctionWrapper(Function):
    def __init__(self, func):
        """
        This class wrap the given function
        """
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def prox(self, *args, **kwargs):
        return self.func.prox(*args, **kwargs)

    def grad(self, *args, **kwargs):
        return self.func.grad(*args, **kwargs)

class BasicFunction(Function):
    """
    Function class that has prox method and grad method. It is your job to ensure the function is convex proper.
    """
    def __init__(self):
        self._grad_init = True
        self._prox_init = True
        self._grad_func = None
        self.x_shape = None

    def __call__(self, x):
        """ overload this instead of _func if multiple input or mutable input shape """
        if self.x_shape is None:
            self.x_shape = np.shape(x)
        return self._func(x)

    def grad(self, x):
        if self.x_shape is not None:
            assert np.shape(x) == self.x_shape
        return self._grad(x)

    def prox(self, v, alpha=1.0):
        assert alpha > 0
        if self.x_shape is not None:
            assert np.shape(v) == self.x_shape
        return self._prox(v, alpha)


    def _func(self, x):
        """ return func(x) """
        raise NotImplementedError()

    def _grad(self, x):
        """ return grad(func(x)) """
        if self._grad_init:
            print("Using default grad implementation based on autograd package")
            self._grad_init = False
            self._grad_func = grad(self)

        return self._grad_func(x)
        raise NotImplementedError()

    def _prox(self, v, alpha=1.0):
        """ return prox_{\alpha func}(v) """
        if self._prox_init:
            print("Using default prox implementation based on scipy L-BFGS optimizer")
            self._prox_init = False

        p_af = lambda x: alpha * self(x) + 1/2 * np.linalg.norm(x-v, ord=2)
        dp_af = lambda x: alpha * self.grad(x) + x - v
        res = minimize(p_af, v, jac=dp_af)
        x = res.x
        if np.shape(v) == tuple(): # single input
            return x[0]
        else:
            return x
        return NotImplementedError()
