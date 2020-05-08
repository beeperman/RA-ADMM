from .function import FunctionWrapper


class NoisyGradWrapper(FunctionWrapper):
    def __init__(self, func, noise_func):
        """
        This class wrap the function with noise to gradient
        grad = grad + noise_func(grad)
        """
        super().__init__(func)
        self.noise_func = noise_func

    def grad(self, *args, **kwargs):
        grad = self.func.grad(*args, **kwargs)
        return grad + self.noise_func(grad)


# TODO: change based on need
class NoisyProxWrapper(FunctionWrapper):
    def __init__(self, func, noise_func):
        """
        This class wrap the function with noise to proximal operator
        prox = prox + noise_func(prox)
        """
        super().__init__(func)
        self.noise_func = noise_func

    def prox(self, *args, **kwargs):
        prox = self.func.prox(*args, **kwargs)
        return prox + self.noise_func(prox)

class NoisyCallWrapper(FunctionWrapper):
    def __init__(self, func, noise_func):
        """
        This class wrap the function with noise to function call
        f = f + noise_func(x)
        """
        super().__init__(func)
        self.noise_func = noise_func

    def __call__(self, *args, **kwargs):
        f = self.func(*args, **kwargs)
        return f + self.noise_func(*args, **kwargs)
