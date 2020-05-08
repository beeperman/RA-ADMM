from functions import *
from autograd import numpy as np

class sq(BasicFunction):
    def __call__(self, x):
        return np.sum(np.square(x))

s = sq()
print(s.prox(2.0))

q = Quadratic.simple_random(2)
print(q.A)
print(q(np.ones((2,1))))
print(q.grad(np.ones((2,1))))
print(q.prox(np.ones((2,1))))
