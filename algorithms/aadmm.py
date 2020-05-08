from .admm import ADMM
import numpy as np


# TODO: check for correct implementation
class AADMM(ADMM):
    def __init__(self, *args, **kwargs):
        """ note that x0 = [x0, z0, v0, a0, zh0, vh0]"""
        super().__init__(*args, **kwargs)

        self.step_shape = tuple() # must specify step_size shape rho

    @classmethod
    def _update(cls, f, g, A, B, C, xs, rho):
        x, z, v, a_old, zh, vh = xs[-1]
        z_old = xs[-2][1] if len(xs) > 1 else z
        v_old = xs[-2][2] if len(xs) > 1 else v
        x = f.prox(-(B * zh - C + v) / A, alpha=1.0 / (rho * A ** 2))
        z = g.prox(-(A * x - C + v) / B, alpha=1.0 / (rho * B ** 2))
        v = vh + (A * x + B * z - C)
        a = (1 + np.sqrt(1 + 4 * a_old ** 2)) / 2.0
        zh = z + (a-1) / a_old * (z - z_old)
        vh = v + (a-1) / a_old * (v - v_old)
        return x, z, v, a, zh, vh