# Problem statement:
# min 1/2 * x^T M x + 1/2 z^T N z,      s.t. x = z

import numpy as np
from matplotlib import pyplot as plt

rho = 1
M = np.random.rand(10, 10)
N = np.random.rand(10, 10)
M = 0.1 * (M + M.T + 10 * np.eye(10))
N = 15 * (N + N.T + 10 * np.eye(10))
max_iter = 400


def f(x):
    return 1 / 2 * x.T.dot(M).dot(x)


def g(z):
    return 1 / 2 * z.T.dot(N).dot(z)


def prox_f(z, v):
    return np.linalg.inv(M + rho * np.eye(10)).dot(rho * z - v)


def prox_g(x, v):
    return np.linalg.inv(N + rho * np.eye(10)).dot(rho * x + v)


# Accelerated ADMM:
def A_ADMM_update(z, v, z_hat, v_hat, k):
    x_next = prox_f(z_hat, v_hat)
    z_next = prox_g(x_next, v_hat)
    v_next = v_hat + (x_next - z_next) * (1 + 2 * np.random.randn(1, 1))  # perturbed gradient
    v_hat_next = v_next + k / (k + 3) * (v_next - v)
    z_hat_next = z_next + k / (k + 3) * (z_next - z)
    err_next = np.log(f(x_next) + g(z_next))
    return x_next, z_next, v_next, z_hat_next, v_hat_next, err_next


# -------------------------------------------------------------------------------------
# The belows are the main codes
x = np.random.rand(10, 1)
z = np.random.rand(10, 1)
v = np.random.rand(10, 1)
z_hat = np.array(z)
v_hat = np.array(v)
err = f(x) + g(z)
for k in np.arange(max_iter):
    x_next, z_next, v_next, z_hat_next, v_hat_next, err_next = A_ADMM_update(z[:, -1].reshape(-1, 1),
                                                                             v[:, -1].reshape(-1, 1),
                                                                             z_hat[:, -1].reshape(-1, 1),
                                                                             v_hat[:, -1].reshape(-1, 1),
                                                                             k)

    x = np.hstack([x, x_next])
    z = np.hstack([z, z_next])
    v = np.hstack([v, v_next])
    z_hat = np.hstack([z_hat, z_hat_next])
    v_hat = np.hstack([v_hat, v_hat_next])
    err = np.hstack([err, err_next])

plt.figure()
plt.plot(np.arange(err.size), err.reshape(-1, 1))
plt.xlabel('iterations')
plt.ylabel('error')
# plt.show()
plt.savefig('test_ADMM.png')
