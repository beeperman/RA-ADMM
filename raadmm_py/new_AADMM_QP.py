# Problem statement:
# min 1/2 * x^T M x + 1/2 z^T N z,      s.t. x = z

import numpy as np
from matplotlib import pyplot as plt

# Quadratic Programming problem parameters
rho = 1
M = np.random.rand(10, 10)
N = np.random.rand(10, 10)
M = 1*(M + M.T + 10 * np.eye(10))
N = 10* (N + N.T + 10 * np.eye(10))
max_iter = 100


# New algorithm parameters
m = 1 / ((np.linalg.eigvals(M).max() + np.linalg.eigvals(N).max()) + rho)
# L = 1/((np.linalg.eigvals(M).min() + np.linalg.eigvals(N).min()) + rho)
L = 1 / rho
kappa_ = L / m
# rho_ = 0.5 * (1 - 1 / kappa_ ** (0.5) + 1 - 1 / kappa_)
rho_ = 1 - 1 / kappa_**(0.5)
alpha_ = kappa_ * (1 - rho_) ** 2 * (1 + rho_) / (L)
beta_ = kappa_ * rho_ ** 3 / (kappa_ - 1)
gamma_ = rho_ ** 3 / ((kappa_ - 1) * (1 - rho_) ** 2 * (1 + rho_))


def f(x):
    return 1 / 2 * x.T.dot(M).dot(x)


def g(z):
    return 1 / 2 * z.T.dot(N).dot(z)


def prox_f(z, v):
    return np.linalg.inv(M + rho * np.eye(10)).dot(rho * z - v)


def prox_g(x, v):
    return np.linalg.inv(N + rho * np.eye(10)).dot(rho * x + v)


# Accelerated ADMM in [Rene Vidal, 2018 ICML paper]:
def A_ADMM_update(z, v, z_hat, v_hat, k):
    x_next = prox_f(z_hat, v_hat)
    z_next = prox_g(x_next, v_hat)
    noise = 0 * np.random.randn(1, 1)
    v_next = v_hat + (x_next - z_next)*(1 + noise)
    v_hat_next = v_next + k / (k + 3) * (v_next - v)
    z_hat_next = z_next + k / (k + 3) * (z_next - z)
    err_next = np.log(f(x_next) + g(z_next))
    return x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise


# New Accelerated ADMM designed according to [Bin Hu, 2018 ACC paper]:
def A_ADMM_update2(z, v, z_hat, v_hat, v_last, k):
    x_next = prox_f(z_hat, v_hat)
    z_next = prox_g(x_next, v_hat)
    noise = 0 * np.random.randn(1, 1)   # Gradient Noise
    v_next = v + beta_ * (v - v_last) + alpha_ * (x_next - z_next)*(1 + noise)
    v_hat_next = v_next + gamma_ * (v_next - v)
    # z_hat_next = z_next + gamma_ * (z_next - z)           # Alternative option
    z_hat_next = prox_g(x_next + gamma_ * (x_next - x), v_hat_next)           # Alternative option
    err_next = np.log(f(x_next) + g(z_next))
    return x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise


# -------------------------------------------------------------------------------------
# The belows are the main codes
x = np.random.rand(10, 1)
z = np.random.rand(10, 1)
v = np.random.rand(10, 1)
z_hat = np.array(z)
v_hat = np.array(v)
err = f(x) + g(z)
noise = np.array([[0]])
for k in np.arange(max_iter):
    if k == 0:
        x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise_next = A_ADMM_update(z[:, -1].reshape(-1, 1),
                                                                                             v[:, -1].reshape(-1, 1),
                                                                                             z_hat[:, -1].reshape(-1,
                                                                                                                  1),
                                                                                             v_hat[:, -1].reshape(-1,
                                                                                                                  1),
                                                                                             k)
    else:
        x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise_next = A_ADMM_update2(z[:, -1].reshape(-1, 1),
                                                                                              v[:, -1].reshape(-1, 1),
                                                                                              z_hat[:, -1].reshape(-1,
                                                                                                                   1),
                                                                                              v_hat[:, -1].reshape(-1,
                                                                                                                   1),
                                                                                              v[:, -2].reshape(-1,
                                                                                                               1),
                                                                                              k)

    x = np.hstack([x, x_next])
    z = np.hstack([z, z_next])
    v = np.hstack([v, v_next])
    z_hat = np.hstack([z_hat, z_hat_next])
    v_hat = np.hstack([v_hat, v_hat_next])
    err = np.hstack([err, err_next])
    noise = np.hstack([noise, noise_next])

plt.figure()
plt.plot(np.arange(err.size-1), err[0,1:].reshape(-1, 1))
plt.title('Robust Accelerated ADMM')
plt.xlabel('iterations')
plt.ylabel('error')
plt.savefig('New_A_ADMM_QP.png')
