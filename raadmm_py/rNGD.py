# Implementation of Robust Momentum Method [Hu et al., 2018]
import dill
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

# from copy import deepcopy as dcopy

# Define parameters
A = np.array([[1, 0], [0, 10]])
b = np.array([[3], [0]])
x_opt = np.linalg.inv(A).dot(-b)  # optimal solution
abs_eig = np.linalg.eigvals(A)
abs_eig.sort()
L = abs_eig[-1]
m = abs_eig[0]
kappa = L / m
max_iter = 200
tol = 1e-3
# ---------------------------------------------------------------------------
delta = 0.5  # noise level
rho = 1 - 1 / np.sqrt(kappa)  # algorithm parameter
x0 = 10 * np.random.rand(2, 1)  # the initial condition
alpha = kappa * (1 - rho) ** 2 * (1 + rho) / L
beta = kappa * rho ** 3 / (kappa - 1)
gamma = rho ** 3 / ((kappa - 1) * (1 - rho) * (1 + rho))
mu = (1 + rho) * (1 - kappa + 2 * kappa * rho - kappa * rho ** 2) / (2 * rho)


def obj(x):
    return 1 / 2 * x.T.dot(A).dot(x) + b.T.dot(x)


def grad(x):
    return A.dot(x) + b


def update(x, x_last, method='GD'):
    if method == 'GD':
        gradient = grad(x)
        noise = - delta * gradient  # gradient noise
        x_next = x - 1 / L * (gradient + noise)
    elif method == 'NGD':
        beta_ = (np.sqrt(L) - np.sqrt(m)) / (np.sqrt(L) + np.sqrt(m))
        y = x + beta_ * (x - x_last)
        gradient = grad(y)
        noise = - delta * gradient  # gradient noise
        x_next = y - 1 / L * (gradient + noise)
    else:
        y = x + gamma * (x - x_last)
        gradient = grad(y)
        noise = - delta * gradient  # gradient noise
        x_next = x + beta * (x - x_last) - alpha * (gradient + noise)

    err = np.log(np.linalg.norm(x_next - x_opt, 2))
    return x_next, err, obj(x), obj(x_next)


# ---------------------------------------------------------------------------
# The belows are the main codes

x = np.array(x0)
err = np.array(0)
method = ['GD', 'NGD', 'rNGD']
for k in range(max_iter):
    if k == 0:
        x_next, err_next, _, _ = update(x, [])
    else:
        x_next, err_next, _, _ = update(x[:, -1].reshape(-1, 1), x[:, -2].reshape(-1, 1), 'rNGD')

    x = np.hstack([x, x_next])
    err = np.hstack([err, err_next])

plt.figure()
plt.plot(np.arange(0, len(err), 1), err)
plt.xlabel('iterations')
plt.ylabel('error')
plt.savefig('test.png')

save_time = str(datetime.now())
filename = 'test.pkl'
dill.dump_session(filename)

# plt.show()

# def rNGD_update(x, x_last):
#     gradient = grad(x + gamma * (x - x_last))
#     noise = - delta * gradient  # gradient noise
#     x_next = x + beta * (x - x_last) - alpha * (gradient + noise)
#     err = np.log(np.linalg.norm(x_next - x_opt, 2))
#     return x_next, err, obj(x), obj(x_next)
# def GD_update(x):
#     gradient = grad(x)
#     noise = - delta * gradient  # gradient noise
#     x_next = x - 1 / L * (gradient + noise)
#     err = np.log(np.linalg.norm(x_next - x_opt, 2))
#     return x_next, err, obj(x), obj(x_next)
# def NGD_update(x, x_last):
#     beta = (np.sqrt(L) - np.sqrt(m)) / (np.sqrt(L) + np.sqrt(m))
#     alpha = 1 / L
#     y = x + beta * (x - x_last)
#     gradient = grad(y)
#     noise = - delta * gradient  # gradient noise
#     x_next = y - alpha * (gradient + noise)
#     err = np.log(np.linalg.norm(x_next - x_opt, 2))
#     return x_next, err, obj(x), obj(x_next)
