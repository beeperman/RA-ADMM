import numpy as np
from functions import Quadratic
from scipy import optimize
from matplotlib import pyplot as plt

seed = 0
rs = np.random.RandomState(seed)

n = 10
itrs = 200
rho = 0.05
tol = 1.0

L = 100
m = 1
k = L / m
alpha = 1/L

def get_proj(y, A, b, M):
    return y + M @ (b - A @ y)

def get_A():
    #return 2 * rs.randn(n, 2*n)
    A = np.sqrt(m) * np.eye(n, 2*n)
    for i in range(int(n/2)):
        A[i, i] = np.sqrt(L)
    return A

f = Quadratic.simple_random(n, 2, 1)

A = get_A()
M = A.T @ np.linalg.pinv(A @ A.T)
x_ = 2 * rs.rand(2*n, 1) - 1
#x_ = np.zeros_like(x_)
b = A @ x_

lbd = np.zeros_like(b)
x0 = 2 * rs.rand(2*n, 1) - 1
xs = []

for i in range(itrs):
    l = lambda x: 0.5 * np.linalg.norm(x, 2) ** 2 + np.sum(lbd.T @ (A @ x - b))# + rho / 2.0 * np.square(np.linalg.norm(A@x - b, ord=2))
    g = lambda x: x + A.T @ lbd# + rho * A.T @ (A @ x - b)
    wl = lambda x: l(np.reshape(x, newshape=(-1, 1)))
    wg = lambda x: g(np.reshape(x, newshape=(-1, 1))).flatten()

    res = optimize.minimize(wl, np.zeros_like(x0), jac=wg)
    x0 = np.reshape(res.x, newshape=(-1, 1)) #+ (2 * rs.rand(2*n, 1) - 1) * tol

    xs.append((x0, lbd))

    x0 = x0

    lbd = lbd + alpha * (A @ x0 - b) * (1 + rs.randn(n, 1) * tol)


l = lambda x, lbdd: 0.5 * np.linalg.norm(x, 2) ** 2 + np.sum(lbdd.T @ (A @ x - b))# + rho / 2.0 * np.square(np.linalg.norm(A@x - b, ord=2))
f = lambda x: 0.5 * np.linalg.norm(x, 2) ** 2# + rho / 2.0 * np.square(np.linalg.norm(A@x - b, ord=2))
ls = []
fs = []
fds = []
x_opt = M @ b
f_opt = f(x_opt)
ds = []
gps = []
for x in xs:
    x_proj = get_proj(x[0], A, b, M)
    ls.append(l(x[0], x[1]))
    fs.append(f(x_proj))
    fds.append(fs[-1] - f_opt)
    ds.append(np.linalg.norm(x[0] - x_opt))
    gps.append(fs[-1] - ls[-1])
#plt.plot(ls, label="laguangian")
#plt.plot(fds, label="f-f*")
plt.semilogy(ds, label="x - x*")
#plt.semilogy(gps, label="dual gap")
#plt.plot(gps, label="dual gap")
plt.legend()
plt.show()

import pickle as pkl

with open("lm_{}_tol{}.pkl".format(seed, tol), 'wb') as f:
    pkl.dump({"fds": fds, "ds": ds, "gps": gps}, f)

A
