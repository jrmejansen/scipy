from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from scipy.interpolate import CubicHermiteSpline

def fun(t,y,Z):
    y_tau1 = Z[:,0]
    y_tau2 = Z[:,1]
    return [-y_tau1 - y_tau2]

tau1 = 3/2
tau2 = 1/3
y0 = [1.]
t0 = 0.0
tf = 10.0
atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau1, tau2]

def h(t):
    return y0

sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

# sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/2delays_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']


p_dev = CubicHermiteSpline(t,y,yp)
y_dev_mat = p_dev(t_mat)


eps = 1e-2
mask = np.abs(y_mat) > eps
err_dev_mat = np.abs(np.abs(y_dev_mat[mask] - y_mat[mask]) / y_mat[mask])

plt.figure()
plt.plot(t, y, label='solve_dde')
plt.plot(t_mat, y_mat, 'o', label='dde23')
plt.xlabel(r'$t$')
plt.xlabel(r'$y(t)$')
plt.legend()
plt.savefig('figures/2delays/y')

plt.figure()
plt.plot(t_mat[mask], err_dev_mat, label='solve_dde/dde23')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.title('relative errors')
plt.savefig('figures/2delays/error')

# plt.figure()
# plt.plot(y, yp, label='scipy-dev')
# plt.plot(y_mat, yp_mat, 'o', label='matlab')
# plt.xlabel('y')
# plt.ylabel('dydt')
# plt.title('phase graph')

plt.figure()
plt.plot(t[:-1],np.diff(t),'-o',label='solve__dde')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dde23')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/2delays/dt')
plt.show()

