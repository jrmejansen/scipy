
from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

"""
DDE with 2 delays and a discontinuity at initial time
    -Equation : y'(t) = -y(t-tau1) - y(t-tau2)
    -Initial and history functions :
      y(t0) = 1.5
      h(t) = 1.0 for t<t0

Tested features:
    - initial discontinuities
    - dense output with init discont

"""
def fun(t,y,Z):
    y_tau1 = Z[:,0]
    y_tau2 = Z[:,1]
    return [-y_tau1 - y_tau2]

tau1 = 1.
tau2 = 1./3.

t0 = 0.0
tf = 10.0
atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau1, tau2]

y0 = [1.5]
def h(t):
    return [1.0]

sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', dense_output=True,
        atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

# # sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/2delays_discont_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']


p_dev = CubicHermiteSpline(t,y,yp)
y_dev_mat = p_dev(t_mat)

eps = 1e-2
mask = np.abs(y_mat) > eps
err_dev_mat = np.abs(np.abs(y_dev_mat[mask] - y_mat[mask]) / y_mat[mask])

t_n = np.append(np.array([t0]), np.linspace(t0,tf,200))
plt.figure()
plt.plot(t, y, label='solve_dde y')
plt.plot(t_n, sol.sol(t_n)[0,:], label='interp y')
plt.plot(t, yp, label='yp')
plt.plot(t_mat, y_mat, 'o', label='dde23 y')
plt.plot(t_mat, yp_mat, '^', label='dde23 yp')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.savefig('figures/2delays_discont/y')

plt.figure()
plt.plot(t_mat[mask], err_dev_mat, label='solve_dde/dde23')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.title('relative errors')
plt.savefig('figures/2delays_discont/error')

plt.figure()
plt.plot(y, yp, label='scipy-dev')
plt.plot(y_mat, yp_mat, 'o', label='matlab')
plt.legend()
plt.xlabel('y')
plt.ylabel('dydt')
plt.title('phase graph')
plt.savefig('figures/2delays_discont/phase')

dt = np.diff(t)

plt.figure()
plt.plot(t[:-1], dt, '-o',label='solve__dde')
plt.vlines(sol.discont, np.min(dt), np.max(dt), label='discont tracked')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dde23')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/2delays_discont/dt')
plt.show()

