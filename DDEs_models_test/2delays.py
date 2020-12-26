from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from scipy.interpolate import CubicHermiteSpline
from jitcdde import jitcdde
from jitcdde import y as y_jit
from jitcdde import t as t_jit

"""
DDE 1 eq, 2 delays : 
    -Equation : y'(t) = -y(t-tau1) - y(t-tau2)
    -Initial and history functions :
      h(t) = 1.0 for t<=t0

Comparison solve_dde, jitcdde, dde23
"""

def fun(t,y,Z):
    y_tau1 = Z[:,0]
    y_tau2 = Z[:,1]
    return [-y_tau1 - y_tau2]

tau1 = 1.
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

#
sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', tracked_stages=None,
        atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

#jit cdde
f = [ -y_jit(0,t_jit-tau1) -y_jit(0,t_jit-tau2)]
DDE = jitcdde(f)
DDE.set_integration_parameters(atol=atol,rtol=rtol)
DDE.constant_past(y0)
DDE.step_on_discontinuities()
print(DDE.t)
data = []
t_jit = np.linspace(DDE.t+0.01, tf+0.01, 101)
dt_jit = []
for ti in t_jit:
    data.append(DDE.integrate(ti))
    dt_jit.append(DDE.dt)
y_jit = np.asarray(data).T[0,:]


# sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/2delays_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']


p_dev = CubicHermiteSpline(t,y,yp)
y_dev_mat = p_dev(t_mat)
y_dev_jit = p_dev(t_jit)


eps = 1e-2
mask_jit = np.abs(y_jit) > eps
err_dev_jit = np.abs(np.abs(y_dev_jit[mask_jit] - y_jit[mask_jit]) / y_jit[mask_jit])
mask_mat = np.abs(y_mat) > eps
err_dev_mat = np.abs(np.abs(y_dev_mat[mask_mat] - y_mat[mask_mat]) / y_mat[mask_mat])

plt.figure()
plt.plot(t, y, label='solve_dde')
plt.plot(t_jit, y_jit, 'o', label='jit')
plt.plot(t_mat, y_mat, 'o', label='dde23')
plt.xlabel(r'$t$')
plt.xlabel(r'$y(t)$')
plt.legend()
plt.savefig('figures/2delays/y')

plt.figure()
plt.plot(t_jit[mask_jit], err_dev_jit, label='solve_dde/jit')
plt.plot(t_mat[mask_mat], err_dev_mat, label='solve_dde/dde23')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.title('relative errors')
plt.savefig('figures/2delays/error')

dt = np.diff(t)
plt.figure()
plt.plot(t_jit, dt_jit,'-o',label='jit')
plt.plot(t[:-1], dt,'-o',label='solve_dde')
plt.vlines(sol.discont, np.min(dt), np.max(dt), label='discont tracked')
plt.plot(t_mat[:-1], np.diff(t_mat),'-o',label='dde23')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/2delays/dt')
plt.show()

