from scipy.integrate._dde import solve_dde
#from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

sol_ar = []

def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ beta * y_tau[0] / (1 + y_tau[0]**n) - gamma*y[0] ]

t0 = 0.0
tf = 100.
tf2 = 200.0
tau = 15.0

n = 10
beta = 0.25
gamma = 0.1

atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]
y0 = [1.0]
def h(t):
    return [1.]

sol23 = solve_dde(fun, tspan, delays, y0, h, method='RK23', dense_output=True,
        atol=atol, rtol=rtol)
t = sol23.t
y = sol23.y[0,:]
yp = sol23.yp[0,:]

t_restart = sol23.t[-1] * 0.6
print('\n restart before tf =%s  \
       \n at t0 = %s' % (sol23.t[-1],t_restart))

tspan = [t_restart,  tf2]
y0 = sol23.sol(t_restart)

sol23 = solve_dde(fun, tspan, delays, y0, sol23, method='RK23', atol=atol, rtol=rtol)

sol_ar.append(sol23)
t = sol23.t
y = sol23.y[0,:]
yp = sol23.yp[0,:]

print('\n calcul dun seul coup t0 tf2   \n')

tspan = [t0,  tf2]
y0 = [1]
sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
sol_ar.append(sol)

t_all = sol.t
y_all = sol.y[0,:]
yp_all = sol.yp[0,:]

print('relative error last y val = ',
        np.abs((y_all[-1] - y[-1]) / y_all[-1]))
print('err matlab 5.7900e-06')
plt.figure()
plt.plot(t, y, '*',label='y restart')
plt.plot(t_all, y_all, label='y all')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()

plt.figure()
plt.plot(t, yp, '*',label='y_p restart')
plt.plot(t_all, yp_all, '*',label='y_p all')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()

plt.figure()
plt.plot(t, yp/y, '*',label='solve_dde y_p/y')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()

plt.figure()
plt.plot(y,yp,'o', label='phase restart')
plt.plot(y_all,yp_all,'o', label='phae all')
plt.xlabel('y')
plt.ylabel('yp')
plt.legend()

plt.show()


