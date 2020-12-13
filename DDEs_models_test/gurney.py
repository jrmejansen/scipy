from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from jitcdde import jitcdde
from jitcdde import y as y_jit
from jitcdde import t as t_jit

"""
The revisited blowflies problem (from Gurney 1980, 
Nicholson's blowflies revisited)

Tested features:

    - change right hand side during integration

"""

def fun(t, y, Z, tau, N_D, delta, P):
    y_tau = Z[:,0]
    if t < tau:
        f = [-delta * y]
    else:
        f = [P * y_tau * np.exp(-y_tau / N_D) - delta * y]
    return f

t0 = 0.0
tf = 300.
tau = 12.0

delta = 0.25
P = 10.0
N_D = 300.0
args = (tau, N_D, delta, P)

atol = 1e-10
rtol = 1e-5

tspan = [t0, tf]
delays = [tau]

y0 = [100]

jumps = [tau]
sol = solve_dde(fun, tspan, delays, y0, y0, method='RK23',
            args=args, jumps=jumps, atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

plt.figure()
plt.plot(y,yp, label='solve_dde')
plt.xlabel('y')
plt.ylabel('yp')
plt.legend()
plt.savefig('figures/gurney/phase')

plt.figure()
plt.plot(t, y,label='solve_dde')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('figures/gurney/y')

plt.show()
