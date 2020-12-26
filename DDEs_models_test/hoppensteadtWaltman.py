from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from jitcdde import jitcdde
from jitcdde import y as y_jit
from jitcdde import t as t_jit

import warnings
warnings.simplefilter("ignore")

"""
The Hoppensteadt-Waltman model (Example 5 from Oberle et al 1981
Numerical Treatment of Delay Differential Equations by Hermite Interpolation)

Tested features:
    - piecewise DDEs

Comparison with dde23, jitcdde and ref value from Oberle.
"""

r = 0.5
mu = r / 10.0
c = np.sqrt(2)**-1

def fun(t,y,Z):
    if t <= 1 - c:
        f = -r * y[0] * 0.4 * (1 - t)
    elif t <= 1:
        f = -r * y[0] * (0.4 * (1 - t) + 10.0 - np.exp(mu) * y[0])
    elif t <= 2 - c:
        f = -r * y[0] * (10. - np.exp(mu) * y[0])
    else:
        f = -r * np.exp(mu) * y[0] * (Z[:,0] - y[0])
    return [f]

tau = 1.0
y0 = [10.0]
jumps = [1.0 - c, 1.0, 2.0 - c]
t0 = 0.0
tf = 10.0
atol = 1e-8
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]


sol = solve_dde(fun, tspan, delays, y0, y0,
                    method='RK23', jumps=jumps, atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]


# #jitcdde
from symengine import exp
ts = [1-c,1,2-c,tf]
fs = [
		[-r * y_jit(0) * 0.4 * (1 - t_jit)],
		[-r * y_jit(0) * (0.4 * (1 - t_jit) + 10.0 - exp(mu) * y_jit(0))],
		[-r * y_jit(0) * (10. - exp(mu) *y_jit(0))],
		[-r * exp(mu) * y_jit(0) * ( y_jit(0,t_jit-tau) - y_jit(0))]
	]

from chspy import CubicHermiteSpline
histo = CubicHermiteSpline(n=1)
histo.constant(y0)

print(ts)
y_jit = []
dt_jit = []
t_jit = []
for target_time,f in zip(ts,fs):
	DDE = jitcdde(f,max_delay=tau)
	DDE.set_integration_parameters(atol=atol,rtol=rtol)
	DDE.add_past_points(histo)
	DDE.adjust_diff()
	for ti in np.linspace(DDE.t,target_time,100):
		t_jit.append(ti)
		y_jit.append(DDE.integrate(ti)[0])
		dt_jit.append(DDE.dt)
	histo = DDE.get_state()
	histo.truncate(target_time)



mat = 0.06301980845
ref = 0.06302089869
f90 = np.loadtxt('data_dde_solver_fortran/hoppensteadtWaltman.dat')
t_f90 = f90[:,0]
y_f90 = f90[:,1]
print(' solve_dde =  ', y[-1], 'err', np.abs(y[-1]-ref)/ref)
print(' dde23 y(10) =  ', mat, 'err', np.abs(mat-ref)/ref)
print(' jitcdde y(10) =  ', y_jit[-1], 'err', np.abs(y_jit[-1]-ref)/ref)
print(' f90    y(10) =  ', y_f90[-1], 'err', np.abs(y_f90[-1]-ref)/ref)
print(' Reference solution y(10) =  ', ref)

I = -(1/r)*(yp / y)

plt.figure()
plt.plot(t, y, label='solve_dde')
plt.plot(t_jit, y_jit, label='jit')
plt.plot(t_f90, y_f90, label='f90')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.savefig('figures/hoppensteadtWaltman/y')

plt.figure()
plt.plot(t[:-1], np.diff(t), label='solve_dde')
plt.plot(t_jit, dt_jit, label='jit')
plt.plot(t_f90[:-1], np.diff(t_f90), label='f90')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta t$')
plt.legend()
plt.savefig('figures/hoppensteadtWaltman/dt')

plt.figure()
plt.plot(t, I, label='I(t)')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$I(t)$')
plt.savefig('figures/hoppensteadtWaltman/I')
plt.show()
