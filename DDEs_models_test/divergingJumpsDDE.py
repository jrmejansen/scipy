from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from jitcdde import jitcdde
from jitcdde import y as y_jit
from jitcdde import t as t_jit

"""
The so-called diverging problem with jumps, seen in Example 4 of
Willé & Baker 1992 DELSOL - a numerical code for the solution of systems of 
delay–differential equations.

    - Equation : $y'(t) = y(t-1)$
    - History :  $h(t<0) = (-1)^[-5t]$ with [s] = integer part of s.
Comparison to analytical solution in t [0,10]
"""
def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ y_tau ]

tau = 1
y0 = [1.]
t0 = 0.0
tf = 5
atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]

def h(t):
    return [(-1)**(np.floor(-5*t))]

jumps = sorted([-k*(1/5) for k in range(1,6)])
tspan = [t0, tf]
sol_all = solve_dde(fun, tspan, delays, y0, h, method='RK23', jumps=jumps, atol=atol, rtol=rtol)

t = sol_all.t
y = sol_all.y[0,:]
yp = sol_all.yp[0,:]


#jit cdde
f = [ y_jit(0,t_jit-tau)]

DDE = jitcdde(f,
		max_delay=20 # for plotting; lest history is forgotten
	)
DDE.set_integration_parameters(atol=atol,rtol=rtol)

ε = 1e-8
DDE.add_past_point(-1,1,0)
for i,x in enumerate(np.arange(-0.8,-0.2,0.2)):
	DDE.add_past_point(x-ε,(-1)** i   ,0)
	DDE.add_past_point(x+ε,(-1)**(i+1),0)
DDE.add_past_point(-ε,1,0)
DDE.add_past_point( 0,1,1)
DDE.initial_discontinuities_handled = True

DDE.adjust_diff()
data = []
dt_jit = []
times = np.linspace(DDE.t+0.01, tf, 101)
for time in times:
    data.append(DDE.integrate(time))
    dt_jit.append(DDE.dt)
y_jit = np.asarray(data).T[0,:]


# sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/solDivJumps.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']

p_mat = CubicHermiteSpline(t_mat,y_mat,yp_mat)
y_mat_jit = p_mat(times)
y_mat_solve_dde = p_mat(t)

err_jit = np.abs(y_mat_jit - y_jit)/y_jit
err_mat = np.abs(y_mat_solve_dde - y)/y

plt.figure()
plt.plot(t, y, label='scipy-dev y(t)')
plt.plot(times, y_jit, 'o', label='jit y(t)')
plt.plot(t_mat, y_mat, 'o', label='matlab y(t)')
plt.xlabel(r'$t$')
plt.xlabel(r'$y$')
plt.legend()
plt.savefig('figures/solDiv/jumps_y')

plt.figure()
plt.plot(times, err_jit,'-o',label='err mat jit')
plt.plot(t,err_mat,'-o',label='err mat scipy')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$err$')
plt.xlabel(r'$t$')
plt.savefig('figures/solDiv/jumps_err')

plt.figure()
plt.plot(t[:-1],np.diff(t),'-o',label='dt scipy-dev')
plt.plot(times, dt_jit,'-o',label='dt jit')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dt matlab solver')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/solDiv/jumps_dt')
plt.show()




