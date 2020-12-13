from scipy.integrate import solve_dde
from scipy.special import factorial
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

Tested features:
    - initial discontinuities and jumps in histories

Comparison to analytical solution in t [0,5]
"""

def ana(t, tau):
    """
    Solution of $y'(t>=0)=\mu y(t-tau) $
    et $y(t<=0)= -1^{[-5t]}$
    """
    if t < 0.0:
        return (-1)**(np.floor(-5*t))
    elif t < 0.2:
        return t + 1.
    elif t < 0.4:
        return -t + 1.4
    elif t < 0.6:
        return t + 0.6
    elif t < 0.8:
        return -t + 1.8
    elif t < 1.0:
        return t + 0.2
    elif t < 1.2:
        return t**2 * 0.5 + 0.7
    elif t < 1.4:
        return -t**2/2 + 12*t/5 - 0.74
    elif t < 1.6:
        return t**2/2 - 2*t/5 + 1.22
    elif t < 1.8:
        return -t**2/2 + 14*t/5 - 1.34
    elif t < 2.0:
        return t**2/2 - 4*t/5 + 1.9
    elif t < 2.2:
        return t**3/6 - t**2/2 + 6*t/5 + 0.566666666666667
    elif t <2.4:
        return -t**3/6 + 17*t**2/10 - 91*t/25 + 4.116
    elif t < 2.6:
        return t**3/6 - 7*t**2/10 + 53*t/25 - 0.492
    elif t < 2.8:
        return -t**3/6 + 19*t**2/10 - 116*t/25 + 5.36666666666666
    elif t < 3.0:
        return t**3/6 - 9*t**2/10 + 16*t/5 - 1.95066666666668
    elif t < 3.2:
        return t**4/24 - t**3/3 + 27*t**2/20 - 13*t/10 + 1.42433333333332
    elif t < 3.4:
        return -t**4/24 + 11*t**3/15 - 377*t**2/100 + 7217*t/750 - 7.31379999999998
    elif t < 3.6:
        return t**4/24 - 2*t**3/5 + 201*t**2/100 - 2609*t/750 + 3.82233333333336
    elif t < 3.8:
        return -t**4/24 + 4*t**3/5 - 447*t**2/100 + 362200000000001*t/30000000000000 - 10.1744666666667
    elif t < 4.0:
        return t**4/24 - 7*t**3/15 + 11*t**2/4 - 466300000000001*t/75000000000000 + 7.20166666666679
    elif t < 4.2:
        return t**5/120 - t**4/8 + 13*t**3/15 - 31*t**2/12 + 111233333333333*t/25000000000000 - 1.33166666666654
    elif t < 4.4:
        return -t**5/120 + 9*t**4/40 - 311*t**3/150 + 14647*t**2/1500 - 107407333333333*t/5000000000000 + 20.4502053333331
    elif t < 4.6:
        return t**5/120 - 17*t**4/120 + 173*t**3/150 - 6649*t**2/1500 + 2925800000000009*t/300000000000000 - 7.03583200000007
    elif t < 4.8:
        return -t**5/120 + 29*t**4/120 - 178*t**3/75 + 707400000000001*t**2/60000000000000 - 826784000000003*t/30000000000000 + 27.2913306666669
    elif t <= 5.0:
        return t**5/120 - 19*t**4/120 + 22*t**3/15 - 996300000000001*t**2/150000000000000 + 500320000000003*t/30000000000000 - 15.1759973333335

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
sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', jumps=jumps, atol=atol, rtol=rtol)
sol45 = solve_dde(fun, tspan, delays, y0, h, method='RK45', jumps=jumps, atol=atol, rtol=rtol)

t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]
t45 = sol45.t
y45 = sol45.y[0,:]
yp45 = sol45.yp[0,:]


#jit cdde
f = [ y_jit(0,t_jit-tau)]

DDE = jitcdde(f,
		max_delay=20 # for plotting; lest history is forgotten
	)
DDE.set_integration_parameters(atol=atol,rtol=rtol)
DDE.past_from_function(h,times_of_interest=np.linspace(-1.1,-1e-8,10))

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

# sol with rtol=1e-10 atol=1e-12
f90 = np.loadtxt('data_dde_solver_fortran/divJ.dat')
t_f90 = f90[:,0]
y_f90 = f90[:,1]
yp_f90 = f90[:,1]


ana_spdev = np.array([ana(t[i], tau) for i in range(len(t))])
ana_spdev45 = np.array([ana(t45[i], tau) for i in range(len(t45))])
ana_mat = np.array([ana(t_mat[i], tau) for i in range(len(t_mat))])
ana_jit = np.array([ana(times[i],tau) for i in range(len(times))])
ana_f90 = np.array([ana(t_f90[i],tau) for i in range(len(t_f90))])

err_f90 = np.abs(ana_f90 - y_f90) / ana_f90
err_jit = np.abs(ana_jit - y_jit) / ana_jit
err_scipy = np.abs(ana_spdev - y) / ana_spdev
err_scipy45 = np.abs(ana_spdev45 - y45) / ana_spdev45
err_mat = np.abs(ana_mat - y_mat) / ana_mat

plt.figure()
plt.plot(t, ana_spdev, 'o-', label='ana scipy-dev y(t)')
plt.plot(t, y, 'o-', label='scipy-dev y(t)')
plt.plot(times, y_jit, 'o-', label='jit y(t)')
plt.plot(t_mat, y_mat, 'o-', label='matlab y(t)')
plt.plot(t_f90, y_f90, '-', label='f90')
plt.xlabel(r'$t$')
plt.xlabel(r'$y$')
plt.legend()
plt.savefig('figures/solDiv/jumps_y')

plt.figure()
plt.plot(times, err_jit,'-o',label='err jit')
plt.plot(t, err_scipy,'-s',label='err scipy RK23')
plt.plot(t45, err_scipy45,'-o',label='err scipy RK45')
plt.plot(t_mat, err_mat, '-o', label='err mat')
plt.plot(t_f90, err_f90, '-o', label='err f90')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$err$')
plt.xlabel(r'$t$')
plt.savefig('figures/solDiv/jumps_err')

plt.figure()
plt.plot(t[:-1],np.diff(t),'-o',label='dt RK23')
plt.plot(t45[:-1],np.diff(t45),'-o',label='dt RK45')
# plt.plot(t_f90[:-1],np.diff(t_f90),'-o',label='dt f90')
plt.plot(times, dt_jit,'-o',label='dt jit')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dt matlab solver')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/solDiv/jumps_dt')

plt.show()




