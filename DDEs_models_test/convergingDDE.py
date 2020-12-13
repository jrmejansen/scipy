from scipy.integrate import solve_dde
from scipy.special import factorial
import matplotlib.pyplot as plt
import numpy as np
from jitcdde import jitcdde
from jitcdde import y as y_jit
from jitcdde import t as t_jit

"""
Converging problem
    - Equation : y'(t) = -y(t-1)

Comparison to analytical solution in t [0,10]
"""

def ana(t, tau, mu, phi):
    """
    Solution of $y'(t>=0)=\mu y(t-tau) $
    et $y(t<=0)=h$
    $y(t) = \sum_0^{[t/\tau]+1}
        \frac{(\mu(-t+(n-1)\tau))^n}{n!}$
    t :float
        current time
    tau : float
        delay
    phi : float
        past state t<=t0
    """
    s = 0.
    for n in range(int(np.floor(t/tau)) + 2):
        s += (mu*(-t + (n - 1.) * tau)**n) / \
                factorial(n)
    return s * phi

def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ - y_tau ]

tau = 1
y0 = [1.]
t0 = 0.0
tf = 10.0
atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]

def h(t):
    return [1]

sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', tracked_stages=None,
        atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

#jit cdde
f = [ -y_jit(0,t_jit-tau)]
DDE = jitcdde(f)
DDE.set_integration_parameters(atol=atol,rtol=rtol)
DDE.constant_past(y0)
DDE.step_on_discontinuities()
data = []
t_jit = np.linspace(DDE.t+0.01, tf, 101)
dt_jit = []
for ti in t_jit:
    data.append(DDE.integrate(ti))
    dt_jit.append(DDE.dt)
y_jit = np.asarray(data).T[0,:]

# sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/solConverging_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']


ana_spdev = np.array([ana(t[i], tau, 1.0, 1.0) for i in range(len(t))])
# ana_spdev45 = np.array([ana(t45[i], tau, 1.0, 1.0) for i in range(len(t45))])
ana_mat = np.array([ana(t_mat[i], tau, 1.0, 1.0) for i in range(len(t_mat))])
ana_jit = np.array([ana(t_jit[i],tau, 1.0, 1.0) for i in range(len(t_jit))])

eps = 1e-2
mask = np.abs(y) > eps
mask_jit = np.abs(y_jit) > eps
mask_mat = np.abs(y_mat) > eps
err_spdev = np.abs(np.abs(y[mask] - ana_spdev[mask]) / ana_spdev[mask])
err_jit= np.abs(np.abs(y_jit[mask_jit] - ana_jit[mask_jit]) / ana_jit[mask_jit])
err_mat = np.abs(np.abs(y_mat[mask_mat] - ana_mat[mask_mat]) / ana_mat[mask_mat])

plt.figure()
plt.plot(t, ana_spdev, label="analytic")
plt.plot(t, y, label='scipy-dev y(t)')
plt.plot(t_jit, y_jit, 'o', label='jit y(t)')
plt.plot(t_mat, y_mat, 'o', label='matlab y(t)')
plt.xlabel(r'$t$')
plt.xlabel(r'$y$')
plt.legend()
plt.savefig('figures/solConv/y')

plt.figure()
plt.plot(t[mask], err_spdev, label='scipy-dev err')
plt.plot(t_jit[mask_jit], err_jit, label='jit')
plt.plot(t_mat[mask_mat], err_mat, label="matlab err")
plt.yscale('log')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.savefig('figures/solConv/error')

plt.figure()
plt.plot(y, yp, label='scipy-dev')
plt.plot(y_mat, yp_mat, 'o', label='matlab')
plt.legend()
plt.xlabel('y')
plt.ylabel('dydt')
plt.title('phase graph')
plt.savefig('figures/solConv/phase')

plt.figure()
plt.plot(t_jit, dt_jit,'-o',label='jit')
plt.plot(t[:-1],np.diff(t),'-o',label='dt scipy-dev')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dt matlab solver')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/solConv/dt')
plt.show()

