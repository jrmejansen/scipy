from scipy.integrate import solve_dde
from scipy.special import factorial
import matplotlib.pyplot as plt
import numpy as np
from jitcdde import jitcdde
from jitcdde import y as y_jit
from jitcdde import t as t_jit

"""
The so-called diverging problem, seen in Willé & Baker 1992 
DELSOL - a numerical code for the solution of systems of 
delay–differential equations.

    - Equation : y'(t) = y(t-1)
    - h(t<=t0) = 1.0

Comparison to analytical solution in t [0,10]

"""

def ana(t, tau, mu, phi):                    
    """                                      
    Solution of $y'(t>=0)=\mu y(t-tau) $     
    et $y(t<=0)=h$                           
    $y(t) = \sum_0^{[t/\tau]+1}              
        \frac{(\mu(t-(n-1)\tau))^n}{n!}$     
    t :float                                 
        current time                         
    tau : float                              
        delay                                
    phi : float                              
        past state t<=t0          

    Reference
    --------
    Baker, 1995, Issues in the numerical solution of evolutionary delay 
        differential equations, Advances in Computational Mathematics.
    """                                      
    s = 0.                                   
    for n in range(int(np.floor(t/tau)) + 2):
        s += (mu*(t - (n - 1.) * tau)**n) / \
                factorial(n)                 
    return s * phi                           


def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ + y_tau ]

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

sol23 = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
t = sol23.t
y = sol23.y[0,:]
yp = sol23.yp[0,:]

sol45 = solve_dde(fun, tspan, delays, y0, h, method='RK45', atol=atol, rtol=rtol)
t45 = sol45.t
y45 = sol45.y[0,:]
yp45 = sol45.yp[0,:]

#jit cdde
f = [ y_jit(0,t_jit-tau)]
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
path_matlab = 'data_dde23/solDiverging_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']


ana_spdev = np.array([ana(t[i], tau, 1.0, 1.0) for i in range(len(t))])
ana_spdev45 = np.array([ana(t45[i], tau, 1.0, 1.0) for i in range(len(t45))])
ana_mat = np.array([ana(t_mat[i], tau, 1.0, 1.0) for i in range(len(t_mat))])
ana_jit = np.array([ana(t_jit[i],tau, 1.0, 1.0) for i in range(len(t_jit))])

err_spdev = np.abs(np.abs(y - ana_spdev) / ana_spdev)
err_spdev45 = np.abs(np.abs(y45 - ana_spdev45) / ana_spdev45)
err_jit = np.abs(np.abs(y_jit - ana_jit) / ana_jit)
err_mat = np.abs(np.abs(y_mat - ana_mat) / ana_mat)

plt.figure()
plt.plot(t, ana_spdev, label='analytic')
plt.plot(t, y, label='solve_dde RK23')
plt.plot(t_jit, y_jit, 'o', label='jit y(t)')
plt.plot(t_mat, y_mat, 'o', label='matlab dde23')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('figures/solDiv/y')

plt.figure()
plt.plot(t, err_spdev, label='solve_dde RK23')
plt.plot(t45, err_spdev45, label='solve_dde RK45')
plt.plot(t_jit, err_jit, label='jit')
plt.plot(t_mat, err_mat, label="matlab err")
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.legend()
plt.title('relative errors')
plt.savefig('figures/solDiv/error')

plt.figure()
plt.plot(y, yp, label='solve_dde RK23')
plt.plot(y_mat, yp_mat, 'o', label='dde23')
plt.xlabel('y')
plt.ylabel('dydt')
plt.legend()
plt.title('phase graph')

plt.figure()
plt.plot(t_jit, dt_jit,'-o',label='jit')
plt.plot(t[:-1],np.diff(t),'-o',label='solve_dde RK23')
plt.plot(t45[:-1],np.diff(t45),'-o',label='solve_dde RK45')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dde23')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/solDiv/dt')

plt.show()
