import dill
dill.settings['recurse'] = True
from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np

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

sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

# sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/solConverging_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']

# sol analytique
def expresion(times,fct_np):
    kmax = len(fct_np)
    sol = np.zeros(times.shape)
    Nt = len(times)
    k = 1
    for i in range(Nt):
        if (times[i]<= k*tau):
            sol[i] = fct_np[k](times[i])
        else:#elif(k<kmax):
            k = k+1
            sol[i] = fct_np[k](times[i])
    return sol

fct_np = dill.load(open("data_ana/converging_tf6_tau1.pkl", "rb"))

ana_spdev = expresion(t,fct_np)
ana_mat = expresion(t_mat,fct_np)

eps = 1e-2
mask = np.abs(y) > eps
mask_mat = np.abs(y_mat) > eps
err_spdev = np.abs(np.abs(y[mask] - ana_spdev[mask]) / ana_spdev[mask])
err_mat = np.abs(np.abs(y_mat[mask_mat] - ana_mat[mask_mat]) / ana_mat[mask_mat])

plt.figure()
plt.plot(t, ana_spdev, label="analytic")
plt.plot(t, y, label='scipy-dev y(t)')
plt.plot(t_mat, y_mat, 'o', label='matlab y(t)')
plt.xlabel(r'$t$')
plt.xlabel(r'$y$')
plt.legend()
plt.savefig('figures/solConv/y')

plt.figure()
plt.plot(t[mask], err_spdev, label='scipy-dev err')
plt.plot(t_mat[mask_mat], err_mat, label="matlab err")
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.savefig('figures/solConv/error')

plt.figure()
plt.plot(y, yp, label='scipy-dev')
plt.plot(y_mat, yp_mat, 'o', label='matlab')
plt.xlabel('y')
plt.ylabel('dydt')
plt.title('phase graph')

plt.figure()
plt.plot(t[:-1],np.diff(t),'-o',label='dt scipy-dev')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dt matlab solver')
plt.legend()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.savefig('figures/solConv/dt')
plt.show()

