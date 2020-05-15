from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ beta * y_tau[0] / (1 + y_tau[0]**n) - gamma*y[0] ]

t0 = 0.0
tf = 100.
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

t1 = time.time()
sol23 = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
t2 = time.time()
print('for RK23 Elapsed time is %s seconds' % (t2-t1))
t = sol23.t
y = sol23.y[0,:]
yp = sol23.yp[0,:]

t1 = time.time()
sol45 = solve_dde(fun, tspan, delays, y0, h, method='RK45', atol=atol, rtol=rtol)
t2 = time.time()
print('for RK45 Elapsed time is %s seconds' % (t2-t1))
t45 = sol45.t
y45 = sol45.y[0,:]
yp45 = sol45.yp[0,:]

# julia 
path_ju = 'data_julia'
t_ju = np.load('%s/solMackeyGlassBS3_t.npz' % path_ju)
y_ju = np.load('%s/solMackeyGlassBS3_u.npz' % path_ju)[:,0]

# sol matlab
import scipy.io as spio
path_matlab = 'data_dde23/mackeyGlass_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']

p_dev = CubicHermiteSpline(t,y,yp)
y_dev_ju = p_dev(t_ju)
p_mat = CubicHermiteSpline(t_mat,y_mat,yp_mat)
y_mat_ju = p_mat(t_ju)

err_mat_ju = np.abs(y_mat_ju - y_ju)/y_ju
err_dev_ju = np.abs(y_dev_ju - y_ju)/y_ju

plt.figure()
plt.plot(y,yp)
plt.xlabel('y')
plt.ylabel('yp')
plt.savefig('figures/mackeyGlass/phase')

plt.figure()
plt.plot(t_mat, y_mat, 'r o', label='matlab y(t)')
plt.plot(t_ju, y_ju, 'g s',label='julia y(t)')
plt.plot(t, y, 'b *',label='scipy-dev y(t)')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('figures/mackeyGlass/y')

plt.figure()
plt.plot(t_ju, err_mat_ju, label='err mat / ju')
plt.plot(t_ju, err_dev_ju, 'o', label='err dev / ju')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.title('error relative')
plt.savefig('figures/mackeyGlass/error.png')

plt.figure()
plt.plot(t[:-1],np.diff(t),'-o',label='solve_dde')
plt.plot(t_ju[:-1],np.diff(t_ju),'-o',label='julia DDE BS3')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='matlab dde23')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta t$')
plt.savefig('figures/mackeyGlass/dt.png')
plt.show()
