import time
from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np

def fun(t,y,Z):
    y_tau1 = Z[:,0]
    y_tau10 = Z[:,1]
    return [-y[0] * y_tau1[1]  + y_tau10[1],
            y[0] * y_tau1[1] -  y[1],
            y[1] - y_tau10[1]]

def zero_y0(t,y,Z):
    y_tau1 = Z[:,0]
    y_tau10 = Z[:,1]
    return -y[0] * y_tau1[1]  + y_tau10[1]
zero_y0.direction = -1
zero_y0.terminal = False

def zero_y1(t,y,Z):
    y_tau1 = Z[:,0]
    return y[0] * y_tau1[1] -  y[1]
zero_y1.direction = -1
zero_y1.terminal = False

def zero_y2(t,y,Z):
    y_tau10 = Z[:,1]
    return y[1] - y_tau10[1]
zero_y2.direction = -1
zero_y2.terminal = False

t0 = 0.0
tf = 40
tau1 = 1.
tau10 = 10.
gamma = 0.248;
beta  = 1;
A = 0.75;
omega = 1.37;
eta = np.arcsin(gamma/A);

y0 = [5.,.1,1.0]
delays = [tau1,tau10]
tspan = [t0, tf]

zeros = [zero_y0,zero_y1,zero_y2]
atol = 1e-10
rtol = 1e-5
t1 = time.time()
sol23 = solve_dde(fun, tspan, delays, y0, y0, method='RK23', events=zeros, atol=atol, rtol=rtol)
t2 = time.time()

t = sol23.t
y0 = sol23.y[0,:]
y1 = sol23.y[1,:]
y2 = sol23.y[2,:]

y0_e = sol23.y_events[0][:,0]
y1_e = sol23.y_events[1][:,1]
y2_e = sol23.y_events[2][:,2]

import scipy.io as spio
path_mat = 'data_dde23/virusEvents.mat'
mat = spio.loadmat(path_mat, squeeze_me=True)
t0_e_mat = mat['x1']
t1_e_mat = mat['x2']
t2_e_mat = mat['x3']

te_mat = [t0_e_mat, t1_e_mat, t2_e_mat]

for i in range(len(sol23.t_events)):
    err = np.abs((sol23.t_events[i] - te_mat[i]) / te_mat[i])
    print('     t_event y%s :' % (i))
    print("         solver_dde method='RK23' = %s" % (sol23.t_events[i]))
    print('         Matlab dde23 =             %s' % (sol23.t_events[i]))
    print('         relative error =           %s ' % (err))

y0_e_mat = mat['y1']

y1_e_mat = mat['y2']
y2_e_mat = mat['y3']

plt.figure(figsize=(14,10))
plt.plot(t,y0,'r',label='y0')
plt.plot(t,y1,'b',label='y1')
plt.plot(t,y2,'g',label='y2')
plt.plot(sol23.t_events[0],y0_e,'s r', markersize=10)
plt.plot(sol23.t_events[1],y1_e,'s b', markersize=10)
plt.plot(sol23.t_events[2],y2_e,'s g', markersize=10)
plt.plot(t0_e_mat,y0_e_mat,'C1 o',label='matlab')
plt.plot(t1_e_mat,y1_e_mat,'C1 o')
plt.plot(t2_e_mat,y2_e_mat,'C1 o')
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('figures/virus/virus')
plt.show()
