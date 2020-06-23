import time
from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

t0 = 0.0
tf = 12
tspan = [t0, tf]
tau = .1
gamma = 0.248
beta  = 1
A = 0.75
omega = 1.37
eta = np.arcsin(gamma/A);
y0 = [0.0, 0.0]
def h(t):
    return y0

atol = 1e-10
rtol = 1e-5
delays = [tau]

def fun(t,y,Z):
    y_tau = Z[:,0]
    return [y[1],
            np.sin(y[0]) - np.sign(y[0]) * gamma * np.cos(y[0]) - beta * y_tau[0]
            + A * np.sin(omega * t + eta)]

def finalEvent(t,y,Z):
    return np.abs(y[0])-np.pi*.5
finalEvent.direction = 0 # % All events have to be reported
finalEvent.terminal = True

def hitGround(t,y,Z):                                     
    return y[0]                            
hitGround.direction = 0 # % All events have to be reported
hitGround.terminal = True                                 

events = [finalEvent, hitGround]
sol23 = solve_dde(fun, tspan, delays, y0, y0, method='RK23',
                  atol=atol, rtol=rtol ,events=events)
print("\nKind of Event:               solve_dde         dde23       reference    DDE_SOLVER")
# ref values of matlab dde23 example script 
ref = np.array([4.516757065, 9.751053145, 11.670393497])
# computed values from matlab dde23 with same atol & rtol
mat = np.array([4.5167708185, 9.7511043904, 11.6703836720])
# from DDE_SOLVER  fortran routine example : Example 4.4.5: Events and Change Routine
f90 = np.array([4.5167570861630821, 9.7510847727976273, 11.670385883524640])

e = 0
while(sol23.t[-1]<tf):
    if not (sol23.t_events[0]): # if there is not finalEvent 
        print('A wheel hit the ground. ',sol23.t[-1],'',mat[e],'',ref[e],'',f90[e])
        t_val = np.array([sol23.t[-1],mat[e],ref[e],f90[e]])
        print('relative error to ref   ', np.abs(t_val-ref[e])/ref[e])
        y0 = [0.0, sol23.y[1,-1]*0.913]
        tspan = [sol23.t[-1],tf]
        sol23 = solve_dde(fun, tspan, delays, y0, sol23, method='RK23',
                  atol=atol, rtol=rtol ,events=events)
        e += 1
    else:
        print("The suitcase fell over. ",sol23.t[-1],'',mat[e],'',ref[e],'',f90[e])
        t_val = np.array([sol23.t[-1],mat[e],ref[e],f90[e]])
        print('relative error to ref   ', np.abs(t_val-ref[e])/ref[e])
        break

t = sol23.t
y = sol23.y[0,:]
yp = sol23.y[1,:]

path = 'data_dde23/suitcase_dde23.mat'
import scipy.io as spio
mat = spio.loadmat(path, squeeze_me=True)
t_mat = mat['t']
y_mat = mat['y'][0,:]
yp_mat = mat['y'][1,:]


plt.figure(figsize=(18,14))
plt.plot(t, y,'o', label=r'solve_dde $\theta(t)$')
plt.plot(t, yp,'o', label=r'solve_dde $\dot{\theta}(t)$')
plt.plot(t_mat, y_mat,'-', label=r'dde23 $\theta(t)$')
plt.plot(t_mat, yp_mat,'-', label=r'dde23 $\dot{\theta}(t)$')
plt.legend()
plt.savefig('figures/suitecase/t_y_yp')

plt.figure(figsize=(14,12))
plt.plot(y, yp, 'o-', label='solve_dde')
plt.plot(y_mat, yp_mat,'o',markerfacecolor='none', label='dde23 from Matlab')
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$\dot{\theta}$', fontsize=20)
plt.legend()
plt.savefig('figures/suitecase/phase_diag')
plt.show()
