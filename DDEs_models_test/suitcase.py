import time
from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np

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
print('y0', y0)
sol23 = solve_dde(fun, tspan, delays, y0, y0, method='RK23',
                  atol=atol, rtol=rtol ,events=events)
print("\nKind of Event:               scipy-dev         dde23       reference ")
ref = [4.516757065, 9.751053145, 11.670393497];
mat = [4.5167708185, 9.7511043904, 11.6703836720]

e = 0
while(sol23.t[-1]<tf):
    if not (sol23.t_events[0]): # if there is not finalEvent 
        print('A wheel hit the ground. ',sol23.t[-1],'',mat[e],'',ref[e])
        y0 = [0.0, sol23.y[1,-1]*0.913]
        tspan = [sol23.t[-1],tf]
        sol23 = solve_dde(fun, tspan, delays, y0, sol23, method='RK23',
                  atol=atol, rtol=rtol ,events=events)
        e += 1
    else:
        print("The suitcase fell over. ",sol23.t[-1],'',mat[e],'',ref[e])
        break

t = sol23.t
y = sol23.y[0,:]
yp = sol23.y[1,:]

path = 'data_dde23/suitcase_dde23.mat'
import scipy.io as spio
mat = spio.loadmat(path, squeeze_me=True)
t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']

plt.figure(figsize=(18,14))
plt.plot(t, y,'o', label='scipy-dev y(0)(t)')
plt.plot(t_mat, y_mat[0,:],'-', label='scipy-dev y(0)(t)')
plt.legend()

plt.figure(figsize=(14,12))
plt.plot(y, yp, 'o-', label='solve_dde')
plt.plot(y_mat[0,:], y_mat[1,:],'o',markerfacecolor='none', label='dde23 from Matlab')
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$\dot{\theta}$', fontsize=20)
plt.legend()
plt.savefig('figures/suitecase/phase_diag')
plt.show()
