import scipy
import time
from scipy.integrate._dde.dde import solve_dde
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

def chuteValise(t,y,Z):
    return np.abs(y[0])-np.pi*.5
chuteValise.direction = 0 # % All events have to be reported
chuteValise.terminal = True

def hitGround(t,y,Z):                                     
    return y[0]                            
hitGround.direction = 0 # % All events have to be reported
hitGround.terminal = True                                 

events = [chuteValise, hitGround]
t1 = time.time()
sol23 = solve_dde(fun, tspan, delays, y0, y0, method='RK23',
                  atol=atol, rtol=rtol ,events=events)
t2 = time.time()
print("\nKind of Event:               scipy-dev         dde23       reference ")
ref = [4.516757065, 9.751053145, 11.670393497];
mat = [4.5167708185, 9.7511043904, 11.6703836720]

e = 0
while(sol23.t[-1]<tf):
    if not (sol23.t_events[0]): # if there is not chuteValise 
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

plt.figure(figsize=(18,14))
plt.plot(t, y,'o', label='scipy-dev y(0)(t)')
plt.legend()

plt.figure(figsize=(18,14))
plt.plot(y, yp, label='phase diagram')
plt.legend()
plt.show()

