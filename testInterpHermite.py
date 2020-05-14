import scipy
import time
from scipy.integrate._dde.dde import solve_dde
import matplotlib.pyplot as plt
import numpy as np



def fun(t,y,Z):
    y_tau = Z[:,0]
    return [-y_tau[0], y_tau[0]]


t0 = 0.0
tf = 20
tau = 1.

y0 = [1.0, 1.0]

def h(t):
    return [1 - t, 1 + t]

delays = [tau]
tspan = [t0, tf]


atol = 1e-10
rtol = 1e-5
t1 = time.time()
sol23 = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
t2 = time.time()
t = sol23.t
yc = sol23.y[0,:]
yd = sol23.y[1,:]

t_p = np.linspace(-tau,t0,100)
y_p = np.array([1-t_p, 1+t_p])
dy_p = -np.ones(y_p.shape)
mytuple = (t_p, y_p, dy_p)
t1 = time.time()
sol23_tuple = solve_dde(fun, tspan, delays, y0, mytuple, method='RK23', atol=atol, rtol=rtol)
t2 = time.time()
t_tuple = sol23_tuple.t
yc_tuple = sol23_tuple.y[0,:]
yd_tuple = sol23_tuple.y[1,:]
interp_sol23_tuple = sol23_tuple.sol(t)[0,:]

print('err',(interp_sol23_tuple-sol23.y[0,:])/sol23.y[0,:])

plt.plot(t,yc,'o-',label='yc from fct')
plt.plot(t,yd,'s-',label='yd from fct')
plt.plot(t_tuple,yc_tuple,'o-',label='yc from fct')
plt.plot(t_tuple,yd_tuple,'s-',label='yd from fct')
plt.legend()
plt.show()

