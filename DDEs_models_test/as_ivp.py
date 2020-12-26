from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

tspan = [0, 100]
y0 = [0, 10]
atol = 1e-8
rtol = 1e-5
def upward_cannon(t, y): return [y[1], -0.5]

def hit_ground(t, y): return y[0]
hit_ground.terminal = True
hit_ground.direction = -1

sol_ivp = solve_ivp(upward_cannon, tspan, y0, method='RK23', events=hit_ground, atol=atol, rtol=rtol)

print(sol_ivp.t_events)

t_ivp = sol_ivp.t
y_0_ivp = sol_ivp.y[0,:]
y_1_ivp = sol_ivp.y[1,:]

def fun(t,y,Z):
    return [y[1], -0.5]
def ev(t, y, Z): return y[0]
ev.terminal = True
ev.direction = -1

delays = []


sol = solve_dde(fun, tspan, delays, y0, y0,
                    method='RK23', events=ev, dense_output=True,
                    atol=atol, rtol=rtol)
t = sol.t
y_0 = sol.y[0,:]
y_1 = sol.y[1,:]

t_ = np.linspace(0, t[-1])
y1_interp = sol.sol(t_)[1,:]
y0_interp = sol.sol(t_)[0,:]
plt.figure()
plt.plot(t_ivp, y_1_ivp, 'o', label='solve_ivp y1')
plt.plot(t, y_1, 'k', label='solve_dde y1')
plt.plot(t_ivp, y_0_ivp, 'o', label='solve_ivp y0')
plt.plot(t, y_0, '-', label='solve_dde y0')
plt.plot(t_, y0_interp, 'o-', label='solve_dde y0 with denseoutput')
plt.plot(t_, y1_interp, 'o-', label='solve_dde y1 with denseoutput')
plt.xlabel(r'$t$')
plt.xlabel(r'$y(t)$')
plt.legend()
plt.savefig('figures/as_ivp/y')

plt.show()
