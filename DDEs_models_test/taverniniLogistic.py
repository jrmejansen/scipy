from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np

"""
The perturbation of equilibrium in delay logistic equation.
(Example 6 from Shampine 2000, Solving Delay Differential 
Equations with dde23)

Tested features:
    - initial discontinuities/jump
    - restart and management of init discont
"""

r = 3.5
m =  19

def fun(t,y,Z):
    return [ r * y[0] * (1 - Z[:,0]/m)]



tau = 0.74
y0 = [19.001]
h = [19.0]
t0 = 0.0
tf = 40.0
atol = 1e-8
rtol = 1e-8
tspan = [t0, tf]
delays = [tau]


sol = solve_dde(fun, tspan, delays, y0, h, method='RK23',
        atol=atol, rtol=rtol, dense_output=True)
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]

t_m = 5.5
y0_ = sol.sol(t_m)
sol_re = solve_dde(fun, [t_m, tf], delays, y0_, sol, method='RK23', dense_output=True,
        atol=atol, rtol=rtol)

t_re = sol_re.t
y_re = sol_re.y[0,:]
yp_re = sol_re.yp[0,:]

print('err restart vs no restart %s' % (np.abs(y_re[-1]-y[-1])/y[-1]))


plt.figure()
plt.plot(t, y, label='solve_dde')
plt.plot(t_re, y_re, label='solve_dde restart')
plt.xlabel(r'$t$')
plt.xlabel(r'$y(t)$')
plt.legend()
plt.savefig('figures/tavernini/y')

plt.figure()
plt.plot(y, yp, label='I(t)')
plt.legend()
plt.xlabel(r'$y$')
plt.ylabel(r'$yp$')
plt.savefig('figures/tavernini/phase')
plt.show()

