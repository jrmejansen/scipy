import dill
dill.settings['recurse'] = True
from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np

"""
Converging problem
    - Equation : y'(t) = -y(t-1)

Comparison with the analytical solution at t [0,10]. The 
influence of bad management of the tracking of propagative discontinuities 
for the RK23 pair is not significant.
"""
# analytic sol
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

def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ - y_tau ]

tau = 1
y0 = [1.]
t0 = 0.0
tf = 10.0
atol = 1e-8
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]

def h(t):
    return [1]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig_dt, ax_dt = plt.subplots()
ax1.set_title("RK23 pair, sensitivity of the nbr of steps for tracking discont")
ax2.set_title("RK23 pair, sensitivity of the nbr of steps for tracking discont")

err_t10 = []
ii = [i for i in range(6)]
for i in ii:
    sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', tracked_stages=i,
        atol=atol, rtol=rtol)
    print('i = %s sol.discont = %s' % (i, sol.discont))
    t = sol.t
    y = sol.y[0,:]
    yp = sol.yp[0,:]

    ana = expresion(t,fct_np)

    eps = 1e-2
    mask = np.abs(y) > eps
    err = np.abs(np.abs(y[mask] - ana[mask]) / ana[mask])
    err_t10.append(err[-1])
    
    ax1.plot(t[mask], err, label='%s stages discont = %s' % (i, sol.discont))
    ax_dt.plot(t[1:], np.diff(t), 'o-', label='%s stages' % i)


ax2.plot(ii, err_t10, 'o')

ax1.legend()
ax2.legend()

ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$\varepsilon$')
ax1.set_yscale('log')

ax2.set_xlabel(r'$i$ nbr of tracking stages')
ax2.set_ylabel(r'$\varepsilon(10)$')
ax2.set_yscale('log')

ax_dt.set_xlabel(r'$t$')
ax_dt.set_ylabel(r'$\Delta t$')
ax_dt.set_yscale('log')

fig1.savefig('figures/solConv/RK23error_trackingDiscontStages_t')
fig2.savefig('figures/solConv/RK23error_trackingDiscontStages_t10')
fig_dt.savefig('figures/solConv/RK23error_trackingDiscontStages_dt')
plt.show()
