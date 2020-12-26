from scipy.integrate import solve_dde
import matplotlib.pyplot as plt
import numpy as np


def fun(t,y,Z):
    if y[0] >= 0.0:
        return [-1.0]
    else:
        return [-10.0]

y0 = 1.0

jumps = [1.]
tf = 2.0
tspan = [0.0, tf]

delays = []

rtol = 1e-5
atol = 1e-10
sol23 = solve_dde(fun, tspan, delays, [y0], [y0],
                    method='RK23', atol=atol, rtol=rtol)
sol23_j = solve_dde(fun, tspan, delays, [y0], [y0], #tracked_stages=0,
                    method='RK23', jumps=jumps, atol=atol, rtol=rtol)

print('nfev of f without jumps option : %s' % (sol23.nfev))
print('nfev of f wit     jumps option : %s' % (sol23_j.nfev))
print('nfaild without jumps option : %s' % (sol23.nfailed))
print('nfaild with    jumps option : %s' % (sol23_j.nfailed))

def anaf(t):
    if t <= 1.:
        return 1. - t
    else:
        return -10. * (t - 1.)

t_j = sol23_j.t
y_j = sol23_j.y[0,:]
t_ = sol23.t
y_ = sol23.y[0,:]

ana_j = np.zeros(y_j.shape)
ana_ = np.zeros(y_.shape)

for i in range(len(t_)):
    ana_[i] = anaf(t_[i])
for i in range(len(t_j)):
    ana_j[i] = anaf(t_j[i])

eps = 1e-2
mask_ = np.abs(y_) > eps
mask_j = np.abs(y_j) > eps


err_j = np.abs(np.abs(ana_j[mask_j]-y_j[mask_j])/ana_j[mask_j])
err_ = np.abs(np.abs(ana_[mask_]-y_[mask_])/ana_[mask_])

plt.figure()
plt.plot(t_[mask_], err_, 'o-', label='with jump')
plt.plot(t_j[mask_j], err_j, 'o-', label="without jump")
plt.yscale('log')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\varepsilon$')
plt.savefig('figures/discODEex1/error')


plt.figure()
plt.plot(t_, ana_, 'o-', label='ana')
plt.plot(t_, y_, 'o-', label='no jump')
plt.plot(t_j, y_j, 'o-', label=' jump')
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.savefig('figures/discODEex1/y')


plt.figure()
plt.plot(t_[0:-1], np.diff(t_), 'o', label='dt')
plt.plot(t_j[0:-1], np.diff(t_j), 'o', label='dt jump')
plt.yscale('log')
plt.legend()
plt.xlabel("t")
plt.ylabel("dt")
plt.legend()
plt.savefig('figures/discODEex1/dt')

plt.show()

