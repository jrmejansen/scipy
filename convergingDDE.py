import scipy
import time
from scipy.integrate._dde.dde import solve_dde
import matplotlib.pyplot as plt
from sympy import *
import numpy as np

############################################################
#### analytical solution with sympy
############################################################
def fct_sy_interval(past,tf,tau_num):
    t_ = Symbol('t_')
    t = Symbol('t')
    tau = Symbol('tau')
    interval = int(tF//tau_num) + 1
    xP = past

    x1 = xP -  integrate(xP, (t_, 0, t))
    fct_sy = []
    fct_sy.append(xP)
    fct_sy.append(x1)

    for k in range(2,interval+1):
        x_interv = fct_sy[k-1].subs(t,(k-1)*tau) -  integrate(fct_sy[k-1].subs(t,t-tau).subs(t,t_) , (t_, (k-1)*tau, t))
        x_interv = x_interv.subs(tau,tau_num)

        fct_sy.append(x_interv)

    fct_np = []
    for k in range(len(fct_sy)):
        f_np = lambdify(t, fct_sy[k], 'numpy')
        fct_np.append(f_np)

    return fct_np

def methodOfStep(tau,discont):
    t= symbols("t")
    y =  Function("y")
    dydt = y(t).diff(t)
    phi = 1

    yMoS = []
    yMoS.append(phi)


    for i in range(len(discont)):
        print('i',i)
        if(i==0):
            if(isinstance(phi, int)):
                pastExpr = yMoS[i]
            else:
                pastExpr = yMoS[i].subs(t,t-tau)
        else:
                pastExpr = yMoS[i].subs(t,t-tau)


        print('pastExpr',pastExpr)
        eq = Eq( dydt , - pastExpr)
        print('eq',eq)
        if(i==0):
            if(isinstance(phi, int)):
                ic = yMoS[i]
            else:
                ic = yMoS[i].subs(t,discont[i])
        else:
                ic = yMoS[i].subs(t,discont[i])
        t_bord = discont[i]
        print('t_bord',t_bord,'ic',ic)
        res = dsolve( eq, ics={y(t_bord): ic } )
        print('res',res,'\n')
        yMoS.append(res.rhs)

    fct_np = []
    for k in range(len(yMoS)):
        f_np = lambdify(t, yMoS[k], 'numpy')
        fct_np.append(f_np)

    return fct_np

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

############################################################



def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ - y_tau ]

t1 = time.time()
tau = 1
y0 = [1.]
t0 = 0.0
tf = 6.0
atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]

def h(t):
    return [1]

sol = solve_dde(fun, tspan, delays, y0, h, method='RK23', atol=atol, rtol=rtol)
t2 = time.time()
print('Elapsed time is %s seconds' % (t2-t1))
t = sol.t
y = sol.y[0,:]
yp = sol.yp[0,:]


### sol matlab
import scipy.io as spio
path_matlab = '/home/jjansen/Bureau/These/recherche/matlab/dde_benchmark/solConverging_dde23.mat'
mat = spio.loadmat(path_matlab, squeeze_me=True)

t_mat = mat['t']
y_mat = mat['y']
yp_mat = mat['yp']

### sol DDEMOI
path_ddeMoi = '/home/jjansen/Bureau/These/recherche/python/DDEmoi/benchmark/data/solConv'
t_ddeMoi = np.load('%s/t_ddeRK.npy' % path_ddeMoi)
dt_ddeMoi = np.load('%s/dt_ddeRK.npy' % path_ddeMoi)
y_ddeMoi = np.load('%s/y_ddeRK.npy' % path_ddeMoi)

# sol analytique
fct_np = methodOfStep(tau,sol.discont)

ana_spdev = expresion(t,fct_np)
ana_ddeMoi = expresion(t_ddeMoi,fct_np)
ana_mat = expresion(t_mat,fct_np)

err_spdev = np.abs(np.abs(y - ana_spdev) / ana_spdev)
err_ddeMoi = np.abs(np.abs(y_ddeMoi - ana_ddeMoi) / ana_ddeMoi)
err_mat = np.abs(np.abs(y_mat - ana_mat) / ana_mat)



plt.figure(figsize=(18,14))
plt.plot(t, y, label='scipy-dev y(t)')
plt.plot(t, yp, label="scipy-dev y'(t)")
plt.plot(t_mat, y_mat, 'o', label='matlab y(t)')
plt.plot(t_mat, yp_mat, 'o', label="matlab y'(t)")
plt.legend()
plt.savefig('testFigure/solConv/y')

plt.figure(figsize=(18,14))
plt.plot(t, err_spdev, label='scipy-dev err')
#plt.plot(t_ddeMoi, err_ddeMoi, label='ddeMoi err')
plt.plot(t_mat, err_mat, label="matlab err")
plt.legend()
plt.savefig('testFigure/solConv/error')

plt.figure(figsize=(18,14))
plt.plot(y, yp, label='scipy-dev')
plt.plot(y_mat, yp_mat, 'o', label='matlab')
plt.xlabel('y')
plt.ylabel('dydt')
plt.title('phase graph')

plt.figure(figsize=(18,14))
plt.plot(t[:-1],np.diff(t),'-o',label='dt scipy-dev')
#plt.plot(t_ddeMoi[:-1],np.diff(t_ddeMoi),'-o',label='dt ddeMOI')
plt.plot(t_mat[:-1],np.diff(t_mat),'-o',label='dt matlab solver')
plt.legend()
plt.savefig('testFigure/solConv/dt')

plt.show()






