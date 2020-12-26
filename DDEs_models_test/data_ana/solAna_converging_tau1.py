import dill
dill.settings['recurse'] = True

from sympy import *
import numpy as np

def methodOfStep(tau,disconti,y0):
    """
    tau a single delay
    discont list of discontinuity
    y0 initial condition can be int of in sympy expression
    """
    t= symbols("t")
    y =  Function("y")
    dydt = y(t).diff(t)
    phi = y0

    yMoS = []
    yMoS.append(phi)

    for i in range(len(discont)):
        print('i',i)
        if(i==0):
            if(isinstance(phi, (int, float))): # in phi is cst
                pastExpr = yMoS[i]
            else:  # phi is a sympy variable
                pastExpr = yMoS[i].subs(t,t-tau)
        else:
                pastExpr = yMoS[i].subs(t,t-tau)

        eq = Eq( dydt , - pastExpr)
        if(i==0):    # definition on condition of continuity
            if(isinstance(phi, (int, float))):
                ic = yMoS[i]
            else:
                ic = yMoS[i].subs(t,discont[i])
        else:
                ic = yMoS[i].subs(t,discont[i])
        t_bord = discont[i]
        res = dsolve( eq, ics={y(t_bord): ic } )
        yMoS.append(res.rhs)
    # transformation of sympy expression to list of lambdify 
    fct_np = []
    for k in range(len(yMoS)):
        f_np = lambdify(t, yMoS[k], 'numpy')
        fct_np.append(f_np)

    return fct_np


############################################################



def fun(t,y,Z):
    y_tau = Z[:,0]
    return [ + y_tau ]


tau = 1
y0 = [1.]
t0 = 0
tf = 10
atol = 1e-10
rtol = 1e-5
tspan = [t0, tf]
delays = [tau]


# sol analytique
discont = [k for k in range(0,tf+1)]
fct_np = methodOfStep(tau,discont,y0[0])


dill.dump(fct_np, open("converging_tf6_tau1.pkl", "wb"))

