import sys
import numpy as np
from inspect import isfunction
from scipy.interpolate import CubicHermiteSpline
from .common import EPS

def check_arguments(fun, y0, h):
    """Helper function for checking arguments for solve_dde.
    Complex problem avoid. The function define the type of history given by the user 
    see init_history_function
    """
    y0 = np.asarray(y0)
    if np.issubdtype(y0.dtype, np.complexfloating):
        raise ValueError("`y0` is complex, but the chosen solver does "
                             "not support integration in a complex domain.")
    else:
        dtype = float
    y0 = y0.astype(dtype, copy=False)
    if(type(h) is tuple):
        h_info = 'from tuple'
        h_n = h
    elif(isfunction(h)):
        h_info = 'from function'
        def h_n(t):
            return np.asarray(h(t),dtype=dtype)
        test = h_n(0.0)
        if(test.shape != y0.shape):
            raise ValueError("size of output of history function 'h'\
                              is not compatible with size of y0")
    elif(h.__class__.__name__ == 'DdeResult'):
        h_info = 'from previous simu'
        h_n = h
    else:
        h_info = 'from constant'
        h_n = np.asarray(h)
        h_n = h_n.astype(dtype, copy=False)
        if(h_n.shape != y0.shape):
            raise ValueError("size of output of history function 'h'\
                              is not compatible with size of y0")
    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    def fun_wrapped(t, y, Z):
        return np.asarray(fun(t, y, Z), dtype=dtype)
    
    return fun_wrapped, y0, h_n, h_info



class DdeSolver(object):
    """Base class for DDE solvers.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y, Z)``.
        Here ``t`` is a scalar, there are two options for ndarray ``y`` and
        ``h`` the history function.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    h :
    t_bound : float
        Boundary time --- the integration won't continue beyond it. It also
        determines the direction of the integration.
    delays :

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    h : history function
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of the system's rhs evaluations.
    njev : int
        Number of the Jacobian evaluations.
    nlu : int
        Number of LU decompositions.
    """
    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, t_bound, h,
                 delays):
        self._fun, self.y0, self.h, self.h_info = check_arguments(fun,y0,h)
        self.y = self.y0.copy()
        self.t_old = None
        self.t0 = t0
        self.t = t0
        self.t_bound = t_bound
        self.n = self.y.size
        if not delays:
            self.delayMin = np.inf
        else:
            self.delays = sorted(delays)
            self.Ndelays = len(delays)
            self.delayMin = min(self.delays)
            self.delayMax = max(self.delays)
        # check if negative delays
        if(self.delayMin < 0.0):
            raise ValueError("delay min has negative value")

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.status = 'running'

        if(self.h_info != 'from previous simu'):
            # if first computation init of stat params
            self.nfev = 0
            self.njev = 0
            self.nlu = 0
            self.nfailed = 0
            self.nOverlap = 0
        else:
            # restart from previous computation -> recovering previous stats
            self.nfev = self.h.nfev
            self.njev = self.h.njev
            self.nlu = self.h.nlu
            self.nfailed = self.h.nfailed
            self.nOverlap = self.h.nOverlap

            self.before = False # default value
            if(self.h.t[0] <= self.t and self.t < self.h.t[-1]):
                self.before = True
                print('t0 given in tspan is %s <= %s < %s \n \
                        restart from previous computation \
                        ' % (self.h.t[0], self.t, self.h.t[-1]))
            elif(self.h.t[0] > self.t or self.t > self.h.t[-1]):
                raise ValueError("error tspan t0 > or < self.h.t[0]")

        # init of the history function
        self.init_history_function()

        fun_single = self._fun
        def fun(t, y, Z):
            self.nfev += 1
            return self.fun_single(t, y, Z)

        self.fun = fun
        self.fun_single = fun_single
        # init y(t-tau_i) at t0
        self.Z0 = self.eval_y_past(self.t)
        # evaluate y0 from history function
        self.y0_h = self.eval_y0_from_h(self.t)
        # check if there is a discontinuity of order 0, i.e. if 
        # self.y0_h != self.y
        if(np.max(np.abs(self.y0_h - self.y)) < EPS):
            self.order_track = self.order
            # print('continuous at t=t0')
            self.init_discont = False
        else:
            self.init_discont = True
            print('*********')
            print('Note: discontinuity of order 0 at initial time')
            print('*********')
            self.order_track = self.order + 1

        self.discontDetection()
        # print('self.discont', self.discont)
        
        self.f = self.fun(self.t, self.y, self.Z0) # initial value of f(t0,y0,Z0)

    def discontDetection(self):
        """Discontinuity detection between t0 and tf.
    
        Parameters
            t0 (): initial time
            tf (): final time
            delays (list): list of delays
        ----------
        Returns
        -------
        nxtDisc : (int)
            index of the nearst discontinuity
        discont : ndarray, shape (nbr_discontinuities,)
            array with all discont within my interval of integration
    
        References
        ----------
        .. [1] S. Shampine, Thompson, "?????" dde23 MATLAB
        """
    
        self.discont = []
        #  discontinuites detection
        if not self.delays:
            self.discont = self.t_bound
            self.delayMin = np.inf
        else:
    
            d = np.asarray(self.delays)
            # print('d', d)
            tmp = self.t + d
            # print('order+1', self.order_track+1)
            for i in range(1,self.order_track+1):
                self.discont.append(tmp.tolist())
                # print('tmp', tmp)
                z = 1 # number of time for delays
                # print('i+1',i+1, 'self.order_track+1', self.order_track+1)
                for j in range(i+1,self.order_track+1): # get intermediere discont
                    # print('j', j)
                    # print('len(d)',len(d))
                    for k in range(1,len(d)):
                        # print('k',k)
                        # print('tmp[k:]', tmp[k:])
                        # print('d[:-k]', d[:-k])
                        # print('z', z)
                        inter_d = tmp[k:] + d[:-k] * z
                        # print('inter_d', inter_d)
                        self.discont.append(inter_d.tolist())
                    z += 1
                tmp += d
            # flatened the list of list of discont and discont as array
            self.discont = np.asarray(
                                     sorted([val for sub_d in self.discont
                                                 for val in sub_d])
                                     )
            self.discont = np.delete(self.discont,
                                     np.argwhere(
                                                 np.ediff1d(self.discont) < EPS
                                                 ) + 1)
            self.nxtDisc = 0  # indice diiie la prochain discontinuite

    def init_history_function(self):
        """
        initialisation of past values according to the type of history function given by the user
        i.e. history as a function / tuple / constant / previous computation
        """
        if( self.h_info == 'from tuple'):
            (self.t_past, self.y_past, self.yp_past) = self.h
            self.t_oldest = self.t_past[0]
            if(self.t_oldest < (self.t0 - self.delayMax)):
                    raise("history tuple give in history not enough to describe\
                            all past values")
            self.y_oldest = self.y_past[:,0]
            self.yp_oldest = self.yp_past[:,0]
            self.h = []
            for k in range(self.n):
                # extrapolation not possible
                p = CubicHermiteSpline(self.t_past, self.y_past[k,:],
                                       self.yp_past[k,:], extrapolate=False)
                self.h.append(p)
        elif(self.h_info == 'from function'):
            self.t_oldest = self.t0-self.delayMax
            self.t_past = [self.t_oldest, self.t0]
            self.y_oldest = self.h(self.t_oldest)
            self.yp_oldest = np.zeros(self.y_oldest.shape)
        elif(self.h_info == 'from constant'):
            self.t_oldest = self.t0-self.delayMax
            self.y_oldest = self.h
            self.yp_oldest = np.zeros(self.h.shape)
            self.t_past = [self.t_oldest, self.t0]
        elif(self.h_info == 'from previous simu'):
            if(self.before):
                # reorganisation of the interpolation function
                self.h.sol.reorganize(self.t)
                self.solver_old = self.h
                self.h = self.solver_old.sol
            else:
                self.solver_old = self.h
                self.h = self.solver_old.sol
        else:
            print('h_info',self.h_info)
            raise ValueError("wrong initialisation of the dde history")


    def eval_y_past(self, t_eval):
        """ Z[:,i] is the evaluation of the solution a past time:  
            Z[:,i] = y(t-delays[i]).
            from the value of t-delays[i], Z[:,i] can by evaluate by several ways
        """
        if not self.delays:
            Z = None
            return Z

        Z = np.zeros((self.n,self.Ndelays))
        t_past = t_eval - np.asarray(self.delays)

        for k in range(self.Ndelays):
            t_past_k = t_past[k]
            if(t_past_k < self.t0):
                # case where we can not use continuous extension
                if(self.h_info == 'from function'):
                    Z_k = self.h(t_past_k)
                elif(self.h_info == 'from tuple'):
                    Z_k = np.zeros(self.n)
                    for i in range(self.n):
                        Z_k[i] = self.h[i](t_past_k)
                        if(np.isnan(Z_k[i])):
                            raise ValueError("NaN value found in eval_y_past")
                elif(self.h_info == 'from previous simu'):
                    Z_k = self.h(t_past_k)
                elif(self.h_info == 'from constant'):
                    Z_k = self.h
            elif(np.abs(t_past_k-self.t0) < EPS):
                Z_k = self.y0
            else:
                if(t_past_k <= self.t):
                    # use of continous extansion of RK method
                    Z_k = self.CE(t_past_k)
                else:
                    # overlapping
                    sol = self.dense_output()
                    Z_k = sol(t_past_k)
                    self.nOverlap += 1
            Z[:,k] = Z_k
        return Z

    def eval_y0_from_h(self, t_eval):
        """
        evaluation of y at time t_eval<=self.t0 i.e. less than current time step
        """
        Y0 = np.zeros(self.n)
        if(t_eval <= self.t0):
            # case where we can not use continuous extension
            if(self.h_info == 'from function'):
                Y0 = self.h(t_eval)
            elif(self.h_info == 'from tuple'):
                Y0 = np.zeros(self.n)
                for i in range(self.n):
                    Y0[i] = self.h[i](t_eval)
                    if(np.isnan(Y0[i])):
                        raise ValueError("NaN value found in eval_y_past")
            elif(self.h_info == 'from previous simu'):
                Y0 = self.h(t_eval)
            elif(self.h_info == 'from constant'):
                Y0 = self.h
        else:
            raise ValueError("can not eval time")
        return Y0

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = 'failed'
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = 'finished'

        return message

    def dense_output(self):
        """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        if self.t_old is None:
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")

        if self.n == 0 or self.t == self.t_old:
            # Handle corner cases of empty solver and no integration.
            return ConstantDenseOutput(self.t_old, self.t, self.y)
        else:
            return self._dense_output_impl()

    def _step_impl(self):
        raise NotImplementedError

    def _dense_output_impl(self):
        raise NotImplementedError


class DenseOutput(object):
    """Base class for local interpolant over step made by an ODE solver.

    It interpolates between `t_min` and `t_max` (see Attributes below).
    Evaluation outside this interval is not forbidden, but the accuracy is not
    guaranteed.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, t_old, t):
        self.t_old = t_old
        self.t = t
        self.t_min = min(t, t_old)
        self.t_max = max(t, t_old)

    def __call__(self, t):
        """Evaluate the interpolant.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate the solution at.

        Returns
        -------
        y : ndarray, shape (n,) or (n, n_points)
            Computed values. Shape depends on whether `t` was a scalar or a
            1-D array.
        """
        t = np.asarray(t)
        if t.ndim > 1:
            raise ValueError("`t` must be a float or a 1-D array.")
        return self._call_impl(t)

    def _call_impl(self, t):
        raise NotImplementedError


class ConstantDenseOutput(DenseOutput):
    """Constant value interpolator.

    This class used for degenerate integration cases: equal integration limits
    or a system with 0 equations.
    """
    def __init__(self, t_old, t, value):
        super(ConstantDenseOutput, self).__init__(t_old, t)
        self.value = value

    def _call_impl(self, t):
        if t.ndim == 0:
            return self.value
        else:
            ret = np.empty((self.value.shape[0], t.shape[0]))
            ret[:] = self.value[:, None]
            return ret
