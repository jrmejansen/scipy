import numpy as np
from inspect import isfunction

def check_arguments(fun, y0, h):
    """Helper function for checking arguments common to all solvers."""
    y0 = np.asarray(y0)
    if not np.issubdtype(y0.dtype, np.complexfloating):
        raise ValueError("`y0` is complex, but the chosen solver does "
                             "not support integration in a complex domain.")
    else:
        dtype = float
    y0 = y0.astype(dtype, copy=False)

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    def fun_wrapped(t, y, h):
        return np.asarray(fun(t, y, h), dtype=dtype)

    return fun_wrapped, y0


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
        self.t_old = None
        self.h = h
        self.t = t0
        self.delays = delays
        self.Ndelays = len(delays)
        self.delayMin = min(self.delays)
        self.delayMax = max(self.delays)

        (self.nxtDisc, self.discont) = discontinuityDetection(t0,t_bound)
        self.init_history(h)

        self.Z0 = self.delaysEval(self.t)
        
        self._fun, self.y = check_arguments(fun, y0, self.Z0)
        self.f = self.fun(self.t, self.y, Z0) # initial value of f(t0,y0,Z0)

        self.t_bound = t_bound

        fun_single = self._fun

        def fun(t, y, Z):
            self.nfev += 1
            return self.fun_single(t, y, Z)

        self.fun = fun
        self.fun_single = fun_single

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size
        self.status = 'running'

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def init_history(self):
        """
        """
        if(type(self.h) is tuple):
            self.h_info = 'from tuple'
            (self.t_past, self.y_past, self.yp_past) = h
        elif(isfunction(h)):
            self.h_info = 'from function'
            self.h_fun = h
            self.t_past = np.linspace(self.t0-self.delayMax, self.t0 , 100)
            self.y_past = self.h_fun(self.t_past)
            self.yp_past = np.zeros(self.y_past.shape)
        else:
            self.h_info = 'from constant'
            self.t_past = np.linspace(self.t0-self.delayMax, self.t0 , 100)
            self.y_past = np.ones(self.t_past.shape) * h
            self.yp_past = np.zeros(self.y_past.shape)

    def delaysEval(self,t):
        t_delays = t - np.asarray(self.delays)
        for k in range(self.Ndelays):
            t_delay_k = t_delays[k]
            if(t_delay_k < self.t0):  
                # case where we can not use continuous extension
                if(self.h_info == 'from function'):
                    Z_k = self.h_fun(t_delay_k)
                elif(self.h_info == 'from tuple'):
                    Z_k = hermiteInterp(k,t_delay_k, self.t_past,
                                        self.y_past, self.yp_past)
                else:
                    Z_k = self.y_past[0]
            else:
                # use of continous extansion of RK method
                print("note implemented")
                if(t_delay_k < self.t):
                    print('on peut faire du RKCE')
                else:
                    # overlapping
                    print('extrapolation')
                    Z_k = self.denseOutputFormula(t_delay_k)
            Z[:,k] = Z_k
        return Z

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
