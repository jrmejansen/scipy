import numpy as np
from .base import DdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.8
# 0.9 for solve_ivp but as DDEs are favorable to cyclic solutions whith some
# possible strong variations in short amount of time, this value is put to 0.8

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10.  # Maximum allowed increase in a step size.




class RungeKutta(DdeSolver):
    """Base class for explicit Runge-Kutta methods."""
    C = NotImplemented
    A = NotImplemented
    B = NotImplemented
    E = NotImplemented
    P = NotImplemented
    order = NotImplemented
    error_estimator_order = NotImplemented
    n_stages = NotImplemented

    def __init__(self, fun, t0, y0, h, t_bound, delays, jumps,
                 tracked_stages, max_step=np.inf, rtol=1e-3, atol=1e-6,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, h, delays,
                                            jumps, tracked_stages)
        self.y_old = None
        self.yp = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, self.Z0, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            # bool to locate next discont and adapt time step
            isCloseToDiscont = False
            discontWillCome = False

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h
            # secondary stepsize controls
            # addition of killing discontinuity feature compared to _ivp code
            if(self.nxtDisc < len(self.discont)):
                # we deacrese the max_factor when tracking discontinuities 
                # as discontinuities .....
                # length to next discontinuity
                len2discont = self.discont[self.nxtDisc] - t
                isCloseToDiscont = 1.1 * h >= len2discont
                # if close enough modification of the t_new to kill it
                if isCloseToDiscont:
                    h = len2discont
                    t_new = self.discont[self.nxtDisc]

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = self.rk_step(t, y, self.f, h)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1.0: # accept step
                if(isCloseToDiscont):
                    # update self.nxtDisc and put factor to 1.0
                    self.nxtDisc += 1
                if np.isclose(error_norm,0.0):
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else: # reject step
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True
                self.nfailed += 1

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new
        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)

    def rk_step(self, t, y, f, h):
        """Perform a Runge-Kutta step. The rk_step method of _ivp were added 
        within the RungeKutta class as an instance method because during DDEs
        integration process, we need to access to the eval_Z method.

        This function computes a prediction of an explicit Runge-Kutta method 
        and also estimates the error of a less accurate method.

        Notation for Butcher tableau is as in [1]_.

        Parameters
        ----------
        t : float
            Current time.
        y : ndarray, shape (n,)
            Current state.
        f : ndarray, shape (n,)
            Current value of the derivative, i.e., ``fun(t, y, Z)``.
        h : float
            Step to use.
        Returns
        -------
        y_new : ndarray, shape (n,)
            Solution at t + h computed with a higher accuracy.
        f_new : ndarray, shape (n,)
            Derivative ``fun(t + h, y_new)``.

        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
        """
        self.K[0] = f
        for s, (a, c) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
            dy = np.dot(self.K[:s].T, a[:s]) * h
            # eval Z at c * h 
            Z = self.eval_Z(t + c * h)
            self.K[s] = self.fun(t + c * h, y + dy, Z)

        y_new = y + h * np.dot(self.K[:-1].T, self.B)
        # last eval of Z for the step
        Z = self.eval_Z(t + h)
        f_new = self.fun(t + h, y_new, Z)

        self.K[-1] = f_new

        return y_new, f_new

class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2) for DDEs resolution as 
    describe in [2]_. 
    
    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.
    The 3(2) pair has the interesting property that the continuous extension, 
    used for evaluation of delayed states, and the formula for integration are 
    both third-order accurate formula.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y, Z)``.
        Here ``t`` is a scalar, ``y`` is the current state and ``Z[:,i]`` the
        state of ``y`` evaluate at time ``$t-\tau_i$`` for $\tau_i=delays[i]$.

    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here, `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    h : callable or float or tuple or DdeResult 
        history function
    h_info : 
        type of history given by user
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    f : ndarray
        Current right hand side
    Z0 : ndarray
        evaluation of Z at initial time.
    t0 : float
        Initial time.
    delays : list
        list of delays
    Ndelays : int
        number of delays
    delayMax : float
        maximal delay
    delayMin : float
        minimal delay
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    order_track : int
        bl
    init_discont : bool
        is there or not an initial discontinuity at initial time
    nxtDisc : int
        next discontinuity to be kill
    discont : ndarray (nbr_discontinuities,)
        times where discontinuities will be killed
    nfev : int
        Number of the system's rhs evaluations.
    nfailed : int
        Number of rejected evaluations.
    nOverlap : int
        Number of overlapping evaluations of Z

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [2] L.F. Shampine and S. Thompson, "Solving DDEs in Matlab", 
            Applied Numerical Mathematics Vol. 37, No. 4. pp. 441-458, 2001.
    """
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1/2, 3/4])
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = np.array([2/9, 1/3, 4/9])
    E = np.array([5/72, -1/12, -1/9, 1/8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2/3],
                  [0, 4/3, -8/9],
                  [0, -1, 1]])


class RK45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.
    
    The 5(4) pair is to be used with care. Integration is made with a 
    fifth-order accurate formula although the continous extension,
    used for evaluation of delayed states, is a fourth-order
    accuracy. As evaluation of delayed terms Z is at a lower order accuracy
    than integration we can not guarante the preservatin of the global order 
    of the DDE method. This can undermine stepsize control strategies.
    For the RK23 pair, the order of interpolation and integration are the same
    [3]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y, Z)``.
        Here ``t`` is a scalar, ``y`` is the current state and ``Z[:,i]`` the
        state of ``y`` evaluate at time ``$t-\tau_i$`` for $\tau_i=delays[i]$.

    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here, `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    h : callable or float or tuple or DdeResult 
        history function
    h_info : 
        type of history given by user
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    f : ndarray
        Current right hand side
    Z0 : ndarray
        evaluation of Z at initial time.
    t0 : float
        Initial time.
    delays : list
        list of delays
    Ndelays : int
        number of delays
    delayMax : float
        maximal delay
    delayMin : float
        minimal delay
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    order_track : int
        bl
    init_discont : bool
        is there or not an initial discontinuity at initial time
    nxtDisc : int
        next discontinuity to be kill
    discont : ndarray (nbr_discontinuities,)
        times where discontinuities will be killed
    nfev : int
        Number of the system's rhs evaluations.
    nfailed : int
        Number of rejected evaluations.
    nOverlap : int
        Number of overlapping evaluations of Z

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] L.F. Shampine and S. Thompson, "Solving DDEs in Matlab", 
            Applied Numerical Mathematics Vol. 37, No. 4. pp. 441-458, 2001.
    """
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])



class RkDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super(RkDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        y = self.h * np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y
