import numpy as np
from .base import DdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9 # 0.9 for solve_ivp but if less error are smalles that in dde23

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.




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

    def __init__(self, fun, t0, y0, h, t_bound, delays, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, h, delays)
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
            # bool for killing discontinuity
            killDiscont = False

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
            else:
                # length to next discontinuity
                len2discont = self.discont[self.nxtDisc] - t
                isCloseToDiscont = 1.1 * h >= len2discont
                #print('len2discount=',len2discont)
                #print('self.nxtDisc=', self.nxtDisc)
                #print('next discont=', self.discont[self.nxtDisc])
                #print('isCloseToDiscont',isCloseToDiscont)
                if(isCloseToDiscont):
                    h = len2discont
                    t_new = self.discont[self.nxtDisc]
                    killDiscont = True # useless no ?
                    self.nxtDisc = self.nxtDisc + 1

            h = t_new - t
            h_abs = np.abs(h)


            y_new, f_new = self.rk_step(t, y, self.f, h)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
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
        """Perform a single Runge-Kutta step.

        This function computes a prediction of an explicit Runge-Kutta method and
        also estimates the error of a less accurate method.

        Notation for Butcher tableau is as in [1]_.

        Parameters
        ----------
        t : float
            Current time.
        y : ndarray, shape (n,)
            Current state.
        f : ndarray, shape (n,)
            Current value of the derivative, i.e., ``fun(x, y)``.
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
            Z = self.delaysEval(t + c * h)
            self.K[s] = self.fun(t + c * h, y + dy, Z)

        y_new = y + h * np.dot(self.K[:-1].T, self.B)
        Z = self.delaysEval(t + h)
        f_new = self.fun(t + h, y_new, Z)

        self.K[-1] = f_new

        return y_new, f_new

class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar and there are two options for ndarray ``y``.
        It can either have shape (n,), then ``fun`` must return array_like with
        shape (n,). Or alternatively it can have shape (n, k), then ``fun``
        must return array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
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
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
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
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
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

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
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
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
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
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
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

class Vern6(RungeKutta):
    """Explicit Runge-Kutta method of order 6(5).
    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 6
    error_estimator_order = 5
    n_stages = 9
    C = np.array([0, 3/50, 1439/15000, 1439/10000,
                  4973/10000, 389/400, 1999/2000, 1, 1 ])
    A = np.zeros((n_stages,n_stages-1))
    A[1,0] =  3/50
    A[2,0] =  519479/27000000
    A[2,1] =  2070721/27000000
    A[3,0] =  1439/40000
    A[3,1] =  0
    A[3,2] =  4317/40000
    A[4,0] =  109225017611/82828840000
    A[4,1] =  0
    A[4,2] = -417627820623/82828840000
    A[4,3] =  43699198143/10353605000
    A[5,0] = -8036815292643907349452552172369/\
            191934985946683241245914401600
    A[5,1] =  0
    A[5,2] =  246134619571490020064824665/\
            1543816496655405117602368
    A[5,3] = -13880495956885686234074067279/\
            113663489566254201783474344
    A[5,4] =  755005057777788994734129/\
            136485922925633667082436
    A[6,0] = -1663299841566102097180506666498880934230261/\
            30558424506156170307020957791311384232000
    A[6,1] =  0
    A[6,2] =  130838124195285491799043628811093033/\
                631862949514135618861563657970240
    A[6,3] = -3287100453856023634160618787153901962873/\
                20724314915376755629135711026851409200
    A[6,4] =  2771826790140332140865242520369241/\
                396438716042723436917079980147600
    A[6,5] = -1799166916139193/96743806114007800
    A[7,0] = -832144750039369683895428386437986853923637763/\
                15222974550069600748763651844667619945204887
    A[7,1] =  0
    A[7,2] =  818622075710363565982285196611368750/\
                    3936576237903728151856072395343129
    A[7,3] = -9818985165491658464841194581385463434793741875/\
                61642597962658994069869370923196463581866011
    A[7,4] =  31796692141848558720425711042548134769375/\
            4530254033500045975557858016006308628092
    A[7,5] = -14064542118843830075/766928748264306853644
    A[7,6] = -1424670304836288125/2782839104764768088217
    A[8,0] =  382735282417/11129397249634
    A[8,1] =  0
    A[8,2] =  0
    A[8,3] =  5535620703125000/21434089949505429
    A[8,4] =  13867056347656250/32943296570459319
    A[8,5] =  626271188750/142160006043
    A[8,6] = -51160788125000/289890548217
    A[8,7] =  163193540017/946795234

    B = np.array([382735282417/11129397249634,
                  0, 0,
                  5535620703125000/21434089949505429,
                  13867056347656250/32943296570459319,
                  626271188750/142160006043,
                  -51160788125000/289890548217,
                  163193540017/946795234,
                  0])
    # get from Verner's website avec bh-b = E 
    E = np.array([12461131651614938103148389/1445036234394733213298413835,
                  0, 0,
                  -21633909117387045317965953125/1113197271463372303940319369579,
                  21633909117387045317965953125/760416658004702652949661077764,
                  -6922850917563854501749105/3281421349740748616670708,
                  173071272939096362543727625/1672856277382934317688463,
                  -74791376208282344108625901/737588957781464692067010,
                  1/30])
    # from Verner site bi5 COEFFICIENTS FOR INTERPOLANT  bi5  WITH  10  STAGES
    P = np.array([[1, 
        -2834058897718490495086218793721472473/533905113718645083237432337636248322,
        2718025628974094767211106485595747024/266952556859322541618716168818124161,
        -2007493102587435133656511668645819580/266952556859322541618716168818124161,
        249346645146318025711596899739877112/266952556859322541618716168818124161,
        199378106425839009650374224000000000/266952556859322541618716168818124161],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 2149739120967678287896284375471359375000/342749026901784884824664927174733230519,
         -5492958105397111152037592078122406250000/342749026901784884824664927174733230519,
         4402390631408885178088267799125656250000/342749026901784884824664927174733230519,
         -56249742645646759461666034659375000000/48964146700254983546380703882104747217,
         -1730712729390963625437276250000000000000/1028247080705354654473994781524199691557],
        [0, 3622473030746576800982284292813464843750/526790867681432602629802992033384351309,
            -38933691634017210049674664360163992187500/1580372603044297807889408976100153053927,
            17495139028182305773126471135900867187500/526790867681432602629802992033384351309,
            -9216003564492900591852706378813281250000/526790867681432602629802992033384351309,
            432678182347740906359319062500000000000/175596955893810867543267664011128117103],
        [0, -80801688121532406876813280779008750/2273257406793442890403455868970073,
            1130047284618441598167544907799477500/6819772220380328671210367606910219,
            -876257846328841227135521923123077500/2273257406793442890403455868970073,
            1005762761452595148951569429250170000/2273257406793442890403455868970073,
            -138457018351277090034982100000000000/757752468931147630134485289656691],
        [0, 8894101767966865321149886325974625000/4635592345813325786648924742878787,
            -128889699381092513087660977440685250000/13906777037439977359946774228636361,
            96690747476972701449592103439602750000/4635592345813325786648924742878787,
            -104976825419006157083194997793935000000/4635592345813325786648924742878787,
            13845701835127709003498210000000000000/1545197448604441928882974914292929],
        [0, -85529300113974351208051144641213185/45420143224168312813776130613122,
            620054801234124026686518242620266725/68130214836252469220664195919683,
            -464947578142702593618050980843076970/22710071612084156406888065306561,
            72055052308090805849478606896950124/3244295944583450915269723615223,
            -598331009666258752869007208000000000/68130214836252469220664195919683],
        [0, 5709918156918632012901/47972509359049343095633,
            -17993572040875704216709/143917528077148029286899,
            85388999974381230343470/47972509359049343095633,
            -223596609894610627617468/47972509359049343095633,
            415486647330808000000000/143917528077148029286899],
        [0, -8, 32, -40, 16, 0]])



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
