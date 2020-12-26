import sys
from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix
from inspect import isfunction

EPS = np.finfo(float).eps


def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > np.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def validate_max_step(max_step):
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


def warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    The initializer of each solver class is expected to collect keyword
    arguments that it doesn't understand and warn about them. This function
    prints a warning for each key in the supplied dictionary.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn("The following arguments have no effect for a chosen solver: {}."
             .format(", ".join("`{}`".format(x) for x in extraneous)))


def validate_tol(rtol, atol, n):
    """Validate tolerance values."""
    if rtol < 100 * EPS:
        warn("`rtol` is too low, setting to {}".format(100 * EPS))
        rtol = 100 * EPS

    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError("`atol` has wrong shape.")

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


def norm(x):
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size ** 0.5


def select_initial_step(fun, t0, y0, Z0, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    Z0 : ndarray, shape (n,Ndelays)
        Initial values of delayed variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1, Z0)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


class ContinuousExt(object):
    """Continuous extension of the solution.

    It is organized as a collection of `DenseOutput` objects which represent
    local interpolants. It provides an algorithm to select a right interpolant
    for each given point.

    The interpolants cover the range between `t_min` and `t_max` (see
    Attributes below).

    ***************
    Evaluation outside this interval have to be forbidden, but
    not yet implemented
    *********

    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.

    Parameters
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of history and DenseOutput with respectively 1 and
        n_segments-1 elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.
    ys : array_like, shape (n_segments + 1,)
        solution associated to ts. Needed to get solution values when
        discontinuities occur
    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    repeated_t : bool
        Is here repeated time values in ts which translate discontinuities.
    t_discont : list
        Times where there are discontinuies.
    y_discont : list
        Values at discontinuities.
    n_segments : int
        Number of interpolant.
    n : int
        Number of equations.
    """

    def __init__(self, ts, interpolants, ys):
        self.repeated_t = False
        if np.any(np.ediff1d(ts) < EPS):
            # where is at least 2 repeated time in ts
            self.repeated_t = True
            # locate duplicate values
            # print('type(ts)', type(ts))
            idxs = np.argwhere(np.diff(ts) < EPS) + 1
            idxs = idxs[:,0].tolist()
            # save discont values
            self.t_discont = [ts[i] for i in idxs]
            self.y_discont = [ys[i] for i in idxs]
            # remove discont values and times from ts and ys
            ts = [i for j, i in enumerate(ts) if j not in idxs]
            ys =[i for j, i in enumerate(ys) if j not in idxs]
        else:
            # no discontinuities to take care of
            self.t_discont = []
            self.y_discont = []

        ts = np.asarray(ts)
        d = np.diff(ts)

        if not ((ts.size == 2 and ts[0] == ts[-1]) or
                np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        self.n = len(ys[0])

        # condToReapeted = len(ts) != len(ys) and self.repeated_t

        if ts.shape != (self.n_segments + 1,) or len(ts) != len(ys):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")
        if len(ts) != len(ys):
            raise ValueError("number of ys and ts don't match.")

        print('ContinousExt len(ts)', len(ts), 'len interp', len(interpolants))
        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.ts_sorted = ts[::-1]

    def _call_single(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # then we prioritize a segment with a lower index.

        # if discont case + t is a discont
        if self.repeated_t and np.any(np.abs(self.t_discont - t) < EPS):
            print('return the discont value at t=%s' % t)
            if self.ascending:
                ind = np.searchsorted(self.t_discont, t, side='left')
            else:
                ind = np.searchsorted(self.t_discont, t, side='right')
            return self.y_discont[ind]

        if self.ascending:
            ind = np.searchsorted(self.ts_sorted, t, side='left')
        else:
            ind = np.searchsorted(self.ts_sorted, t, side='right')

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        # special management for the first segment which is the
        # history conditions. As we store history values between t0-delayMax
        # and t0, the first segment is not a dense output of the RK integration.
        if(segment==0):
            history = self.interpolants[segment]
            if(type(history) is list):
                # this is list of cubicHermiteSpline as history was a tuple
                va = np.zeros(self.n)
                for k in range(self.n):
                    va[k] = history[k](t)
            elif(isfunction(history)):
                # history given from a function
                va = history(t)
            elif(isinstance(history, np.ndarray)):
                # from a cte
                va = history
            return va
        else:
            return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate at.

        Returns
        -------
        y : ndarray, shape (n_states,) or (n_states, n_points)
            Computed values. Shape depends on whether `t` is a scalar or a
            1-D array.
        """
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)

        # check if repeated time detected at initialisation of the class
        if(self.repeated_t):
            # check if repeated time values are present in t given by user
            isDiscont_in_t = np.any(np.any(np.sort(t)[0] <=
                            np.asarray(self.t_discont))
                            and np.any(np.asarray(self.t_discont) <
                                np.sort(t)[-1]))
            if isDiscont_in_t:
                warn("Discontinuities are present in time interval interpolation."
                     "Special management of this case is made")

            if not np.any(np.diff(t) < EPS) and isDiscont_in_t:
                raise ValueError("As discontinuities are within integration time interval"
                      "the user have to provide t with duplicates values in %s" %self.t_discont)
            # if it is the case, we remove repeated time to add them at the end
            # of interp process
            idxs = np.argwhere(np.diff(t) < EPS) + 1
            t = np.delete(t, idxs)
            order = np.argsort(t)
            reverse = np.empty_like(order)
            reverse[order] = np.arange(order.shape[0])
            t_sorted = t[order]
        else:
            isDiscont_in_t = False
            order = np.argsort(t)
            reverse = np.empty_like(order)
            reverse[order] = np.arange(order.shape[0])
            t_sorted = t[order]


        # See comment in self._call_single.
        if self.ascending:
            segments = np.searchsorted(self.ts_sorted, t_sorted, side='left')
        else:
            segments = np.searchsorted(self.ts_sorted, t_sorted, side='right')
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        group_start = 0
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            # special managment of first history segment
            if(segment==0):
                Nt = len(t_sorted[group_start:group_end])
                interp = self.interpolants[segment]
                # n = len(self.ys[0])
                y = np.zeros((self.n,Nt))
                for k in range(Nt):
                    if(type(interp) is list):
                        # this is list on cubicHermiteSpline
                        va = np.zeros(self.n)
                        for m in range(self.n):
                            va[m] = interp[m](t[k])
                    elif(isfunction(interp)):
                        # from a function
                        va = interp(t[k])
                    elif(isinstance(interp, np.ndarray)):
                        # from a cte
                        va = interp
                    y[:,k] = va
            else:
                y = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            group_start = group_end

        ys = np.hstack(ys)
        ys = ys[:, reverse]

        # Insertion of discontinuities
        if(self.repeated_t and isDiscont_in_t):
            t_tmp = t_sorted.copy()
            for k in range(len(self.t_discont)):
                idx = np.searchsorted(t_tmp, self.t_discont[k]) + 1
                t_tmp = np.insert(t_tmp, idx, self.t_discont[k])
                ys = np.insert(ys, idx, self.y_discont[k], axis=1)
        return ys

class ContinuousExtCyclic(object):
    """ Cyclic collection of dense ouput list and the corresponding times list.
    Informations only in time intervall [t, t-delayMax] are keeped. This class 
    is written from ContinousExtension.
    A cyclic management of data permit a better code efficency

    The interpolants cover the range between `t_min = t_current - delayMax`
    and `t_current` where t_current is the current integration time 
    (see Attributes below).

    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.

    Parameters
    ----------
    t0 : float
        Initial time
    delayMax : float
        Maximal delay
    h : callable
        History function
    t_discont : list
        The times of jumps
    y_discont : list
        The values of y at t_discont

    Attributes
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of history and DenseOutput with respectively 1 and
        n_segments-1 elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.
    t_min, t_max : float
        Time range of the interpolation.
    n_segments : int
        Number of interpolant.
    """

    def __init__(self, t0, delayMax, h, t_discont=[], y_discont=[]):
        self.t_discont = t_discont
        self.y_discont = y_discont
        self.delayMax = delayMax
        self.t0 = t0
        self.t_min = self.t0 - self.delayMax
        self.t_max = self.t0
        self.ts = [self.t_min, self.t0]
        self.interpolants = [h]
        self.n_segments = 1

    def update(self,t,sol):
        """ Update the cyclic storage of past values by adding new time and 
            continous extension

        Parameters
        ----------
        t : float or list
            New time points to add in self.ts
        sol : callable or list of callable
            New sol in intervall between self.ts[-1] et t

        Returns
        -------
        self.ts : list
            Update of self.ts
        self.interpolants : list
            Update of self.interpolants
        """
        if isinstance(t,list) and isinstance(sol,list):
            print('self.t0', self.t0, 't', t)
            if not np.isclose(t[-1], self.t0):
                raise ValueError('Problem of time continuity')
            self.ts = t
            self.interpolants = sol
            self.n_segments = len(self.interpolants)
        else:
            self.t_min = t - self.delayMax
            self.t_max = t
            self.ts.append(t)
            # print('after ts = %s' %(self.ts))
            self.interpolants.append(sol)
            self.n_segments += 1

        self.cleanUp()
        if len(self.ts) != self.n_segments + 1:
            raise ValueError("Numbers of time stamps and interpolants don't match.")
        if not np.all(np.diff(self.ts) > 0):
             raise ValueError("`ts` must be strictly increasing")

    def cleanUp(self):
        """Remove times and callable function sol (continous extension) when 
        not anymore useful. Useful informations for past values are in the 
        intervall [t_current-delayMax, t_current]

        Returns
        -------
        self.ts : list
            Update of self.ts
        self.interpolants : list
            Update of self.interpolants
        self.n_segments : int
            Update of self.n_segments
        """
        idx = [x for x, val in enumerate(self.ts) if val < self.t_min]
        to_rm = []
        for i in idx:
            if not (self.ts[i] < self.t_min and self.t_min < self.ts[i+1]):
                to_rm.append(i)
        # reverse the to-rm list for not throw off the subsequent indexes.
        for i in sorted(to_rm, reverse=True):
            del self.ts[i], self.interpolants[i]
        self.n_segments = len(self.interpolants)

    def _call_single(self, t):

        if not (self.t_min <= t and t <= self.t_max):
            raise ValueError('t not in delayed state time intervall')

        if self.t_discont and np.any(np.abs(np.array(self.t_discont) - t) < EPS):
            print('discont in Zeval')
            print('self.t_discont %s t=%s' % (self.t_discont,t))
            return np.asarray(self.y_discont)[np.abs(np.array(self.t_discont) - t) < EPS]
        else:
            ind = np.searchsorted(self.ts, t, side='left')
            segment = min(max(ind - 1, 0), self.n_segments - 1)
            return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.

        Parameters
        ----------
        t : float
            Points to evaluate at.

        Returns
        -------
        y : ndarray, shape (n_states,)
            Computed values. Shape depends on whether `t` is a scalar
        """
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)
        else:
            raise ValueError('t must be array where ndim == 0')
