from itertools import product
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import RK23, RK45

from scipy.integrate.dde.solve_dde import solve_dde
from scipy.integrate.dde.common import ContinuousExt
from scipy.integrate.dde.base import ConstantDenseOutput


def fun_diverging(t, y, Z):
    y_tau0 = Z[:,0]
    return np.array([ y_tau0 ])

def test_integration():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1.0]
    h = [1.0]
    delays = [1.0]

    for method, t_span in product(['RK23', 'RK45'],[[0.0, 2.0], [0.0, 5.0]]):
        fun = fun_rational
        res = solve_ivp(fun, t_span, delays, y0, h, rtol=rtol, atol=atol,
                        method=metho)
        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        if method in ['RK23', 'RK45']:
            assert_equal(res.njev, 0)
            assert_equal(res.nlu, 0)

        #y_true = sol_rational(res.t)
        #e = compute_error(res.y, y_true, rtol, atol)
        #assert_(np.all(e < 5))

        #tc = np.linspace(*t_span)
        #yc_true = sol_rational(tc)
        #yc = res.sol(tc)

        #e = compute_error(yc, yc_true, rtol, atol)
        #assert_(np.all(e < 5))

        #tc = (t_span[0] + t_span[-1]) / 2
        #yc_true = sol_rational(tc)
        #yc = res.sol(tc)

        #e = compute_error(yc, yc_true, rtol, atol)
        #assert_(np.all(e < 5))


#def test_first_step():
#    rtol = 1e-3
#    atol = 1e-6
#    y0 = [1/3, 2/9]
#    first_step = 0.1
#    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
#        for t_span in ([5, 9], [5, 1]):
#            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
#                            max_step=0.5, atol=atol, method=method,
#                            dense_output=True, first_step=first_step)
#
#            assert_equal(res.t[0], t_span[0])
#            assert_equal(res.t[-1], t_span[-1])
#            assert_allclose(first_step, np.abs(res.t[1] - 5))
#            assert_(res.t_events is None)
#            assert_(res.success)
#            assert_equal(res.status, 0)
#
#            y_true = sol_rational(res.t)
#            e = compute_error(res.y, y_true, rtol, atol)
#            assert_(np.all(e < 5))
#
#            tc = np.linspace(*t_span)
#            yc_true = sol_rational(tc)
#            yc = res.sol(tc)
#
#            e = compute_error(yc, yc_true, rtol, atol)
#            assert_(np.all(e < 5))
#
#            # See comment in test_integration.
#            if method is not LSODA:
#                assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)
#
#            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
#                          t_span[1], first_step=-1)
#            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
#                          t_span[1], first_step=5)


# def test_t_eval():
    # rtol = 1e-3
    # atol = 1e-6
    # y0 = [1/3, 2/9]
    # for t_span in ([5, 9], [5, 1]):
        # t_eval = np.linspace(t_span[0], t_span[1], 10)
        # res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                        # t_eval=t_eval)
        # assert_equal(res.t, t_eval)
        # assert_(res.t_events is None)
        # assert_(res.success)
        # assert_equal(res.status, 0)

        # y_true = sol_rational(res.t)
        # e = compute_error(res.y, y_true, rtol, atol)
        # assert_(np.all(e < 5))

    # t_eval = [5, 5.01, 7, 8, 8.01, 9]
    # res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    # t_eval=t_eval)
    # assert_equal(res.t, t_eval)
    # assert_(res.t_events is None)
    # assert_(res.success)
    # assert_equal(res.status, 0)

    # y_true = sol_rational(res.t)
    # e = compute_error(res.y, y_true, rtol, atol)
    # assert_(np.all(e < 5))

    # t_eval = [5, 4.99, 3, 1.5, 1.1, 1.01, 1]
    # res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    # t_eval=t_eval)
    # assert_equal(res.t, t_eval)
    # assert_(res.t_events is None)
    # assert_(res.success)
    # assert_equal(res.status, 0)

    # t_eval = [5.01, 7, 8, 8.01]
    # res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    # t_eval=t_eval)
    # assert_equal(res.t, t_eval)
    # assert_(res.t_events is None)
    # assert_(res.success)
    # assert_equal(res.status, 0)

    # y_true = sol_rational(res.t)
    # e = compute_error(res.y, y_true, rtol, atol)
    # assert_(np.all(e < 5))

    # t_eval = [4.99, 3, 1.5, 1.1, 1.01]
    # res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    # t_eval=t_eval)
    # assert_equal(res.t, t_eval)
    # assert_(res.t_events is None)
    # assert_(res.success)
    # assert_equal(res.status, 0)

    # t_eval = [4, 6]
    # assert_raises(ValueError, solve_ivp, fun_rational, [5, 9], y0,
                  # rtol=rtol, atol=atol, t_eval=t_eval)


# def test_t_eval_dense_output():
    # rtol = 1e-3
    # atol = 1e-6
    # y0 = [1/3, 2/9]
    # t_span = [5, 9]
    # t_eval = np.linspace(t_span[0], t_span[1], 10)
    # res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    # t_eval=t_eval)
    # res_d = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                      # t_eval=t_eval, dense_output=True)
    # assert_equal(res.t, t_eval)
    # assert_(res.t_events is None)
    # assert_(res.success)
    # assert_equal(res.status, 0)

    # assert_equal(res.t, res_d.t)
    # assert_equal(res.y, res_d.y)
    # assert_(res_d.t_events is None)
    # assert_(res_d.success)
    # assert_equal(res_d.status, 0)

    # # if t and y are equal only test values for one case
    # y_true = sol_rational(res.t)
    # e = compute_error(res.y, y_true, rtol, atol)
    # assert_(np.all(e < 5))


#def test_no_integration():
#    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
#        sol = solve_ivp(lambda t, y: -y, [4, 4], [2, 3],
#                        method=method, dense_output=True)
#        assert_equal(sol.sol(4), [2, 3])
#        assert_equal(sol.sol([4, 5, 6]), [[2, 2, 2], [3, 3, 3]])
#
#
#def test_no_integration_class():
#    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
#        solver = method(lambda t, y: -y, 0.0, [10.0, 0.0], 0.0)
#        solver.step()
#        assert_equal(solver.status, 'finished')
#        sol = solver.dense_output()
#        assert_equal(sol(0.0), [10.0, 0.0])
#        assert_equal(sol([0, 1, 2]), [[10, 10, 10], [0, 0, 0]])
#
#        solver = method(lambda t, y: -y, 0.0, [], np.inf)
#        solver.step()
#        assert_equal(solver.status, 'finished')
#        sol = solver.dense_output()
#        assert_equal(sol(100.0), [])
#        assert_equal(sol([0, 1, 2]), np.empty((0, 3)))
#
#
#def test_empty():
#    def fun(t, y):
#        return np.zeros((0,))
#
#    y0 = np.zeros((0,))
#
#    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
#        sol = assert_no_warnings(solve_ivp, fun, [0, 10], y0,
#                                 method=method, dense_output=True)
#        assert_equal(sol.sol(10), np.zeros((0,)))
#        assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))
#
#    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
#        sol = assert_no_warnings(solve_ivp, fun, [0, np.inf], y0,
#                                 method=method, dense_output=True)
#        assert_equal(sol.sol(10), np.zeros((0,)))
#        assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))
#
#
#def test_ConstantDenseOutput():
#    sol = ConstantDenseOutput(0, 1, np.array([1, 2]))
#    assert_allclose(sol(1.5), [1, 2])
#    assert_allclose(sol([1, 1.5, 2]), [[1, 1, 1], [2, 2, 2]])
#
#    sol = ConstantDenseOutput(0, 1, np.array([]))
#    assert_allclose(sol(1.5), np.empty(0))
#    assert_allclose(sol([1, 1.5, 2]), np.empty((0, 3)))


#def test_classes():
#    y0 = [1 / 3, 2 / 9]
#    for cls in [RK23, RK45]:
#        solver = cls(fun_rational, 5, y0, np.inf)
#        assert_equal(solver.n, 2)
#        assert_equal(solver.status, 'running')
#        assert_equal(solver.t_bound, np.inf)
#        assert_equal(solver.direction, 1)
#        assert_equal(solver.t, 5)
#        assert_equal(solver.y, y0)
#        assert_(solver.step_size is None)
#        assert_equal(solver.nfev, 0)
#        assert_equal(solver.njev, 0)
#        assert_equal(solver.nlu, 0)
#
#        message = solver.step()
#        assert_equal(solver.status, 'running')
#        assert_equal(message, None)
#        assert_equal(solver.n, 2)
#        assert_equal(solver.t_bound, np.inf)
#        assert_equal(solver.direction, 1)
#        assert_(solver.t > 5)
#        assert_(not np.all(np.equal(solver.y, y0)))
#        assert_(solver.step_size > 0)
#        assert_(solver.nfev > 0)
#        assert_(solver.njev >= 0)
#        assert_(solver.nlu >= 0)
#        sol = solver.dense_output()
#        assert_allclose(sol(5), y0, rtol=1e-15, atol=0)


def test_ContinuousExt():
    ts = np.array([0, 2, 5], dtype=float)
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))
    print('s1',s1,'s2',s2)
    sol = ContinuousExt(ts, [s1, s2])

    assert_equal(sol(-1), [-1])
    assert_equal(sol(1), [-1])
    assert_equal(sol(2), [-1])
    assert_equal(sol(3), [1])
    assert_equal(sol(5), [1])
    assert_equal(sol(6), [1])

    assert_equal(sol([0, 6, -2, 1.5, 4.5, 2.5, 5, 5.5, 2]),
                 np.array([[-1, 1, -1, -1, 1, 1, 1, 1, -1]]))

    ts = np.array([10, 4, -3])
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))

    sol = ContinuousExt(ts, [s1, s2])
    assert_equal(sol(11), [-1])
    assert_equal(sol(10), [-1])
    assert_equal(sol(5), [-1])
    assert_equal(sol(4), [-1])
    assert_equal(sol(0), [1])
    assert_equal(sol(-3), [1])
    assert_equal(sol(-4), [1])

    assert_equal(sol([12, -5, 10, -3, 6, 1, 4]),
                 np.array([[-1, 1, -1, 1, -1, 1, -1]]))

    ts = np.array([1, 1])
    s = ConstantDenseOutput(1, 1, np.array([10]))
    sol = ContinuousExt(ts, [s])
    assert_equal(sol(0), [10])
    assert_equal(sol(1), [10])
    assert_equal(sol(2), [10])

    assert_equal(sol([2, 1, 0]), np.array([[10, 10, 10]]))

