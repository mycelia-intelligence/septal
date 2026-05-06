"""Tests for septal.jax.sqp.convergence."""

import jax.numpy as jnp
import pytest

from septal.jax.sqp.convergence import is_converged, kkt_residuals
from septal.jax.sqp.schema import SQPConfig


def make_quadratic_problem():
    """Simple quadratic min ½‖x‖² s.t. x[0] >= 1 → x*=[1,0], λ*=1."""

    def f(x, p):
        return 0.5 * jnp.sum(x**2)

    def g(x, p):
        return (1.0 - x[0]).reshape(1)  # g<=0 via rhs=0

    lhs = jnp.array([-jnp.inf])
    rhs = jnp.array([0.0])
    return f, g, lhs, rhs


class TestKKTResiduals:
    def test_at_optimum_residuals_small(self):
        f, g, lhs, rhs = make_quadratic_problem()
        # x* = [1, 0], λ* = 1
        x_opt = jnp.array([1.0, 0.0])
        lam_opt = jnp.array([1.0])
        p = jnp.zeros(1)

        stat, feas = kkt_residuals(x_opt, lam_opt, p, f, g, lhs, rhs, 1)
        assert float(stat) == pytest.approx(0.0, abs=1e-6)
        assert float(feas) == pytest.approx(0.0, abs=1e-6)

    def test_at_infeasible_point(self):
        f, g, lhs, rhs = make_quadratic_problem()
        x = jnp.array([0.5, 0.0])  # violates x[0] >= 1
        lam = jnp.zeros(1)
        p = jnp.zeros(1)

        stat, feas = kkt_residuals(x, lam, p, f, g, lhs, rhs, 1)
        # g(x) = 1 - 0.5 = 0.5 > 0 = rhs → feasibility violation
        assert float(feas) > 1e-3

    def test_stationarity_nonzero_away_from_optimum(self):
        f, g, lhs, rhs = make_quadratic_problem()
        x = jnp.array([2.0, 1.0])
        lam = jnp.zeros(1)
        p = jnp.zeros(1)

        stat, _ = kkt_residuals(x, lam, p, f, g, lhs, rhs, 1)
        # grad_f = [2, 1] at x=[2,1], no Lagrangian correction → stat != 0
        assert float(stat) > 1e-3

    def test_no_constraints_feasibility_zero(self):
        def f(x, p):
            return 0.5 * jnp.sum(x**2)

        x = jnp.array([0.0, 0.0])
        p = jnp.zeros(0)
        stat, feas = kkt_residuals(x, jnp.zeros(0), p, f, None, None, None, 0)
        assert float(feas) == pytest.approx(0.0)


class TestIsConverged:
    def test_converged_when_below_tol(self):
        cfg = SQPConfig(tol_stationarity=1e-4, tol_feasibility=1e-4)
        assert bool(is_converged(jnp.array(1e-5), jnp.array(1e-5), cfg))

    def test_not_converged_when_stat_too_large(self):
        cfg = SQPConfig(tol_stationarity=1e-6, tol_feasibility=1e-6)
        assert not bool(is_converged(jnp.array(1e-4), jnp.array(1e-8), cfg))

    def test_not_converged_when_feas_too_large(self):
        cfg = SQPConfig(tol_stationarity=1e-6, tol_feasibility=1e-6)
        assert not bool(is_converged(jnp.array(1e-8), jnp.array(1e-4), cfg))
