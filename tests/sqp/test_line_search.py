"""Tests for septal.jax.sqp.line_search."""

import jax
import jax.numpy as jnp
import pytest

from septal.jax.sqp.line_search import (
    backtracking_line_search,
    constraint_violation,
    l1_merit,
    merit_directional_deriv,
    update_penalty,
)
from septal.jax.sqp.schema import SQPConfig


@pytest.fixture
def cfg():
    return SQPConfig(
        line_search_beta=0.5,
        line_search_c=1e-4,
        line_search_alpha0=1.0,
        max_line_search=30,
    )


class TestConstraintViolation:
    def test_feasible_point(self):
        g = jnp.array([0.5])
        lhs = jnp.array([-jnp.inf])
        rhs = jnp.array([1.0])
        assert float(constraint_violation(g, lhs, rhs)) == pytest.approx(0.0)

    def test_upper_violation(self):
        g = jnp.array([2.0])
        rhs = jnp.array([1.0])
        lhs = jnp.array([-jnp.inf])
        assert float(constraint_violation(g, lhs, rhs)) == pytest.approx(1.0)

    def test_lower_violation(self):
        g = jnp.array([0.0])
        lhs = jnp.array([1.0])
        rhs = jnp.array([jnp.inf])
        assert float(constraint_violation(g, lhs, rhs)) == pytest.approx(1.0)

    def test_zero_for_empty(self):
        assert (
            float(constraint_violation(jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))) == 0.0
        )


class TestL1Merit:
    def test_no_constraints(self):
        merit = l1_merit(
            jnp.array(3.0), jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), jnp.array(10.0), 0
        )
        assert float(merit) == pytest.approx(3.0)

    def test_with_violation(self):
        # f=1, g=2, rhs=1, violation=1 → merit = 1 + rho*1
        merit = l1_merit(
            jnp.array(1.0),
            jnp.array([2.0]),
            jnp.array([-jnp.inf]),
            jnp.array([1.0]),
            jnp.array(5.0),
            1,
        )
        assert float(merit) == pytest.approx(6.0)

    def test_feasible_no_penalty(self):
        # g within bounds → penalty term is zero
        merit = l1_merit(
            jnp.array(1.0),
            jnp.array([0.5]),
            jnp.array([0.0]),
            jnp.array([1.0]),
            jnp.array(100.0),
            1,
        )
        assert float(merit) == pytest.approx(1.0)


class TestUpdatePenalty:
    def test_non_decreasing(self):
        lam = jnp.array([0.5, 1.5])
        rho = jnp.array(2.0)
        # infeasible iterate → non-decreasing branch: target=1.6 < 2.0 → stays at 2.0
        rho_new = update_penalty(
            lam, rho, 2, eps=0.1, feasibility=jnp.array(1.0), decrease_factor=0.5
        )
        assert float(rho_new) == pytest.approx(2.0)

    def test_increases_when_multiplier_large(self):
        lam = jnp.array([5.0])
        rho = jnp.array(1.0)
        rho_new = update_penalty(
            lam, rho, 1, eps=0.01, feasibility=jnp.array(1.0), decrease_factor=1.0
        )
        assert float(rho_new) > float(rho)

    def test_no_constraints_unchanged(self):
        rho = jnp.array(3.0)
        rho_new = update_penalty(
            jnp.zeros(0),
            rho,
            0,
            eps=0.1,
            feasibility=jnp.array(0.0),
            decrease_factor=1.0,
        )
        assert float(rho_new) == pytest.approx(3.0)


class TestBacktrackingLineSearch:
    def test_decreasing_quadratic(self, cfg):
        """Line search on f(x) = ‖x‖² with direction d = -grad_f."""

        def objective(x, p):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 1.0])
        d = jnp.array([-2.0, -2.0])  # -grad_f at x
        p = jnp.zeros(0)

        f0 = objective(x, p)
        merit0 = jnp.asarray(f0)
        dir_deriv = jnp.array(-8.0)  # grad_f @ d = [2,2]@[-2,-2]

        alpha = backtracking_line_search(
            x,
            d,
            p,
            merit0,
            dir_deriv,
            jnp.array(0.0),
            objective,
            None,
            jnp.zeros(0),
            jnp.zeros(0),
            0,
            cfg,
        )
        # Armijo: f(x + alpha*d) <= f(x) + c*alpha*dir_deriv
        x_new = x + alpha * d
        f_new = objective(x_new, p)
        assert (
            float(f_new)
            <= float(merit0)
            + cfg.line_search_c * float(alpha) * float(dir_deriv)
            + 1e-10
        )

    def test_step_positive(self, cfg):
        def objective(x, p):
            return jnp.sum(x**2)

        x = jnp.array([0.5])
        d = jnp.array([-1.0])
        p = jnp.zeros(0)
        alpha = backtracking_line_search(
            x,
            d,
            p,
            jnp.array(0.25),
            jnp.array(-1.0),
            jnp.array(0.0),
            objective,
            None,
            jnp.zeros(0),
            jnp.zeros(0),
            0,
            cfg,
        )
        assert float(alpha) > 0.0

    def test_jit_compatible(self, cfg):
        def objective(x, p):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 1.0])
        d = jnp.array([-1.0, -1.0])
        p = jnp.zeros(0)

        fn = jax.jit(
            lambda x, d: backtracking_line_search(
                x,
                d,
                p,
                jnp.array(2.0),
                jnp.array(-4.0),
                jnp.array(0.0),
                objective,
                None,
                jnp.zeros(0),
                jnp.zeros(0),
                0,
                cfg,
            )
        )
        alpha = fn(x, d)
        assert float(alpha) > 0.0
