"""Shared fixtures for septal.jax.sqp tests."""

import jax.numpy as jnp
import pytest

from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig

# ---------------------------------------------------------------------------
# Reusable problem definitions
# ---------------------------------------------------------------------------


@pytest.fixture
def sqp_cfg():
    """Loose tolerances — fast for unit tests."""
    return SQPConfig(
        max_iter=50,
        tol_stationarity=1e-5,
        tol_feasibility=1e-5,
        osqp_tol=1e-6,
        osqp_max_iter=2000,
    )


@pytest.fixture
def unconstrained_quadratic():
    """min ½‖x - p‖² — optimal x* = clip(p, lb, ub).

    With lb=-2, ub=2 and p within bounds: x* = p, f* = 0.
    """
    def f(x, p):
        return 0.5 * jnp.sum((x - p) ** 2)

    lb = jnp.full(2, -2.0)
    ub = jnp.full(2, 2.0)

    return ParametricNLPProblem(
        objective=f,
        bounds=[lb, ub],
        n_decision=2,
        n_params=2,
    )


@pytest.fixture
def constrained_quadratic():
    """min ½‖x‖² s.t. x[0] >= p[0]  (i.e. p[0] - x[0] <= 0).

    Optimal: x* = [p[0], 0] when p[0] >= 0, x* = [0, 0] otherwise.
    """
    def f(x, p):
        return 0.5 * jnp.sum(x ** 2)

    def g(x, p):
        # g(x, p) = p[0] - x[0] <= 0  →  lhs=-inf, rhs=0
        return (p[0] - x[0]).reshape(1)

    lb = jnp.full(2, -5.0)
    ub = jnp.full(2, 5.0)

    return ParametricNLPProblem(
        objective=f,
        bounds=[lb, ub],
        n_decision=2,
        n_params=1,
        constraints=g,
        constraint_lhs=jnp.array([-jnp.inf]),
        constraint_rhs=jnp.array([0.0]),
    )


@pytest.fixture
def constrained_quadratic_equality():
    """min ½‖x‖² s.t. x[0] + x[1] = p[0].

    Optimal: x* = [p[0]/2, p[0]/2], f* = p[0]²/4.
    """
    def f(x, p):
        return 0.5 * jnp.sum(x ** 2)

    def g(x, p):
        return (x[0] + x[1] - p[0]).reshape(1)

    lb = jnp.full(2, -10.0)
    ub = jnp.full(2, 10.0)

    return ParametricNLPProblem(
        objective=f,
        bounds=[lb, ub],
        n_decision=2,
        n_params=1,
        constraints=g,
        constraint_lhs=jnp.array([0.0]),
        constraint_rhs=jnp.array([0.0]),
    )
