"""Shared fixtures for septal tests."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def simple_bounds():
    """2D box: x in [-1, 1]^2."""
    lb = jnp.array([-1.0, -1.0])
    ub = jnp.array([1.0, 1.0])
    return [lb, ub]


@pytest.fixture
def sphere_bounds():
    """3D box: x in [0, 1]^3."""
    lb = jnp.zeros(3)
    ub = jnp.ones(3)
    return [lb, ub]


@pytest.fixture
def solver_cfg():
    """Minimal solver config compatible with CasadiSolver / JaxSolver."""
    return SimpleNamespace(
        n_starts=4,
        max_solution_time=60.0,
        jax_opt_options=SimpleNamespace(error_tol=1e-3),
    )


@pytest.fixture
def quadratic_objective():
    """sum(x^2) — minimum at 0."""
    def _f(x):
        return jnp.sum(x ** 2).reshape(1, 1)
    return _f


@pytest.fixture
def rosenbrock_objective():
    """2D Rosenbrock (n=2): minimum at (1,1)."""
    def _f(x):
        x_ = x.reshape(-1)
        return (jnp.array([(1 - x_[0]) ** 2 + 100 * (x_[1] - x_[0] ** 2) ** 2])
                .reshape(1, 1))
    return _f


@pytest.fixture
def simple_constraint():
    """g(x) = x[0] + x[1] - 0.5 <= 0 (feasible when x[0]+x[1] <= 0.5)."""
    def _g(x):
        x_ = x.reshape(-1)
        return (x_[0] + x_[1] - 0.5).reshape(1, 1)
    return _g
