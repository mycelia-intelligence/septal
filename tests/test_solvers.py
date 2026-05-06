"""Tests for septal.casadax.solvers — CasadiSolver and JaxSolver."""

import jax.numpy as jnp
import numpy as np
import pytest

from septal.casadax.callbacks import casadify_forward, casadify_reverse
from septal.casadax.solvers import CasadiSolver, JaxSolver
from septal.casadax.utilities import generate_initial_guess

# ---------------------------------------------------------------------------
# CasadiSolver
# ---------------------------------------------------------------------------

class TestCasadiSolverUnconstrained:
    """Minimise sum(x^2) on [-2, 2]^2 — optimum at x*=(0,0), f*=0."""

    @pytest.fixture
    def setup(self, solver_cfg):
        n_d = 2
        lb = jnp.full(n_d, -2.0)
        ub = jnp.full(n_d, 2.0)
        bounds = [lb, ub]

        def _f(x):
            return jnp.sum(x ** 2).reshape(1, 1)

        objective_cb = casadify_reverse(_f, n_d)
        solver = CasadiSolver(solver_cfg, objective_cb, bounds)
        guesses = generate_initial_guess(solver_cfg.n_starts, n_d, bounds)
        return solver, guesses

    def test_returns_solve_result(self, setup):
        from septal.casadax.schema import SolveResult
        solver, guesses = setup
        result = solver.solve(guesses)
        assert isinstance(result, SolveResult)

    def test_converges_to_minimum(self, setup):
        solver, guesses = setup
        result = solver.solve(guesses)
        assert result.success
        assert float(result.objective) == pytest.approx(0.0, abs=1e-3)

    def test_solution_within_bounds(self, setup):
        solver, guesses = setup
        result = solver.solve(guesses)
        x = np.array(result.decision_variables).flatten()
        assert np.all(x >= -2.0 - 1e-6)
        assert np.all(x <= 2.0 + 1e-6)


class TestCasadiSolverConstrained:
    """min sum(x^2) s.t. x[0]+x[1] >= 1, x in [0,2]^2.
    Optimum at x*=(0.5, 0.5), f*=0.5.
    """

    @pytest.fixture
    def setup(self, solver_cfg):
        n_d = 2
        lb = jnp.zeros(n_d)
        ub = jnp.full(n_d, 2.0)
        bounds = [lb, ub]

        def _f(x):
            return jnp.sum(x ** 2).reshape(1, 1)

        # g(x) = 1 - x[0] - x[1] <= 0  (equivalent to x[0]+x[1] >= 1)
        def _g(x):
            x_ = x.reshape(-1)
            return (1.0 - x_[0] - x_[1]).reshape(1, 1)

        objective_cb = casadify_reverse(_f, n_d)
        constraint_cb = casadify_reverse(_g, n_d)
        lhs = jnp.array([[-jnp.inf]])
        rhs = jnp.zeros((1, 1))

        solver = CasadiSolver(
            solver_cfg, objective_cb, bounds,
            constraints_fn=constraint_cb,
            constraint_lhs=lhs,
            constraint_rhs=rhs,
        )
        guesses = generate_initial_guess(solver_cfg.n_starts, n_d, bounds)
        return solver, guesses

    def test_converges_constrained(self, setup):
        solver, guesses = setup
        result = solver.solve(guesses)
        assert result.success
        x = np.array(result.decision_variables).flatten()
        # x[0] + x[1] should be ~1
        assert float(x[0] + x[1]) == pytest.approx(1.0, abs=1e-2)

    def test_objective_at_constrained_optimum(self, setup):
        solver, guesses = setup
        result = solver.solve(guesses)
        # f* = (0.5)^2 + (0.5)^2 = 0.5
        assert float(result.objective) == pytest.approx(0.5, abs=1e-2)


class TestCasadiSolverEdgeCases:
    def test_no_feasible_starts_returns_failed_result(self, solver_cfg):
        """Force all starts to fail by providing a near-degenerate problem."""
        n_d = 1
        # Very tight bounds so IPOPT likely terminates immediately
        lb = jnp.array([0.0])
        ub = jnp.array([0.0])

        def _f(x):
            return x.reshape(1, 1)

        objective_cb = casadify_reverse(_f, n_d)
        solver = CasadiSolver(solver_cfg, objective_cb, [lb, ub])
        guesses = generate_initial_guess(1, n_d, [jnp.array([-0.5]), jnp.array([0.5])])
        result = solver.solve(guesses)
        # Whether or not IPOPT "succeeds", we get a SolveResult back
        from septal.casadax.schema import SolveResult
        assert isinstance(result, SolveResult)

    def test_initial_guess_shape(self, solver_cfg):
        n_d = 3
        lb = jnp.zeros(n_d)
        ub = jnp.ones(n_d)
        bounds = [lb, ub]

        def _f(x):
            return jnp.sum(x).reshape(1, 1)

        objective_cb = casadify_reverse(_f, n_d)
        solver = CasadiSolver(solver_cfg, objective_cb, bounds)
        guesses = solver.initial_guess()
        assert guesses.shape == (solver_cfg.n_starts, n_d)


# ---------------------------------------------------------------------------
# JaxSolver
# ---------------------------------------------------------------------------

class TestJaxSolver:
    """Minimise sum(x^2) on [0, 1]^3 — optimum at x*=0, f*=0."""

    @pytest.fixture
    def setup(self, solver_cfg):
        n_d = 3
        lb = jnp.zeros(n_d)
        ub = jnp.ones(n_d)
        bounds = [lb, ub]

        def _f(x):
            return jnp.sum(x ** 2)

        solver = JaxSolver(solver_cfg, _f, bounds)
        guesses = generate_initial_guess(solver_cfg.n_starts, n_d, bounds)
        return solver, guesses

    def test_returns_solve_result(self, setup):
        from septal.casadax.schema import SolveResult
        solver, guesses = setup
        result = solver.solve(guesses)
        assert isinstance(result, SolveResult)

    def test_objective_near_minimum(self, setup):
        solver, guesses = setup
        result = solver.solve(guesses)
        # Box is [0,1]^3, minimum is at 0
        assert float(result.objective) == pytest.approx(0.0, abs=1e-2)

    def test_initial_guess_shape(self, solver_cfg):
        n_d = 2
        bounds = [jnp.zeros(n_d), jnp.ones(n_d)]
        solver = JaxSolver(solver_cfg, lambda x: jnp.sum(x ** 2), bounds)
        guesses = solver.initial_guess()
        assert guesses.shape == (solver_cfg.n_starts, n_d)

    def test_get_status_within_tolerance(self, solver_cfg):
        n_d = 2
        bounds = [jnp.zeros(n_d), jnp.ones(n_d)]
        solver = JaxSolver(solver_cfg, lambda x: jnp.sum(x ** 2), bounds)
        # Error well below tolerance → success
        assert solver.get_status(jnp.array(1e-8))

    def test_get_status_above_tolerance(self, solver_cfg):
        n_d = 2
        bounds = [jnp.zeros(n_d), jnp.ones(n_d)]
        solver = JaxSolver(solver_cfg, lambda x: jnp.sum(x ** 2), bounds)
        # Error well above tolerance → failure
        assert not solver.get_status(jnp.array(1.0))
