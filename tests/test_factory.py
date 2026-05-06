"""Tests for septal.casadax.factory — SolverFactory."""

import jax.numpy as jnp
import numpy as np
import pytest

from septal.casadax.callbacks import casadify_reverse
from septal.casadax.factory import SolverFactory
from septal.casadax.schema import NLPProblem, SolveResult
from septal.casadax.solvers import CasadiSolver, JaxSolver


def _quadratic(x):
    return jnp.sum(x ** 2).reshape(1, 1)


def _quadratic_jax(x):
    """Scalar output for JaxSolver."""
    return jnp.sum(x ** 2)


class TestSolverFactoryFromMethod:
    def test_creates_casadi_solver(self, solver_cfg, simple_bounds):
        objective_cb = casadify_reverse(_quadratic, nd=2)
        factory = SolverFactory.from_method(
            solver_cfg, "general_constrained_nlp",
            objective_cb, simple_bounds,
        )
        assert isinstance(factory.solver, CasadiSolver)

    def test_creates_jax_solver(self, solver_cfg, simple_bounds):
        factory = SolverFactory.from_method(
            solver_cfg, "box_constrained_nlp",
            _quadratic_jax, simple_bounds,
        )
        assert isinstance(factory.solver, JaxSolver)

    def test_unknown_solver_type_raises(self, solver_cfg, simple_bounds):
        with pytest.raises(NotImplementedError):
            SolverFactory.from_method(
                solver_cfg, "nonexistent_type",
                _quadratic, simple_bounds,
            )

    def test_solve_returns_result(self, solver_cfg, simple_bounds):
        objective_cb = casadify_reverse(_quadratic, nd=2)
        factory = SolverFactory.from_method(
            solver_cfg, "general_constrained_nlp",
            objective_cb, simple_bounds,
        )
        guesses = factory.initial_guess()
        result = factory.solve(guesses)
        assert isinstance(result, SolveResult)

    def test_call_interface(self, solver_cfg, simple_bounds):
        """factory(guesses) should be identical to factory.solve(guesses)."""
        objective_cb = casadify_reverse(_quadratic, nd=2)
        factory = SolverFactory.from_method(
            solver_cfg, "general_constrained_nlp",
            objective_cb, simple_bounds,
        )
        guesses = factory.initial_guess()
        result = factory(guesses)
        assert isinstance(result, SolveResult)


class TestSolverFactoryFromProblem:
    def test_creates_correct_backend(self, solver_cfg, simple_bounds):
        objective_cb = casadify_reverse(_quadratic, nd=2)
        problem = NLPProblem(objective=objective_cb, bounds=simple_bounds)
        factory = SolverFactory.from_problem(solver_cfg, "general_constrained_nlp", problem)
        assert isinstance(factory.solver, CasadiSolver)

    def test_solves_correctly(self, solver_cfg, simple_bounds):
        """sum(x^2) on [-1,1]^2 — minimum at 0."""
        objective_cb = casadify_reverse(_quadratic, nd=2)
        problem = NLPProblem(objective=objective_cb, bounds=simple_bounds, n_starts=4)
        factory = SolverFactory.from_problem(solver_cfg, "general_constrained_nlp", problem)
        result = factory.solve(factory.initial_guess())
        assert result.success
        assert float(result.objective) == pytest.approx(0.0, abs=1e-3)

    def test_with_constraints(self, solver_cfg):
        """min sum(x^2) s.t. x[0]+x[1] >= 1, x in [0,2]^2 — optimum ~0.5."""
        n_d = 2
        lb = jnp.zeros(n_d)
        ub = jnp.full(n_d, 2.0)

        def _g(x):
            x_ = x.reshape(-1)
            return (1.0 - x_[0] - x_[1]).reshape(1, 1)

        objective_cb = casadify_reverse(_quadratic, nd=n_d)
        constraint_cb = casadify_reverse(_g, nd=n_d)

        problem = NLPProblem(
            objective=objective_cb,
            bounds=[lb, ub],
            constraints=constraint_cb,
            constraint_lhs=jnp.array([[-jnp.inf]]),
            constraint_rhs=jnp.zeros((1, 1)),
            n_starts=4,
        )
        factory = SolverFactory.from_problem(solver_cfg, "general_constrained_nlp", problem)
        result = factory.solve(factory.initial_guess())
        assert result.success
        assert float(result.objective) == pytest.approx(0.5, abs=5e-2)


class TestSolverFactoryGuards:
    def test_construct_without_objective_raises(self, solver_cfg, simple_bounds):
        factory = SolverFactory(solver_cfg, "general_constrained_nlp")
        factory.load_bounds(simple_bounds)
        with pytest.raises(ValueError, match="objective_func"):
            factory.construct_solver()

    def test_construct_without_bounds_raises(self, solver_cfg):
        factory = SolverFactory(solver_cfg, "general_constrained_nlp")
        factory.load_objective(_quadratic)
        with pytest.raises(ValueError, match="bounds"):
            factory.construct_solver()
