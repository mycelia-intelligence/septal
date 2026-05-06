"""Tests for septal.casadax.schema — NLPProblem and SolveResult."""

import jax.numpy as jnp
import numpy as np
import pytest

from septal.casadax.schema import NLPProblem, SolveResult


class TestNLPProblem:
    def test_basic_construction(self, simple_bounds, quadratic_objective):
        problem = NLPProblem(objective=quadratic_objective, bounds=simple_bounds)
        assert problem.n_decision == 2
        assert problem.n_starts == 5
        assert not problem.has_constraints

    def test_n_decision_inferred_from_bounds(self, sphere_bounds, quadratic_objective):
        problem = NLPProblem(objective=quadratic_objective, bounds=sphere_bounds)
        assert problem.n_decision == 3

    def test_n_decision_explicit_override(self, simple_bounds, quadratic_objective):
        problem = NLPProblem(
            objective=quadratic_objective, bounds=simple_bounds, n_decision=2
        )
        assert problem.n_decision == 2

    def test_lb_ub_properties(self, simple_bounds, quadratic_objective):
        problem = NLPProblem(objective=quadratic_objective, bounds=simple_bounds)
        assert jnp.all(problem.lb == simple_bounds[0])
        assert jnp.all(problem.ub == simple_bounds[1])

    def test_has_constraints_false(self, simple_bounds, quadratic_objective):
        problem = NLPProblem(objective=quadratic_objective, bounds=simple_bounds)
        assert not problem.has_constraints

    def test_has_constraints_true(
        self, simple_bounds, quadratic_objective, simple_constraint
    ):
        problem = NLPProblem(
            objective=quadratic_objective,
            bounds=simple_bounds,
            constraints=simple_constraint,
            constraint_lhs=jnp.array([-jnp.inf]),
            constraint_rhs=jnp.zeros(1),
        )
        assert problem.has_constraints

    def test_custom_n_starts(self, simple_bounds, quadratic_objective):
        problem = NLPProblem(
            objective=quadratic_objective, bounds=simple_bounds, n_starts=10
        )
        assert problem.n_starts == 10


class TestSolveResult:
    def test_basic_construction(self):
        result = SolveResult(
            success=True,
            objective=0.0,
            decision_variables=np.zeros(2),
        )
        assert result.success
        assert result.objective == 0.0
        assert result.constraints is None
        assert result.message == ""
        assert result.timing == 0.0
        assert result.n_solves == 0

    def test_failed_result(self):
        result = SolveResult(
            success=False,
            objective=np.inf,
            decision_variables=None,
            message="Infeasible",
        )
        assert not result.success
        assert result.message == "Infeasible"

    def test_full_construction(self):
        result = SolveResult(
            success=True,
            objective=1.23,
            decision_variables=np.array([0.5, 0.5]),
            constraints=np.array([-0.1]),
            message="Solve_Succeeded",
            timing=0.42,
            n_solves=3,
        )
        assert result.timing == pytest.approx(0.42)
        assert result.n_solves == 3
        assert result.constraints is not None
