"""Tests for septal.jax.sqp.factory.ParametricSQPFactory."""

import jax
import jax.numpy as jnp
import pytest

from septal.casadax.schema import NLPProblem
from septal.jax.sqp.factory import ParametricSQPFactory
from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig, SQPResult

# ---------------------------------------------------------------------------
# Problem helpers
# ---------------------------------------------------------------------------


def make_quadratic_problem(n=2, m=2):
    """min ½‖x - p‖²,  x ∈ [-2, 2]^n,  p ∈ R^m."""

    def f(x, p):
        return 0.5 * jnp.sum((x - p) ** 2)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(n, -2.0), jnp.full(n, 2.0)],
        n_decision=n,
        n_params=m,
    )


def make_constrained_problem():
    """min ½‖x‖²  s.t.  x[0] + x[1] >= p."""

    def f(x, p):
        return 0.5 * jnp.sum(x**2)

    def g(x, p):
        return (p[0] - x[0] - x[1]).reshape(1)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(2, -5.0), jnp.full(2, 5.0)],
        n_decision=2,
        n_params=1,
        constraints=g,
        constraint_lhs=jnp.array([-jnp.inf]),
        constraint_rhs=jnp.array([0.0]),
    )


@pytest.fixture
def cfg():
    return SQPConfig(
        max_iter=50,
        tol_stationarity=1e-4,
        tol_feasibility=1e-4,
        osqp_tol=1e-6,
        osqp_max_iter=2000,
    )


# ---------------------------------------------------------------------------
# Single solve
# ---------------------------------------------------------------------------


class TestSingleSolve:
    def test_returns_sqp_result(self, cfg):
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        result = factory.solve(jnp.zeros(2), jnp.array([0.5, -0.5]))
        assert isinstance(result, SQPResult)

    def test_converges_to_optimum(self, cfg):
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        p = jnp.array([1.0, -1.0])
        result = factory.solve(jnp.zeros(2), p)
        assert result.success
        assert jnp.allclose(result.decision_variables, p, atol=1e-3)

    def test_initial_guess_within_bounds(self, cfg):
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        x0 = factory.initial_guess()
        lb = factory.problem.lb
        ub = factory.problem.ub
        assert jnp.all(x0 >= lb - 1e-10)
        assert jnp.all(x0 <= ub + 1e-10)

    def test_while_loop_variant(self, cfg):
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        result = factory.solve(jnp.zeros(2), jnp.array([0.5, 0.5]), use_while_loop=True)
        assert isinstance(result, SQPResult)
        assert result.success

    def test_timing_recorded(self, cfg):
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        result = factory.solve(jnp.zeros(2), jnp.array([1.0, 0.0]))
        assert result.timing >= 0.0

    def test_constrained_solve(self, cfg):
        factory = ParametricSQPFactory(make_constrained_problem(), cfg)
        result = factory.solve(jnp.zeros(2), jnp.array([1.0]))
        assert result.success
        assert (
            float(result.decision_variables[0] + result.decision_variables[1])
            >= 1.0 - 1e-3
        )


# ---------------------------------------------------------------------------
# Batched solve
# ---------------------------------------------------------------------------


class TestBatchedSolve:
    def test_batch_output_shapes(self, cfg):
        N = 8
        factory = ParametricSQPFactory(make_quadratic_problem(n=2, m=2), cfg)
        params = jax.random.normal(jax.random.PRNGKey(0), (N, 2)) * 0.5
        x0s = jnp.zeros((N, 2))
        result = factory.solve_batch(x0s, params)

        assert result.decision_variables.shape == (N, 2)
        assert result.success.shape == (N,)
        assert result.objective.shape == (N,)

    def test_batch_broadcast_x0(self, cfg):
        """Single x0 (shape (n,)) should broadcast to all N problems."""
        N = 5
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        params = jnp.ones((N, 2)) * 0.5
        x0 = jnp.zeros(2)  # 1D — should broadcast
        result = factory.solve_batch(x0, params)
        assert result.decision_variables.shape == (N, 2)

    def test_batch_different_params_different_solutions(self, cfg):
        """Different parameters → different optima."""
        N = 6
        factory = ParametricSQPFactory(make_quadratic_problem(n=1, m=1), cfg)
        params = jnp.linspace(-1.0, 1.0, N).reshape(N, 1)
        x0s = jnp.zeros((N, 1))
        result = factory.solve_batch(x0s, params)
        # x* ≈ p for each
        assert jnp.allclose(
            result.decision_variables.reshape(N), params.reshape(N), atol=1e-3
        )

    def test_batch_timing_recorded(self, cfg):
        N = 4
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        params = jnp.zeros((N, 2))
        x0s = jnp.zeros((N, 2))
        result = factory.solve_batch(x0s, params)
        assert result.timing >= 0.0


# ---------------------------------------------------------------------------
# compile_batch
# ---------------------------------------------------------------------------


class TestCompileBatch:
    def test_compile_and_run(self, cfg):
        N = 4
        factory = ParametricSQPFactory(make_quadratic_problem(), cfg)
        solve = factory.compile_batch(N)

        params = jnp.ones((N, 2)) * 0.3
        x0s = jnp.zeros((N, 2))
        result = solve(x0s, params)
        assert result.decision_variables.shape == (N, 2)

    def test_compiled_matches_uncompiled(self, cfg):
        N = 3
        factory = ParametricSQPFactory(make_quadratic_problem(n=2, m=2), cfg)
        solve_compiled = factory.compile_batch(N)

        params = jnp.array([[0.5, -0.5]] * N)
        x0s = jnp.zeros((N, 2))

        result_uncompiled = factory.solve_batch(x0s, params)
        result_compiled = solve_compiled(x0s, params)
        assert jnp.allclose(
            result_compiled.decision_variables,
            result_uncompiled.decision_variables,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# from_nlp_problem class method
# ---------------------------------------------------------------------------


class TestFromNLPProblem:
    def test_construction_from_nlp_problem(self, cfg):
        """from_nlp_problem must wrap an NLPProblem with a (x, p) objective."""

        def f(x, p):
            return jnp.sum(x**2)  # accepts p but ignores it

        nlp = NLPProblem(
            objective=f,
            bounds=[jnp.full(2, -1.0), jnp.full(2, 1.0)],
        )
        factory = ParametricSQPFactory.from_nlp_problem(nlp, n_params=3, cfg=cfg)
        assert isinstance(factory, ParametricSQPFactory)
        assert factory.problem.n_params == 3

    def test_from_nlp_problem_solves(self, cfg):
        def f(x, p):
            return jnp.sum(x**2)

        nlp = NLPProblem(
            objective=f,
            bounds=[jnp.full(2, -1.0), jnp.full(2, 1.0)],
        )
        factory = ParametricSQPFactory.from_nlp_problem(nlp, n_params=2, cfg=cfg)
        result = factory.solve(jnp.full(2, 0.5), jnp.zeros(2))
        assert result.success
        assert float(result.objective) == pytest.approx(0.0, abs=1e-3)
