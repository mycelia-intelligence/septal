"""Tests for septal.jax.sqp.solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig, SQPState
from septal.jax.sqp.solver import (
    batch_state_to_result,
    init_sqp_state,
    make_solver,
    sqp_solve_scan,
    sqp_solve_single,
    state_to_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_unconstrained(n=2):
    """min ½‖x - p‖²,  lb=-2, ub=2.  Optimal x* = clip(p, -2, 2)."""
    def f(x, p): return 0.5 * jnp.sum((x - p) ** 2)
    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(n, -2.0), jnp.full(n, 2.0)],
        n_decision=n,
        n_params=n,
    )


def make_constrained():
    """min ½‖x‖²  s.t.  x[0] + x[1] >= p[0].  Optimal: x* = [p/2, p/2]."""
    def f(x, p): return 0.5 * jnp.sum(x ** 2)
    def g(x, p): return (p[0] - x[0] - x[1]).reshape(1)  # g <= 0
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
        max_iter=60,
        tol_stationarity=1e-5,
        tol_feasibility=1e-5,
        osqp_tol=1e-7,
        osqp_max_iter=2000,
    )


# ---------------------------------------------------------------------------
# init_sqp_state
# ---------------------------------------------------------------------------

class TestInitSQPState:
    def test_shapes_unconstrained(self, cfg):
        problem = make_unconstrained(n=3)
        x0 = jnp.zeros(3)
        p = jnp.ones(3)
        state = init_sqp_state(x0, p, problem, cfg)

        assert state.x.shape == (3,)
        assert state.params_p.shape == (3,)
        assert state.lam.shape == (0,)
        assert state.hessian.shape == (3, 3)
        assert state.converged.shape == ()

    def test_shapes_constrained(self, cfg):
        problem = make_constrained()
        x0 = jnp.zeros(2)
        p = jnp.array([1.0])
        state = init_sqp_state(x0, p, problem, cfg)

        assert state.x.shape == (2,)
        assert state.lam.shape == (1,)
        assert state.hessian.shape == (2, 2)

    def test_initial_hessian_is_identity_scaled(self, cfg):
        problem = make_unconstrained(n=2)
        state = init_sqp_state(jnp.zeros(2), jnp.zeros(2), problem, cfg)
        expected = cfg.bfgs_init_scale * jnp.eye(2)
        assert jnp.allclose(state.hessian, expected, atol=1e-10)

    def test_f_val_correct(self, cfg):
        problem = make_unconstrained(n=2)
        x0 = jnp.array([1.0, 0.0])
        p = jnp.array([0.0, 0.0])
        state = init_sqp_state(x0, p, problem, cfg)
        # f = ½‖[1,0] - [0,0]‖² = 0.5
        assert float(state.f_val) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# sqp_solve_scan  (and sqp_solve_single)
# ---------------------------------------------------------------------------

class TestSQPSolveUnconstrained:
    def test_converges_to_known_optimum(self, cfg):
        problem = make_unconstrained(n=2)
        p = jnp.array([0.5, -0.3])
        x0 = jnp.zeros(2)

        state = sqp_solve_scan(problem, x0, p, cfg)
        # x* = p when p is within bounds
        assert jnp.allclose(state.x, p, atol=1e-3)

    def test_success_flag(self, cfg):
        problem = make_unconstrained(n=2)
        state = sqp_solve_scan(problem, jnp.zeros(2), jnp.array([1.0, -1.0]), cfg)
        assert bool(state.converged)

    def test_while_loop_matches_scan(self, cfg):
        """scan and while_loop variants must agree on converged solutions."""
        problem = make_unconstrained(n=2)
        p = jnp.array([0.3, -0.7])
        x0 = jnp.zeros(2)

        state_scan = sqp_solve_scan(problem, x0, p, cfg)
        state_while = sqp_solve_single(problem, x0, p, cfg)
        assert jnp.allclose(state_scan.x, state_while.x, atol=1e-4)

    def test_jit_compatible(self, cfg):
        problem = make_unconstrained(n=2)
        solve_fn = jax.jit(lambda x0, p: sqp_solve_scan(problem, x0, p, cfg))
        state = solve_fn(jnp.zeros(2), jnp.array([1.0, 1.0]))
        assert jnp.all(jnp.isfinite(state.x))


class TestSQPSolveConstrained:
    def test_satisfies_constraint(self, cfg):
        """min ½‖x‖² s.t. x[0]+x[1] >= 1 → x*=[0.5, 0.5]."""
        problem = make_constrained()
        p = jnp.array([1.0])
        x0 = jnp.array([0.0, 0.0])

        state = sqp_solve_scan(problem, x0, p, cfg)
        # Constraint: x[0] + x[1] >= p[0] = 1
        assert float(state.x[0] + state.x[1]) >= 1.0 - 1e-3

    def test_objective_at_constrained_optimum(self, cfg):
        """f* = ½ * (0.5² + 0.5²) = 0.25 for p=1."""
        problem = make_constrained()
        state = sqp_solve_scan(problem, jnp.zeros(2), jnp.array([1.0]), cfg)
        assert float(state.f_val) == pytest.approx(0.25, abs=5e-3)


# ---------------------------------------------------------------------------
# make_solver + vmap batching
# ---------------------------------------------------------------------------

class TestMakeSolverBatched:
    def test_single_solve(self, cfg):
        problem = make_unconstrained(n=2)
        solve_fn = make_solver(problem, cfg)
        state = solve_fn(jnp.zeros(2), jnp.array([1.0, -1.0]))
        assert jnp.all(jnp.isfinite(state.x))

    def test_batched_vmap(self, cfg):
        """Batch of identical problems must give same result as single solve."""
        problem = make_unconstrained(n=2)
        solve_fn = make_solver(problem, cfg)

        N = 5
        params = jnp.stack([jnp.array([0.5, -0.3])] * N)
        x0s = jnp.zeros((N, 2))

        state_batch = jax.vmap(solve_fn)(x0s, params)
        assert state_batch.x.shape == (N, 2)

        # All solutions should be close to each other (same problem)
        assert jnp.max(jnp.std(state_batch.x, axis=0)) < 1e-4

    def test_batched_different_params(self, cfg):
        """Different parameter values should give different solutions."""
        problem = make_unconstrained(n=1)
        solve_fn = make_solver(problem, cfg)

        params = jnp.linspace(-1.0, 1.0, 8).reshape(8, 1)
        x0s = jnp.zeros((8, 1))

        state_batch = jax.vmap(solve_fn)(x0s, params)
        # x* ≈ p for each problem (unconstrained, within bounds)
        assert jnp.allclose(state_batch.x.reshape(8), params.reshape(8), atol=1e-3)

    def test_batch_constrained(self, cfg):
        """Batch of constrained problems: verify all feasible."""
        problem = make_constrained()
        solve_fn = make_solver(problem, cfg)

        params = jnp.linspace(0.5, 2.0, 6).reshape(6, 1)
        x0s = jnp.zeros((6, 2))

        state_batch = jax.vmap(solve_fn)(x0s, params)
        # For each i: x[i,0] + x[i,1] >= params[i]
        sums = state_batch.x[:, 0] + state_batch.x[:, 1]
        assert jnp.all(sums >= params.reshape(6) - 1e-3)


# ---------------------------------------------------------------------------
# state_to_result / batch_state_to_result
# ---------------------------------------------------------------------------

class TestStateToResult:
    def test_result_from_converged_state(self, cfg):
        problem = make_unconstrained(n=2)
        state = sqp_solve_scan(problem, jnp.zeros(2), jnp.array([1.0, 0.0]), cfg)
        result = state_to_result(state, problem)

        assert result.success == bool(state.converged)
        assert jnp.allclose(result.decision_variables, state.x)
        assert result.constraints is None  # unconstrained

    def test_result_includes_constraints(self, cfg):
        problem = make_constrained()
        state = sqp_solve_scan(problem, jnp.zeros(2), jnp.array([1.0]), cfg)
        result = state_to_result(state, problem)
        assert result.constraints is not None
        assert result.constraints.shape == (1,)
