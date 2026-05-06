"""Tests for septal.jax.sqp.schema."""

import jax.numpy as jnp
import pytest

from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig, SQPResult, SQPState


class TestParametricNLPProblem:
    def test_construction_minimal(self):
        def f(x, p):
            return jnp.sum(x**2)

        prob = ParametricNLPProblem(
            objective=f,
            bounds=[jnp.zeros(3), jnp.ones(3)],
            n_decision=3,
            n_params=2,
        )
        assert prob.n_decision == 3
        assert prob.n_params == 2
        assert prob.n_constraints == 0
        assert not prob.has_constraints

    def test_n_constraints_inferred_from_rhs(self):
        def f(x, p):
            return jnp.sum(x**2)

        def g(x, p):
            return x[:2]

        prob = ParametricNLPProblem(
            objective=f,
            bounds=[jnp.zeros(3), jnp.ones(3)],
            n_decision=3,
            n_params=1,
            constraints=g,
            constraint_lhs=jnp.array([-jnp.inf, -jnp.inf]),
            constraint_rhs=jnp.zeros(2),
        )
        assert prob.n_constraints == 2
        assert prob.has_constraints

    def test_bounds_properties(self):
        lb = jnp.array([-1.0, -2.0])
        ub = jnp.array([1.0, 2.0])

        def f(x, p):
            return jnp.sum(x)

        prob = ParametricNLPProblem(
            objective=f, bounds=[lb, ub], n_decision=2, n_params=0
        )
        assert jnp.allclose(prob.lb, lb)
        assert jnp.allclose(prob.ub, ub)

    def test_n_constraints_explicit(self):
        def f(x, p):
            return jnp.sum(x)

        prob = ParametricNLPProblem(
            objective=f,
            bounds=[jnp.zeros(2), jnp.ones(2)],
            n_decision=2,
            n_params=1,
            n_constraints=5,
        )
        assert prob.n_constraints == 5


class TestSQPConfig:
    def test_defaults(self):
        cfg = SQPConfig()
        assert cfg.max_iter == 100
        assert cfg.tol_stationarity == 1e-6
        assert cfg.tol_feasibility == 1e-6
        assert 0 < cfg.line_search_beta < 1
        assert cfg.osqp_max_iter > 0

    def test_custom_values(self):
        cfg = SQPConfig(max_iter=200, tol_stationarity=1e-8)
        assert cfg.max_iter == 200
        assert cfg.tol_stationarity == 1e-8


class TestSQPState:
    def test_is_namedtuple_pytree(self):
        """SQPState must be a NamedTuple (JAX pytree)."""
        import jax

        n, n_g = 3, 2
        state = SQPState(
            x=jnp.zeros(n),
            params_p=jnp.zeros(1),
            lam=jnp.zeros(n_g),
            hessian=jnp.eye(n),
            grad_lag=jnp.zeros(n),
            f_val=jnp.array(0.0),
            penalty=jnp.array(1.0),
            merit=jnp.array(0.0),
            stationarity=jnp.array(0.0),
            feasibility=jnp.array(0.0),
            iteration=jnp.array(0, dtype=jnp.int32),
            converged=jnp.array(False),
        )
        # Must be flattenable by JAX
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert len(leaves) == 12
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(reconstructed.x, state.x)

    def test_replace(self):
        n = 2
        state = SQPState(
            x=jnp.zeros(n),
            params_p=jnp.zeros(1),
            lam=jnp.zeros(0),
            hessian=jnp.eye(n),
            grad_lag=jnp.zeros(n),
            f_val=jnp.array(5.0),
            penalty=jnp.array(1.0),
            merit=jnp.array(5.0),
            stationarity=jnp.array(1.0),
            feasibility=jnp.array(0.0),
            iteration=jnp.array(0, dtype=jnp.int32),
            converged=jnp.array(False),
        )
        new_state = state._replace(converged=jnp.array(True))
        assert bool(new_state.converged)
        assert not bool(state.converged)


class TestSQPResult:
    def test_construction(self):
        result = SQPResult(
            success=True,
            objective=jnp.array(0.5),
            decision_variables=jnp.array([0.5, 0.5]),
        )
        assert result.success
        assert result.constraints is None
        assert result.message == ""
