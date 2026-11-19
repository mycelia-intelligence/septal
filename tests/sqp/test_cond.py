"""Tests for septal.jax.sqp._cond and its use in the SQP step (issue #4)."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import solve_triangular

from septal.jax.sqp._cond import batch_aware_cond
from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig
from septal.jax.sqp.solver import (
    init_sqp_state,
    make_sqp_step,
    sqp_solve_scan,
    sqp_solve_single,
)

# Matrix size and batch size are distinct primes so a per-lane copy of the
# matrix has an unmistakable shape in the compiled HLO.
N, B = 53, 7
PER_LANE = re.compile(rf"f64\[{B},(?:1,)?{N},{N}\]")


def _hlo(fn, *args):
    return jax.jit(fn).lower(*args).compile().as_text()


# ---------------------------------------------------------------------------
# batch_aware_cond
# ---------------------------------------------------------------------------


class TestBatchAwareCond:
    M = jnp.arange(float(N * N)).reshape(N, N) / (N * N)

    def _both(self, pred, x):
        """Same cond through the helper and through lax.cond."""
        M = self.M

        def t(x):
            return M @ x, jnp.sum(x)

        def f(x):
            return -x, jnp.zeros(())

        return batch_aware_cond(pred, t, f, x), jax.lax.cond(pred, t, f, x)

    @pytest.mark.parametrize("pred", [True, False])
    def test_matches_lax_cond_unbatched(self, pred):
        x = jnp.linspace(-1.0, 1.0, N)
        got, want = jax.jit(self._both)(jnp.array(pred), x)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w)

    def test_matches_lax_cond_under_vmap(self):
        xs = jnp.linspace(-1.0, 1.0, B * N).reshape(B, N)
        preds = jnp.arange(B) % 2 == 0
        got, want = jax.jit(jax.vmap(self._both))(preds, xs)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            np.testing.assert_allclose(g, w)

    def test_batched_operands_unbatched_predicate(self):
        xs = jnp.linspace(-1.0, 1.0, B * N).reshape(B, N)
        for pred in (True, False):
            got, want = jax.vmap(self._both, in_axes=(None, 0))(jnp.array(pred), xs)
            for g, w in zip(got, want):
                assert g.shape == w.shape
                np.testing.assert_allclose(g, w)

    def test_batched_predicate_unbatched_operands(self):
        x = jnp.linspace(-1.0, 1.0, N)
        preds = jnp.arange(B) % 2 == 0
        got, want = jax.vmap(self._both, in_axes=(0, None))(preds, x)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            np.testing.assert_allclose(g, w)

    def test_nested_vmap(self):
        xs = jnp.linspace(-1.0, 1.0, 3 * B * N).reshape(3, B, N)
        preds = (jnp.arange(3 * B) % 3 == 0).reshape(3, B)
        got, want = jax.vmap(jax.vmap(self._both))(preds, xs)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            np.testing.assert_allclose(g, w)

    @pytest.mark.parametrize("jit", [False, True])
    def test_closed_over_batched_values_are_hoisted(self, jit):
        """Per-lane float and int values captured by a branch, not passed in."""

        def fn(pred, x, scale, k):
            return batch_aware_cond(pred, lambda x: scale * x + k, lambda x: x, x)

        xs = jnp.ones((B, 3))
        scales = jnp.arange(1.0, B + 1.0)
        ks = jnp.arange(B)
        preds = jnp.arange(B) % 2 == 0
        vfn = jax.jit(jax.vmap(fn)) if jit else jax.vmap(fn)
        out = vfn(preds, xs, scales, ks)
        want = jnp.where(preds[:, None], scales[:, None] * xs + ks[:, None], xs)
        np.testing.assert_allclose(out, want)

    def test_closed_over_batched_value_inside_loop_body(self):
        """As in the solver: the cond sits in a while_loop body and a branch
        closes over a value batched by a vmap outside the loop."""

        def fn(scale, x):
            def body(c):
                i, x = c
                x = batch_aware_cond(i % 2 == 0, lambda x: scale * x, lambda x: x, x)
                return i + 1, x

            return jax.lax.while_loop(lambda c: c[0] < 4, body, (0, x))[1]

        scales = jnp.arange(1.0, B + 1.0)
        out = jax.vmap(fn, in_axes=(0, None))(scales, jnp.ones(3))
        np.testing.assert_allclose(out, (scales**2)[:, None] * jnp.ones(3))

    def test_forward_mode_derivative(self):
        def fn(x):
            return batch_aware_cond(
                x[0] > 0, lambda x: jnp.sum(x**2), lambda x: jnp.sum(x), x
            )

        xs = jnp.array([[1.0, 2.0], [-1.0, 2.0]])
        jac = jax.vmap(jax.jacfwd(fn))(xs)
        np.testing.assert_allclose(jac, jnp.array([[2.0, 4.0], [1.0, 1.0]]))

    def test_constant_is_not_copied_per_lane(self):
        """The property lax.cond lacks: M stays (N, N) under a batched predicate."""
        M = self.M
        xs = jnp.ones((B, N))
        preds = jnp.arange(B) % 2 == 0

        def with_helper(pred, x):
            return batch_aware_cond(
                pred, lambda x: solve_triangular(M, x), lambda x: x, x
            )

        def with_lax(pred, x):
            return jax.lax.cond(pred, lambda x: solve_triangular(M, x), lambda x: x, x)

        assert PER_LANE.search(_hlo(jax.vmap(with_lax), preds, xs))  # the bug
        assert not PER_LANE.search(_hlo(jax.vmap(with_helper), preds, xs))


# ---------------------------------------------------------------------------
# SQP step: closed-over constants stay unbatched under vmap
# ---------------------------------------------------------------------------


def make_gp_variance_problem():
    """min ‖x − p‖²  s.t.  GP posterior variance(x) ≥ 0.2.

    The constraint closes over an N×N Cholesky factor, as in the issue.
    """
    X = jax.random.normal(jax.random.PRNGKey(0), (N, 2))
    K = jnp.exp(-0.5 * jnp.sum((X[:, None] - X[None]) ** 2, -1)) + 0.1 * jnp.eye(N)
    L = jnp.linalg.cholesky(K)

    def variance(x):
        k = jnp.exp(-0.5 * jnp.sum((X - x) ** 2, -1))
        v = solve_triangular(L, k, lower=True)
        return 1.0 - v @ v

    return ParametricNLPProblem(
        objective=lambda x, p: jnp.sum((x - p) ** 2),
        bounds=[-2.0 * jnp.ones(2), 2.0 * jnp.ones(2)],
        n_decision=2,
        n_params=2,
        constraints=lambda x, p: jnp.atleast_1d(variance(x)),
        constraint_lhs=jnp.array([0.2]),
        constraint_rhs=jnp.array([jnp.inf]),
    )


def make_maratos():
    """Nocedal & Wright Ex. 15.4: min 2(x1²+x2²−1) − x1  s.t.  x1²+x2² = 1."""
    return ParametricNLPProblem(
        objective=lambda x, p: 2.0 * (x[0] ** 2 + x[1] ** 2 - 1.0) - x[0],
        bounds=[jnp.full(2, -5.0), jnp.full(2, 5.0)],
        n_decision=2,
        n_params=0,
        constraints=lambda x, p: (x[0] ** 2 + x[1] ** 2 - 1.0).reshape(1),
        constraint_lhs=jnp.array([0.0]),
        constraint_rhs=jnp.array([0.0]),
    )


def make_wachter_biegler_2000():
    """min x1  s.t.  x1² − x2 + 1 = 0,  x1 − x3 − ½ = 0,  x2, x3 ≥ 0."""
    return ParametricNLPProblem(
        objective=lambda x, p: x[0],
        bounds=[jnp.array([-10.0, 0.0, 0.0]), jnp.full(3, 10.0)],
        n_decision=3,
        n_params=0,
        constraints=lambda x, p: jnp.array([x[0] ** 2 - x[1] + 1.0, x[0] - x[2] - 0.5]),
        constraint_lhs=jnp.zeros(2),
        constraint_rhs=jnp.zeros(2),
    )


class TestSolverKeepsConstantsUnbatched:
    x0 = jnp.zeros((B, 2))
    ps = jax.random.uniform(jax.random.PRNGKey(1), (B, 2), minval=-1.5, maxval=1.5)

    @pytest.mark.parametrize("solve", [sqp_solve_single, sqp_solve_scan])
    @pytest.mark.parametrize("soc_max_iter", [4, 0])
    def test_no_per_lane_copy_in_hlo(self, solve, soc_max_iter):
        problem = make_gp_variance_problem()
        cfg = SQPConfig(max_iter=20, soc_max_iter=soc_max_iter)

        def one(x0, p):
            return solve(problem, x0=x0, p=p, cfg=cfg).x

        hlo = _hlo(jax.vmap(one), self.x0, self.ps)
        assert PER_LANE.findall(hlo) == []

    def test_vmap_matches_sequential(self):
        problem = make_gp_variance_problem()
        cfg = SQPConfig(max_iter=100)

        def one(x0, p):
            res = sqp_solve_single(problem, x0=x0, p=p, cfg=cfg)
            return res.x, res.converged

        xs, conv = jax.jit(jax.vmap(one))(self.x0, self.ps)
        single = jax.jit(one)
        # Which lanes converge within max_iter varies with the jax / BLAS
        # build, so compare the lanes that converged on both paths.
        n_compared = 0
        for i in range(B):
            x_i, conv_i = single(self.x0[i], self.ps[i])
            if bool(conv[i]) and bool(conv_i):
                np.testing.assert_allclose(xs[i], x_i, atol=1e-5)
                n_compared += 1
        assert n_compared > 0

    def test_soc_fires_in_some_lanes_only(self):
        """Maratos start needs a SOC; the lane starting at the optimum does not."""
        problem = make_maratos()
        cfg = SQPConfig(step_acceptance="filter", use_exact_hessian=True)
        step = make_sqp_step(problem, cfg)
        ang = jnp.pi / 4
        x0s = jnp.array([[jnp.cos(ang), jnp.sin(ang)], [1.0, 0.0]])

        def one(x0):
            state = init_sqp_state(x0, jnp.zeros(0), problem, cfg)
            # exact multiplier λ* = 1.5 makes the first QP step the Maratos step
            state = state._replace(lam=jnp.array([-1.5]))
            new, _ = step(state, None)
            return new.n_soc, new.x

        n_soc_b, x_b = jax.jit(jax.vmap(one))(x0s)
        for i in range(2):
            n_soc_i, x_i = jax.jit(one)(x0s[i])
            assert int(n_soc_b[i]) == int(n_soc_i)
            np.testing.assert_allclose(x_b[i], x_i, atol=1e-6)

    def test_restoration_fires_under_vmap(self):
        problem = make_wachter_biegler_2000()
        # A huge gamma_alpha rejects every filter trial, forcing restoration.
        cfg = SQPConfig(step_acceptance="filter", filter_gamma_alpha=1e9)
        step = make_sqp_step(problem, cfg)
        x0s = jnp.array([[-2.0, 1.0, 1.0], [-1.0, 2.0, 0.5], [0.5, 3.0, 1.0]])

        def one(x0):
            state = init_sqp_state(x0, jnp.zeros(0), problem, cfg)
            new, _ = step(state, None)
            return new.n_restoration, new.feasibility, state.feasibility, new.x

        n_r, feas, feas0, x_b = jax.jit(jax.vmap(one))(x0s)
        assert np.all(np.asarray(n_r) == 1)
        assert np.all(np.asarray(feas) < np.asarray(feas0))
        for i in range(len(x0s)):
            np.testing.assert_allclose(x_b[i], jax.jit(one)(x0s[i])[3], atol=1e-6)
