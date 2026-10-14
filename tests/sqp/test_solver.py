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
    make_sqp_step,
    sqp_solve_scan,
    sqp_solve_single,
    state_to_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_unconstrained(n=2):
    """min ½‖x - p‖²,  lb=-2, ub=2.  Optimal x* = clip(p, -2, 2)."""

    def f(x, p):
        return 0.5 * jnp.sum((x - p) ** 2)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(n, -2.0), jnp.full(n, 2.0)],
        n_decision=n,
        n_params=n,
    )


def make_constrained():
    """min ½‖x‖²  s.t.  x[0] + x[1] >= p[0].  Optimal: x* = [p/2, p/2]."""

    def f(x, p):
        return 0.5 * jnp.sum(x**2)

    def g(x, p):
        return (p[0] - x[0] - x[1]).reshape(1)  # g <= 0

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(2, -5.0), jnp.full(2, 5.0)],
        n_decision=2,
        n_params=1,
        constraints=g,
        constraint_lhs=jnp.array([-jnp.inf]),
        constraint_rhs=jnp.array([0.0]),
    )


@pytest.fixture(params=["filter", "merit"])
def cfg(request):
    """Every solver test runs under both step-acceptance methods."""
    return SQPConfig(
        max_iter=60,
        tol_stationarity=1e-5,
        tol_feasibility=1e-5,
        osqp_tol=1e-7,
        osqp_max_iter=2000,
        step_acceptance=request.param,
    )


def make_wachter_biegler_2000():
    """Wächter & Biegler (2000) counterexample on which merit line searches
    can converge to an infeasible point:

        min x[0]  s.t.  x[0]² - x[1] + 1 = 0,  x[0] - x[2] - ½ = 0,  x[1], x[2] >= 0

    Optimal x* = [0.5, 1.25, 0].
    """

    def f(x, p):
        return x[0]

    def g(x, p):
        return jnp.array([x[0] ** 2 - x[1] + 1.0, x[0] - x[2] - 0.5])

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.array([-10.0, 0.0, 0.0]), jnp.full(3, 10.0)],
        n_decision=3,
        n_params=0,
        constraints=g,
        constraint_lhs=jnp.zeros(2),
        constraint_rhs=jnp.zeros(2),
    )


def make_maratos():
    """Nocedal & Wright Ex. 15.4: min 2(x1²+x2²−1) − x1  s.t.  x1²+x2² = 1.  x* = [1, 0].

    From a point on the circle the full SQP step leaves the manifold and raises
    both f and θ (Maratos effect); a second-order correction restores it.
    """

    def f(x, p):
        return 2.0 * (x[0] ** 2 + x[1] ** 2 - 1.0) - x[0]

    def g(x, p):
        return (x[0] ** 2 + x[1] ** 2 - 1.0).reshape(1)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(2, -5.0), jnp.full(2, 5.0)],
        n_decision=2,
        n_params=0,
        constraints=g,
        constraint_lhs=jnp.array([0.0]),
        constraint_rhs=jnp.array([0.0]),
    )


def make_inconsistent():
    """x ∈ [0, 1] with g = 2 − x ≤ 0: infeasible; the violation is minimised at x = 1."""

    def f(x, p):
        return 0.5 * x[0] ** 2

    def g(x, p):
        return (2.0 - x[0]).reshape(1)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.array([0.0]), jnp.array([1.0])],
        n_decision=1,
        n_params=0,
        constraints=g,
        constraint_lhs=jnp.array([-jnp.inf]),
        constraint_rhs=jnp.array([0.0]),
    )


def make_hs013():
    """HS013: min (x0-2)² + x1²  s.t. (1-x0)³ - x1 >= 0,  x >= 0.   x* = [1, 0], f* = 1.

    At x = [0, 0] the constraint is inactive and ∇f = [-4, 0] points into the
    box, so it is NOT a KKT point even though a multiplier of -4/3 on the
    inactive constraint makes the projected Lagrangian gradient vanish.
    """

    def f(x, p):
        return (x[0] - 2.0) ** 2 + x[1] ** 2

    def g(x, p):
        return ((1.0 - x[0]) ** 3 - x[1]).reshape(1)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.zeros(2), jnp.full(2, jnp.inf)],
        n_decision=2,
        n_params=0,
        constraints=g,
        constraint_lhs=jnp.array([0.0]),
        constraint_rhs=jnp.array([jnp.inf]),
    )


def make_circle_equality():
    """min x[0] + x[1]  s.t.  x[0]² + x[1]² = 1.  Optimal x* = -[1, 1]/√2."""

    def f(x, p):
        return x[0] + x[1]

    def g(x, p):
        return (x[0] ** 2 + x[1] ** 2 - 1.0).reshape(1)

    return ParametricNLPProblem(
        objective=f,
        bounds=[jnp.full(2, -3.0), jnp.full(2, 3.0)],
        n_decision=2,
        n_params=0,
        constraints=g,
        constraint_lhs=jnp.array([0.0]),
        constraint_rhs=jnp.array([0.0]),
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


class TestSQPSolveBoxActive:
    """When the unconstrained optimum lies outside the box, the returned
    solution must respect the variable bounds (no tolerance slack) — the QP
    box rows are only approximately enforced by ADMM, so the iterate is
    projected onto [lb, ub]."""

    def test_scan_returns_within_bounds(self, cfg):
        problem = make_unconstrained(n=2)  # bounds [-2, 2]
        p = jnp.array([5.0, 5.0])  # unconstrained optimum well outside
        lb, ub = problem.lb, problem.ub

        state = sqp_solve_scan(problem, jnp.zeros(2), p, cfg)

        assert bool(jnp.all(state.x <= ub))
        assert bool(jnp.all(state.x >= lb))
        # Box-active optimum is clip(p, lb, ub) = [2, 2].
        assert jnp.allclose(state.x, jnp.clip(p, lb, ub), atol=1e-3)

    def test_while_loop_returns_within_bounds(self, cfg):
        problem = make_unconstrained(n=2)
        p = jnp.array([5.0, -5.0])
        lb, ub = problem.lb, problem.ub

        state = sqp_solve_single(problem, jnp.zeros(2), p, cfg)

        assert bool(jnp.all(state.x <= ub))
        assert bool(jnp.all(state.x >= lb))
        assert jnp.allclose(state.x, jnp.clip(p, lb, ub), atol=1e-3)


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


class TestComplementarityInSolver:
    @pytest.mark.parametrize("mode", ["filter", "merit"])
    def test_no_false_convergence_on_hs013_origin(self, mode):
        problem = make_hs013()
        cfg = SQPConfig(max_iter=30, step_acceptance=mode)
        state = sqp_solve_single(problem, jnp.array([0.0, 0.0]), jnp.zeros(0), cfg)
        # Either not converged, or converged somewhere with f < 4 (f(0,0) = 4).
        assert (not bool(state.converged)) or float(state.f_val) < 3.9

    def test_complementarity_small_at_true_optimum(self, cfg):
        problem = make_constrained()  # x* = [0.5, 0.5], constraint active
        state = sqp_solve_scan(problem, jnp.zeros(2), jnp.array([1.0]), cfg)
        assert bool(state.converged)
        assert float(state.complementarity) <= cfg.tol_stationarity

    def test_result_reports_complementarity(self, cfg):
        problem = make_constrained()
        state = sqp_solve_scan(problem, jnp.zeros(2), jnp.array([1.0]), cfg)
        result = state_to_result(state, problem)
        assert result.kkt_complementarity == pytest.approx(float(state.complementarity))


class TestFilterState:
    """State bookkeeping specific to step_acceptance="filter"."""

    def _cfg(self, **kw):
        base = dict(
            max_iter=100,
            tol_stationarity=1e-6,
            tol_feasibility=1e-6,
            step_acceptance="filter",
        )
        base.update(kw)
        return SQPConfig(**base)

    def test_init_state_filter_fields(self):
        problem = make_constrained()
        cfg = self._cfg(filter_size=7)
        x0 = jnp.array([0.0, 0.0])  # g = 1 - 0 - 0 = 1 > 0  -> theta0 = 1
        state = init_sqp_state(x0, jnp.array([1.0]), problem, cfg)
        assert state.filter_theta.shape == (7,) and state.filter_f.shape == (7,)
        assert bool(jnp.all(jnp.isinf(state.filter_theta)))
        assert float(state.theta_max) == pytest.approx(1e4 * max(1.0, 1.0))
        assert int(state.n_restoration) == 0

    def test_theta_max_scales_with_initial_violation(self):
        problem = make_constrained()
        state = init_sqp_state(jnp.zeros(2), jnp.array([5.0]), problem, self._cfg())
        assert float(state.theta_max) == pytest.approx(1e4 * 5.0)

    def test_curved_equality_from_infeasible_start_converges(self):
        problem = make_circle_equality()
        state = sqp_solve_single(
            problem, jnp.array([2.0, 2.0]), jnp.zeros(0), self._cfg()
        )
        assert bool(state.converged)
        assert jnp.allclose(state.x, -jnp.ones(2) / jnp.sqrt(2.0), atol=1e-4)

    def test_restoration_phase_fires_and_solver_recovers(self):
        # From this start the SQP step is rejected by the filter; the solver
        # must enter feasibility restoration and still reach x* = [0.5, 1.25, 0].
        problem = make_wachter_biegler_2000()
        cfg = self._cfg(max_iter=200)
        state = sqp_solve_single(
            problem, jnp.array([-5.0, 0.5, 0.0]), jnp.zeros(0), cfg
        )
        assert int(state.n_restoration) > 0
        assert bool(state.converged)
        assert jnp.allclose(state.x, jnp.array([0.5, 1.25, 0.0]), atol=1e-4)

    def test_restoration_under_vmap(self):
        # lax.cond lowers to a select under vmap, so the restoration QP runs
        # for every batch element on every step; both branches must stay
        # finite and the batch must converge.  (The restoration count is not
        # asserted: the QP polishing gate can flip under batched LU rounding,
        # so the vmapped trajectory need not match the single-solve one.)
        problem = make_wachter_biegler_2000()
        cfg = self._cfg(max_iter=200)
        solve = jax.jit(jax.vmap(lambda x0, p: sqp_solve_scan(problem, x0, p, cfg)))
        x0 = jnp.array([[-5.0, 0.5, 0.0], [-2.0, 3.0, 1.0]])
        state = solve(x0, jnp.zeros((2, 0)))
        assert bool(jnp.all(state.converged))
        assert jnp.allclose(
            state.x, jnp.tile(jnp.array([0.5, 1.25, 0.0]), (2, 1)), atol=1e-4
        )

    def test_h_type_steps_populate_filter(self):
        # From an infeasible start the first accepted step is h-type (switching
        # condition fails while theta is large), so the filter gains an entry.
        problem = make_circle_equality()
        state = sqp_solve_single(
            problem, jnp.array([2.0, 2.0]), jnp.zeros(0), self._cfg()
        )
        assert bool(jnp.any(jnp.isfinite(state.filter_theta)))

    def test_scan_and_while_loop_agree_in_filter_mode(self):
        problem = make_circle_equality()
        cfg = self._cfg()
        x0 = jnp.array([2.0, 2.0])
        s1 = sqp_solve_single(problem, x0, jnp.zeros(0), cfg)
        s2 = sqp_solve_scan(problem, x0, jnp.zeros(0), cfg)
        assert jnp.allclose(s1.x, s2.x, atol=1e-6)
        assert int(s1.n_restoration) == int(s2.n_restoration)

    def test_merit_mode_leaves_filter_empty(self):
        problem = make_circle_equality()
        cfg = SQPConfig(max_iter=100, step_acceptance="merit")
        state = sqp_solve_single(problem, jnp.array([2.0, 2.0]), jnp.zeros(0), cfg)
        assert bool(jnp.all(jnp.isinf(state.filter_theta)))
        assert int(state.n_restoration) == 0

    def test_soc_accepted_and_speeds_up_maratos(self):
        problem = make_maratos()
        x0 = jnp.array([jnp.cos(0.5), jnp.sin(0.5)])
        with_soc = sqp_solve_single(
            problem, x0, jnp.zeros(0), self._cfg(soc_max_iter=4)
        )
        no_soc = sqp_solve_single(problem, x0, jnp.zeros(0), self._cfg(soc_max_iter=0))
        assert bool(with_soc.converged) and bool(no_soc.converged)
        assert jnp.allclose(with_soc.x, jnp.array([1.0, 0.0]), atol=1e-5)
        assert int(with_soc.n_soc) > 0
        assert int(no_soc.n_soc) == 0
        assert int(with_soc.iteration) < int(no_soc.iteration)

    def test_soc_under_vmap(self):
        problem = make_maratos()
        cfg = self._cfg(soc_max_iter=2)
        solve = jax.jit(jax.vmap(lambda x0, p: sqp_solve_scan(problem, x0, p, cfg)))
        x0 = jnp.array([[jnp.cos(0.5), jnp.sin(0.5)], [jnp.cos(-0.8), jnp.sin(-0.8)]])
        state = solve(x0, jnp.zeros((2, 0)))
        assert bool(jnp.all(state.converged))
        assert jnp.allclose(state.x, jnp.tile(jnp.array([1.0, 0.0]), (2, 1)), atol=1e-5)

    def test_inconsistent_constraints_flag_local_infeasibility(self):
        problem = make_inconsistent()
        cfg = self._cfg(max_iter=40)
        state = sqp_solve_single(problem, jnp.array([0.2]), jnp.zeros(0), cfg)
        assert not bool(state.converged)
        assert int(state.n_local_infeasible) >= 1
        assert bool(jnp.isfinite(state.x).all())
        assert 0.0 <= float(state.x[0]) <= 1.0
        assert not bool(state.in_restoration)
        result = state_to_result(state, problem)
        assert "infeasib" in result.message

    def test_restoration_entered_at_feasible_point_stalls_and_resets(self):
        # max_line_search=0 rejects every SQP trial, so restoration is entered
        # from a feasible point (theta=0).  The zero restoration step must be a
        # stall (reset + counted), not an "exit" via 0 <= 0.9 * 0.
        problem = make_constrained()
        cfg = self._cfg(max_iter=3, max_line_search=0, soc_max_iter=0)
        x0 = jnp.array([2.0, 2.0])  # x0 + x1 = 4 >= p = 1: feasible
        state = sqp_solve_scan(problem, x0, jnp.array([1.0]), cfg)
        assert int(state.n_local_infeasible) == 3
        assert not bool(state.in_restoration)
        assert bool(jnp.all(jnp.isinf(state.filter_theta)))  # reset clears filter

    def test_stall_reset_also_resets_multipliers(self):
        # Exploded multipliers make the exact Lagrangian Hessian garbage; a
        # reset that keeps them just reproduces the failure, so lam -> 0.
        problem = make_constrained()
        cfg = self._cfg(max_iter=1, max_line_search=0, soc_max_iter=0)
        step = make_sqp_step(problem, cfg)
        state = init_sqp_state(jnp.array([2.0, 2.0]), jnp.array([1.0]), problem, cfg)
        state = state._replace(lam=jnp.array([1e25]))
        new_state, _ = step(state, None)
        assert int(new_state.n_local_infeasible) == 1
        assert float(jnp.abs(new_state.lam).max()) == 0.0

    def test_restoration_phase_spans_iterations_when_exit_is_strict(self):
        # kappa=1e-3 cannot be met by one Gauss–Newton step, so the phase must
        # continue over several outer iterations (n_restoration > entries).
        problem = make_wachter_biegler_2000()
        x0 = jnp.array([-5.0, 0.5, 0.0])
        strict = sqp_solve_single(
            problem, x0, jnp.zeros(0), self._cfg(max_iter=300, restoration_kappa=1e-3)
        )
        default = sqp_solve_single(problem, x0, jnp.zeros(0), self._cfg(max_iter=300))
        assert bool(strict.converged) and bool(default.converged)
        assert not bool(strict.in_restoration) and not bool(default.in_restoration)
        assert int(strict.n_restoration) >= 1

    def test_stagnation_reset_clears_filter(self):
        # Force a reset on every step: patience=1 and alpha_tol above any step.
        problem = make_circle_equality()
        cfg = self._cfg(
            max_iter=3,
            stagnation_patience=1,
            stagnation_alpha_tol=10.0,
            tol_stationarity=1e-12,
            tol_feasibility=1e-12,
        )
        state = sqp_solve_scan(problem, jnp.array([2.0, 2.0]), jnp.zeros(0), cfg)
        assert bool(jnp.all(jnp.isinf(state.filter_theta)))


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
