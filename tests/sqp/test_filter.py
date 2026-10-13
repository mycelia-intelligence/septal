"""Tests for septal.jax.sqp.filter (Wächter & Biegler 2005 line-search filter)."""

import jax
import jax.numpy as jnp
import pytest

from septal.jax.sqp.filter import (
    augment_filter,
    empty_filter,
    filter_acceptable,
    filter_line_search,
    judge_trial,
    min_step_size,
    restoration_direction,
    restoration_line_search,
    soc_correction,
    switching_condition,
)
from septal.jax.sqp.line_search import constraint_violation
from septal.jax.sqp.schema import SQPConfig

INF = jnp.inf


def _filter(entries, size=4):
    """Build a fixed-size filter from a list of (theta, f) pairs."""
    th, f = empty_filter(size)
    for i, (t, v) in enumerate(entries):
        th = th.at[i].set(t)
        f = f.at[i].set(v)
    return th, f


# ---------------------------------------------------------------------------
# empty_filter
# ---------------------------------------------------------------------------


class TestEmptyFilter:
    def test_shape_and_inf(self):
        th, f = empty_filter(5)
        assert th.shape == (5,) and f.shape == (5,)
        assert bool(jnp.all(jnp.isinf(th))) and bool(jnp.all(jnp.isinf(f)))
        assert th.dtype == jnp.float64


# ---------------------------------------------------------------------------
# filter_acceptable
# ---------------------------------------------------------------------------


class TestFilterAcceptable:
    def test_empty_filter_accepts(self):
        th, f = empty_filter(4)
        assert bool(filter_acceptable(jnp.array(0.5), jnp.array(3.0), th, f, 1e4))

    def test_dominated_point_rejected(self):
        th, f = _filter([(1.0, 2.0)])
        # theta >= 1 and f >= 2 -> inside the forbidden region
        assert not bool(filter_acceptable(jnp.array(1.0), jnp.array(2.0), th, f, 1e4))
        assert not bool(filter_acceptable(jnp.array(1.5), jnp.array(2.5), th, f, 1e4))

    def test_better_theta_accepted(self):
        th, f = _filter([(1.0, 2.0)])
        assert bool(filter_acceptable(jnp.array(0.9), jnp.array(5.0), th, f, 1e4))

    def test_better_f_accepted(self):
        th, f = _filter([(1.0, 2.0)])
        assert bool(filter_acceptable(jnp.array(5.0), jnp.array(1.9), th, f, 1e4))

    def test_any_entry_can_block(self):
        th, f = _filter([(1.0, 2.0), (0.1, 10.0)])
        # Beats first entry on f but dominated by the second
        assert not bool(filter_acceptable(jnp.array(0.2), jnp.array(10.5), th, f, 1e4))

    def test_theta_max_rejects(self):
        th, f = empty_filter(4)
        assert not bool(
            filter_acceptable(jnp.array(10.0), jnp.array(-1e9), th, f, 10.0)
        )
        assert bool(filter_acceptable(jnp.array(9.9), jnp.array(-1e9), th, f, 10.0))

    def test_inf_slots_never_produce_nan(self):
        th, f = empty_filter(3)
        out = filter_acceptable(jnp.array(0.0), jnp.array(-jnp.inf), th, f, 1e4)
        assert bool(out)
        out = filter_acceptable(jnp.array(0.0), jnp.array(jnp.inf), th, f, 1e4)
        # f = +inf is never better than anything but theta=0 < inf passes
        assert bool(out)

    def test_jit_and_vmap(self):
        th, f = _filter([(1.0, 2.0)])
        fn = jax.jit(lambda t, v: filter_acceptable(t, v, th, f, 1e4))
        assert not bool(fn(jnp.array(1.5), jnp.array(2.5)))
        batched = jax.vmap(fn)(jnp.array([1.5, 0.5]), jnp.array([2.5, 2.5]))
        assert batched.tolist() == [False, True]


# ---------------------------------------------------------------------------
# augment_filter
# ---------------------------------------------------------------------------


class TestAugmentFilter:
    def test_inserts_with_margins_into_first_empty_slot(self):
        th, f = empty_filter(3)
        th2, f2 = augment_filter(th, f, jnp.array(2.0), jnp.array(10.0), 0.1, 0.5)
        assert float(th2[0]) == pytest.approx((1 - 0.1) * 2.0)
        assert float(f2[0]) == pytest.approx(10.0 - 0.5 * 2.0)
        assert bool(jnp.all(jnp.isinf(th2[1:])))

    def test_second_insert_goes_to_next_slot(self):
        th, f = _filter([(1.0, 2.0)])
        th2, f2 = augment_filter(th, f, jnp.array(0.5), jnp.array(5.0), 0.0, 0.0)
        assert float(th2[0]) == 1.0 and float(f2[0]) == 2.0
        assert float(th2[1]) == 0.5 and float(f2[1]) == 5.0

    def test_dominated_entries_removed(self):
        # (3, 3) is dominated by the new entry (1, 1); (0.5, 10) is not.
        th, f = _filter([(3.0, 3.0), (0.5, 10.0)])
        th2, f2 = augment_filter(th, f, jnp.array(1.0), jnp.array(1.0), 0.0, 0.0)
        kept = [(float(t), float(v)) for t, v in zip(th2, f2) if jnp.isfinite(t)]
        assert (3.0, 3.0) not in kept
        assert (0.5, 10.0) in kept
        assert (1.0, 1.0) in kept

    def test_evicts_largest_theta_when_full(self):
        th, f = _filter([(1.0, 9.0), (5.0, 1.0), (2.0, 4.0)], size=3)
        th2, f2 = augment_filter(th, f, jnp.array(0.1), jnp.array(100.0), 0.0, 0.0)
        kept = [(float(t), float(v)) for t, v in zip(th2, f2)]
        assert (5.0, 1.0) not in kept
        assert (0.1, 100.0) in kept
        assert (1.0, 9.0) in kept and (2.0, 4.0) in kept

    def test_new_point_is_then_blocked(self):
        th, f = empty_filter(3)
        th2, f2 = augment_filter(th, f, jnp.array(2.0), jnp.array(10.0), 1e-5, 1e-5)
        assert not bool(
            filter_acceptable(jnp.array(2.0), jnp.array(10.0), th2, f2, 1e4)
        )

    def test_jit_and_vmap_shapes(self):
        th, f = empty_filter(4)
        fn = jax.jit(lambda t, v: augment_filter(th, f, t, v, 1e-5, 1e-5))
        th2, f2 = jax.vmap(fn)(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
        assert th2.shape == (2, 4) and f2.shape == (2, 4)


# ---------------------------------------------------------------------------
# switching_condition / min_step_size
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return SQPConfig(line_search_beta=0.5, line_search_alpha0=1.0, max_line_search=30)


class TestSwitchingCondition:
    def test_feasible_descent_holds(self, cfg):
        assert bool(
            switching_condition(jnp.array(1.0), jnp.array(-1.0), jnp.array(0.0), cfg)
        )

    def test_non_descent_fails(self, cfg):
        assert not bool(
            switching_condition(jnp.array(1.0), jnp.array(0.5), jnp.array(0.0), cfg)
        )

    def test_infeasible_small_descent_fails(self, cfg):
        # alpha * (-g)^{s_f} = 1 > delta * theta^{s_theta} = 1 is False
        assert not bool(
            switching_condition(jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0), cfg)
        )

    def test_infeasible_large_descent_holds(self, cfg):
        # 2^2.3 ≈ 4.92 > 1
        assert bool(
            switching_condition(jnp.array(1.0), jnp.array(-2.0), jnp.array(1.0), cfg)
        )

    def test_shrinking_alpha_breaks_it(self, cfg):
        assert not bool(
            switching_condition(jnp.array(0.1), jnp.array(-2.0), jnp.array(1.0), cfg)
        )


class TestMinStepSize:
    def test_descent_infeasible(self, cfg):
        # gamma_alpha * min(gamma_theta, gamma_f*theta/(-g), delta*theta^s/(-g)^s)
        a = min_step_size(jnp.array(-1.0), jnp.array(1.0), cfg)
        assert float(a) == pytest.approx(0.05 * 1e-5)

    def test_non_descent(self, cfg):
        a = min_step_size(jnp.array(0.0), jnp.array(1.0), cfg)
        assert float(a) == pytest.approx(0.05 * 1e-5)

    def test_descent_feasible_is_zero(self, cfg):
        a = min_step_size(jnp.array(-1.0), jnp.array(0.0), cfg)
        assert float(a) == pytest.approx(0.0)

    def test_non_descent_feasible_is_finite(self, cfg):
        # theta=0 and grad_f_d=0 must not produce 0/0 = NaN anywhere
        a = min_step_size(jnp.array(0.0), jnp.array(0.0), cfg)
        assert float(a) == pytest.approx(0.05 * 1e-5)
        assert bool(jnp.isfinite(a))


# ---------------------------------------------------------------------------
# filter_line_search
# ---------------------------------------------------------------------------


def _run_ls(
    objective, constraints, lhs, rhs, lb, ub, x, d, cfg, filt=None, theta_max=1e4
):
    n_g = lhs.shape[0]
    p = jnp.zeros(0)
    f_k = jnp.asarray(objective(x, p))
    g_k = constraints(x, p) if n_g > 0 else jnp.zeros(0)
    theta_k = constraint_violation(g_k, lhs, rhs)
    grad_f_d = jax.grad(objective)(x, p) @ d
    th, f = filt if filt is not None else empty_filter(4)
    return filter_line_search(
        x,
        d,
        p,
        theta_k,
        f_k,
        grad_f_d,
        th,
        f,
        jnp.array(theta_max),
        objective,
        constraints,
        lhs,
        rhs,
        lb,
        ub,
        n_g,
        cfg,
    )


class TestFilterLineSearch:
    def test_unconstrained_newton_step_is_f_type(self, cfg):
        def f(x, p):
            return 0.5 * jnp.sum(x**2)

        x = jnp.array([1.0, 1.0])
        alpha, accepted, f_type = _run_ls(
            f,
            None,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.full(2, -5.0),
            jnp.full(2, 5.0),
            x,
            -x,
            cfg,
        )
        assert float(alpha) == 1.0 and bool(accepted) and bool(f_type)

    def test_trial_point_is_clipped_to_box(self, cfg):
        # Without clipping, alpha=1 lands at x=-1.5 and fails Armijo; with
        # clipping it lands at lb=0 and is accepted.
        def f(x, p):
            return 0.5 * jnp.sum(x**2)

        x = jnp.array([1.5])
        alpha, accepted, _ = _run_ls(
            f,
            None,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.array([0.0]),
            jnp.array([5.0]),
            x,
            jnp.array([-3.0]),
            cfg,
        )
        assert float(alpha) == 1.0 and bool(accepted)

    def test_feasibility_improving_ascent_step_is_h_type(self, cfg):
        # f = -x; g = x - 1 <= 0; at x=2 theta=1.  d=-1 raises f but restores feasibility.
        def f(x, p):
            return -x[0]

        def g(x, p):
            return (x[0] - 1.0).reshape(1)

        alpha, accepted, f_type = _run_ls(
            f,
            g,
            jnp.array([-INF]),
            jnp.array([0.0]),
            jnp.array([-5.0]),
            jnp.array([5.0]),
            jnp.array([2.0]),
            jnp.array([-1.0]),
            cfg,
        )
        assert float(alpha) == 1.0 and bool(accepted) and not bool(f_type)

    def test_filter_entry_blocks_otherwise_acceptable_step(self, cfg):
        # Same as above but the filter already forbids theta>=0 & f>=-5 -> everything.
        def f(x, p):
            return -x[0]

        def g(x, p):
            return (x[0] - 1.0).reshape(1)

        filt = _filter([(0.0, -5.0)])
        alpha, accepted, _ = _run_ls(
            f,
            g,
            jnp.array([-INF]),
            jnp.array([0.0]),
            jnp.array([-5.0]),
            jnp.array([5.0]),
            jnp.array([2.0]),
            jnp.array([-1.0]),
            cfg,
            filt=filt,
        )
        assert not bool(accepted) and float(alpha) == 0.0

    def test_all_trials_rejected_returns_zero(self, cfg):
        # d moves away from feasibility and theta_max is tiny -> nothing acceptable.
        def f(x, p):
            return -x[0]

        def g(x, p):
            return (x[0] - 1.0).reshape(1)

        alpha, accepted, f_type = _run_ls(
            f,
            g,
            jnp.array([-INF]),
            jnp.array([0.0]),
            jnp.array([-5.0]),
            jnp.array([5.0]),
            jnp.array([2.0]),
            jnp.array([1.0]),
            cfg,
            theta_max=0.5,
        )
        assert not bool(accepted) and float(alpha) == 0.0 and not bool(f_type)

    def test_backtracks_when_full_step_too_long(self, cfg):
        # f = ½x², d = -2x: alpha=1 overshoots to -x (same f, Armijo fails); alpha=0.5 -> 0.
        def f(x, p):
            return 0.5 * jnp.sum(x**2)

        x = jnp.array([1.0])
        alpha, accepted, f_type = _run_ls(
            f,
            None,
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.array([-5.0]),
            jnp.array([5.0]),
            x,
            -2.0 * x,
            cfg,
        )
        assert float(alpha) == 0.5 and bool(accepted) and bool(f_type)

    def test_jit_and_vmap(self, cfg):
        def f(x, p):
            return 0.5 * jnp.sum(x**2)

        lb, ub = jnp.full(2, -5.0), jnp.full(2, 5.0)
        th, fl = empty_filter(4)

        def run(x):
            return filter_line_search(
                x,
                -x,
                jnp.zeros(0),
                jnp.array(0.0),
                f(x, None),
                -(x @ x),
                th,
                fl,
                jnp.array(1e4),
                f,
                None,
                jnp.zeros(0),
                jnp.zeros(0),
                lb,
                ub,
                0,
                cfg,
            )

        alpha, acc, ft = jax.jit(jax.vmap(run))(jnp.array([[1.0, 1.0], [2.0, -1.0]]))
        assert alpha.shape == (2,) and bool(jnp.all(acc)) and bool(jnp.all(ft))


# ---------------------------------------------------------------------------
# Feasibility restoration
# ---------------------------------------------------------------------------


def _theta(g_fn, x, lhs, rhs):
    return float(constraint_violation(g_fn(x, jnp.zeros(0)), lhs, rhs))


class TestRestorationDirection:
    """g1 = x0 - 1 <= 0 (violated at x0=2), g2 = x1 >= 0 (active, satisfied)."""

    def _setup(self):
        def g(x, p):
            return jnp.array([x[0] - 1.0, x[1]])

        lhs = jnp.array([-INF, 0.0])
        rhs = jnp.array([0.0, INF])
        return g, lhs, rhs

    def _direction(self, x, lb, ub, cfg):
        g, lhs, rhs = self._setup()
        p = jnp.zeros(0)
        g_val = g(x, p)
        jac = jax.jacfwd(g)(x, p)
        return (
            restoration_direction(x, p, g_val, jac, lhs, rhs, lb, ub, 2, 2, cfg),
            g,
            lhs,
            rhs,
        )

    def test_reduces_violation_and_keeps_satisfied_constraint(self, cfg):
        x = jnp.array([2.0, 0.0])
        d, g, lhs, rhs = self._direction(x, jnp.full(2, -5.0), jnp.full(2, 5.0), cfg)
        x_new = x + d
        assert _theta(g, x_new, lhs, rhs) < 1e-3 * _theta(g, x, lhs, rhs)
        assert float(x_new[1]) >= -1e-8  # g2 stays feasible
        assert float(d[1]) == pytest.approx(0.0, abs=1e-6)

    def test_respects_box(self, cfg):
        x = jnp.array([2.0, 0.0])
        lb = jnp.array([1.5, -5.0])
        d, g, lhs, rhs = self._direction(x, lb, jnp.full(2, 5.0), cfg)
        x_new = x + d
        assert float(x_new[0]) >= 1.5 - 1e-8
        assert _theta(g, x_new, lhs, rhs) == pytest.approx(0.5, abs=1e-6)

    def test_violated_equality(self, cfg):
        def g(x, p):
            return (x[0] + x[1] - 1.0).reshape(1)

        lhs = rhs = jnp.array([0.0])
        x = jnp.array([2.0, 2.0])
        p = jnp.zeros(0)
        d = restoration_direction(
            x,
            p,
            g(x, p),
            jax.jacfwd(g)(x, p),
            lhs,
            rhs,
            jnp.full(2, -5.0),
            jnp.full(2, 5.0),
            2,
            1,
            cfg,
        )
        assert _theta(g, x + d, lhs, rhs) < 1e-3
        assert float(d[0]) == pytest.approx(float(d[1]), abs=1e-6)  # min-norm

    def test_jit_vmap(self, cfg):
        g, lhs, rhs = self._setup()
        lb, ub = jnp.full(2, -5.0), jnp.full(2, 5.0)
        p = jnp.zeros(0)

        def run(x):
            return restoration_direction(
                x, p, g(x, p), jax.jacfwd(g)(x, p), lhs, rhs, lb, ub, 2, 2, cfg
            )

        d = jax.jit(jax.vmap(run))(jnp.array([[2.0, 0.0], [3.0, 1.0]]))
        assert d.shape == (2, 2) and bool(jnp.all(jnp.isfinite(d)))


class TestRestorationLineSearch:
    def _problem(self):
        def f(x, p):
            return -x[0]

        def g(x, p):
            return (x[0] - 1.0).reshape(1)

        return (
            f,
            g,
            jnp.array([-INF]),
            jnp.array([0.0]),
            jnp.array([-5.0]),
            jnp.array([5.0]),
        )

    def test_accepts_full_step_that_restores_feasibility(self, cfg):
        f, g, lhs, rhs, lb, ub = self._problem()
        th, fl = empty_filter(4)
        alpha = restoration_line_search(
            jnp.array([2.0]),
            jnp.array([-1.0]),
            jnp.zeros(0),
            jnp.array(1.0),
            th,
            fl,
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            1,
            cfg,
        )
        assert float(alpha) == 1.0

    def test_backtracks_past_overshoot(self, cfg):
        # d=-4: alpha=1 lands at x=-2 (feasible, theta=0) — accepted at alpha=1;
        # use theta_max to force a smaller step instead: not applicable, so test
        # that a step which increases theta is rejected and alpha shrinks to a
        # value that still reduces theta by the required margin.
        f, g, lhs, rhs, lb, ub = self._problem()
        th, fl = empty_filter(4)
        # d=+1 increases violation for every alpha -> nothing acceptable
        alpha = restoration_line_search(
            jnp.array([2.0]),
            jnp.array([1.0]),
            jnp.zeros(0),
            jnp.array(1.0),
            th,
            fl,
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            1,
            cfg,
        )
        assert float(alpha) == 0.0

    def test_filter_can_block_restoration_step(self, cfg):
        f, g, lhs, rhs, lb, ub = self._problem()
        filt = _filter([(0.0, -5.0)])  # forbids everything with f >= -5
        alpha = restoration_line_search(
            jnp.array([2.0]),
            jnp.array([-1.0]),
            jnp.zeros(0),
            jnp.array(1.0),
            filt[0],
            filt[1],
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            1,
            cfg,
        )
        assert float(alpha) == 0.0


# ---------------------------------------------------------------------------
# judge_trial
# ---------------------------------------------------------------------------


class TestJudgeTrial:
    def _judge(
        self, theta_t, f_t, alpha, grad_f_d, theta_k, f_k, cfg, filt=None, theta_max=1e4
    ):
        th, fl = filt if filt is not None else empty_filter(4)
        return judge_trial(
            jnp.array(theta_t),
            jnp.array(f_t),
            jnp.array(alpha),
            jnp.array(grad_f_d),
            jnp.array(theta_k),
            jnp.array(f_k),
            th,
            fl,
            jnp.array(theta_max),
            cfg,
        )

    def test_f_type(self, cfg):
        # feasible, descent, Armijo satisfied
        acc, ft = self._judge(0.0, 0.5, 1.0, -1.0, 0.0, 1.0, cfg)
        assert bool(acc) and bool(ft)

    def test_f_type_armijo_fails_is_rejected_not_demoted(self, cfg):
        # switching holds (theta_k=0, descent) but f increased -> rejected outright
        acc, ft = self._judge(0.0, 1.5, 1.0, -1.0, 0.0, 1.0, cfg)
        assert not bool(acc) and not bool(ft)

    def test_h_type(self, cfg):
        # ascent direction -> switching fails; theta decreased -> h-type
        acc, ft = self._judge(0.5, 2.0, 1.0, 1.0, 1.0, 1.0, cfg)
        assert bool(acc) and not bool(ft)

    def test_blocked_by_filter(self, cfg):
        acc, ft = self._judge(
            0.5, 2.0, 1.0, 1.0, 1.0, 1.0, cfg, filt=_filter([(0.0, -5.0)])
        )
        assert not bool(acc)

    def test_blocked_by_theta_max(self, cfg):
        acc, _ = self._judge(0.5, 2.0, 1.0, 1.0, 1.0, 1.0, cfg, theta_max=0.4)
        assert not bool(acc)


# ---------------------------------------------------------------------------
# Second-order correction (Maratos example, Nocedal & Wright Ex. 15.4)
# ---------------------------------------------------------------------------


def _maratos():
    def f(x, p):
        return 2.0 * (x[0] ** 2 + x[1] ** 2 - 1.0) - x[0]

    def g(x, p):
        return (x[0] ** 2 + x[1] ** 2 - 1.0).reshape(1)

    lhs = rhs = jnp.array([0.0])
    lb, ub = jnp.full(2, -5.0), jnp.full(2, 5.0)
    return f, g, lhs, rhs, lb, ub


class TestSOCCorrection:
    def _setup(self, t=0.5):
        f, g, lhs, rhs, lb, ub = _maratos()
        x = jnp.array([jnp.cos(t), jnp.sin(t)])
        p = jnp.zeros(0)
        grad_f = jax.grad(f)(x, p)
        g_val = g(x, p)
        jac = jax.jacfwd(g)(x, p)
        d = jnp.array([x[1] ** 2, -x[0] * x[1]])  # SQP step with H = I
        return f, g, lhs, rhs, lb, ub, x, p, grad_f, g_val, jac, d

    def test_full_step_rejected_without_soc(self, cfg):
        f, g, lhs, rhs, lb, ub, x, p, grad_f, g_val, jac, d = self._setup()
        th, fl = empty_filter(4)
        alpha, accepted, _ = filter_line_search(
            x,
            d,
            p,
            jnp.array(0.0),
            f(x, p),
            grad_f @ d,
            th,
            fl,
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            1,
            cfg,
        )
        assert bool(accepted) and float(alpha) < 1.0  # Maratos: only a short step

    def test_soc_step_accepted_and_closer_to_optimum(self, cfg):
        f, g, lhs, rhs, lb, ub, x, p, grad_f, g_val, jac, d = self._setup()
        th, fl = empty_filter(4)
        accepted, f_type, d_soc, lam_soc = soc_correction(
            x,
            d,
            p,
            jnp.array(0.0),
            f(x, p),
            grad_f @ d,
            g_val,
            jac,
            jnp.eye(2),
            grad_f,
            jnp.zeros(1),
            th,
            fl,
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            2,
            1,
            cfg,
        )
        assert bool(accepted) and bool(f_type)
        x_star = jnp.array([1.0, 0.0])
        x_full = x + d
        x_soc = x + d_soc
        assert float(jnp.linalg.norm(x_soc - x_star)) < 0.2 * float(
            jnp.linalg.norm(x_full - x_star)
        )
        theta_full = constraint_violation(g(x_full, p), lhs, rhs)
        theta_soc = constraint_violation(g(x_soc, p), lhs, rhs)
        assert float(theta_soc) < 0.1 * float(theta_full)
        assert lam_soc.shape == (1,) and bool(jnp.isfinite(lam_soc).all())

    def test_soc_disabled_by_filter_block(self, cfg):
        # A filter that forbids everything makes the SOC trial unacceptable.
        f, g, lhs, rhs, lb, ub, x, p, grad_f, g_val, jac, d = self._setup()
        filt = _filter([(0.0, -1e9)])
        accepted, f_type, d_soc, _ = soc_correction(
            x,
            d,
            p,
            jnp.array(0.0),
            f(x, p),
            grad_f @ d,
            g_val,
            jac,
            jnp.eye(2),
            grad_f,
            jnp.zeros(1),
            filt[0],
            filt[1],
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            2,
            1,
            cfg,
        )
        assert not bool(accepted) and not bool(f_type)
        assert bool(jnp.isfinite(d_soc).all())

    def test_jit_vmap(self, cfg):
        f, g, lhs, rhs, lb, ub, x, p, grad_f, g_val, jac, d = self._setup()
        th, fl = empty_filter(4)

        def run(x, d, grad_f, g_val, jac):
            return soc_correction(
                x,
                d,
                p,
                jnp.array(0.0),
                f(x, p),
                grad_f @ d,
                g_val,
                jac,
                jnp.eye(2),
                grad_f,
                jnp.zeros(1),
                th,
                fl,
                jnp.array(1e4),
                f,
                g,
                lhs,
                rhs,
                lb,
                ub,
                2,
                1,
                cfg,
            )

        acc, ft, d_soc, lam = jax.jit(jax.vmap(run))(
            x[None], d[None], grad_f[None], g_val[None], jac[None]
        )
        assert acc.shape == (1,) and d_soc.shape == (1, 2) and lam.shape == (1, 1)


class TestRestorationAtFeasiblePoint:
    def test_zero_violation_entry_cannot_be_restored(self, cfg):
        # theta_k = 0: no trial can strictly reduce the violation, so the
        # restoration line search must report alpha = 0 (a stall), not accept
        # the zero step via theta_t <= (1 - gamma) * 0.
        def f(x, p):
            return -x[0]

        def g(x, p):
            return (x[0] - 1.0).reshape(1)

        lhs, rhs = jnp.array([-INF]), jnp.array([0.0])
        lb, ub = jnp.array([-5.0]), jnp.array([5.0])
        th, fl = empty_filter(4)
        alpha = restoration_line_search(
            jnp.array([0.5]),
            jnp.array([0.0]),
            jnp.zeros(0),
            jnp.array(0.0),
            th,
            fl,
            jnp.array(1e4),
            f,
            g,
            lhs,
            rhs,
            lb,
            ub,
            1,
            cfg,
        )
        assert float(alpha) == 0.0
