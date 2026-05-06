"""Tests for septal.jax.sqp.qp_subproblem (OSQP-style ADMM).

The QP problem solved is:

    min_{d}  ½ dᵀ Q d + cᵀ d    s.t.  l ≤ A d ≤ u

where A stacks general (linearised) constraints above box constraints.
Equality constraints are encoded as l_i = u_i.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from septal.jax.sqp.qp_subproblem import (
    admm_qp,
    form_qp_matrices,
    solve_qp_subproblem,
)
from septal.jax.sqp.schema import SQPConfig

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Default config with enough iterations for tight convergence."""
    return SQPConfig(
        admm_n_iter=1000,
        admm_rho=1.0,
        admm_sigma=1e-6,
        admm_alpha=1.6,
        admm_adaptive_rho=True,
        admm_rho_update_interval=25,
        admm_mu=10.0,
        admm_tau=2.0,
        admm_rho_min=1e-6,
        admm_rho_max=1e6,
        osqp_reg=1e-7,
    )


@pytest.fixture
def cfg_fixed_rho(cfg):
    """Config with adaptive rho disabled for comparison tests."""
    return SQPConfig(
        **{
            **cfg.__dict__,
            "admm_adaptive_rho": False,
        }
    )


# ---------------------------------------------------------------------------
# TestFormQPMatrices — updated for (Q, c, A, l, u) return signature
# ---------------------------------------------------------------------------


class TestFormQPMatrices:
    def test_unconstrained_shape(self, cfg):
        """Box-only QP: A has shape (n, n) — just the identity (box rows)."""
        n, n_g = 3, 0
        H = jnp.eye(n)
        g = jnp.ones(n)
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)

        Q, c, A, l, u = form_qp_matrices(
            H, g, jnp.zeros((0, n)), jnp.zeros(0),
            jnp.zeros(n), lb, ub, jnp.zeros(0), jnp.zeros(0),
            n, n_g, cfg.osqp_reg,
        )
        assert Q.shape == (n, n)
        assert c.shape == (n,)
        assert A.shape == (n, n)       # only box rows: I_n
        assert l.shape == (n,)
        assert u.shape == (n,)

    def test_constrained_shape(self, cfg):
        """Constrained QP: A has shape (n_g + n, n)."""
        n, n_g = 3, 2
        H = jnp.eye(n)
        jac_g = jnp.ones((n_g, n))
        g_val = jnp.zeros(n_g)
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)
        lhs = jnp.full(n_g, -jnp.inf)
        rhs = jnp.zeros(n_g)

        Q, c, A, l, u = form_qp_matrices(
            H, jnp.ones(n), jac_g, g_val, jnp.zeros(n),
            lb, ub, lhs, rhs, n, n_g, cfg.osqp_reg,
        )
        assert A.shape == (n_g + n, n)  # general + box rows
        assert l.shape == (n_g + n,)
        assert u.shape == (n_g + n,)

    def test_psd_regularisation(self, cfg):
        """Q must be (approximately) PSD after regularisation."""
        n = 3
        Q, *_ = form_qp_matrices(
            jnp.eye(n), jnp.zeros(n), jnp.zeros((0, n)), jnp.zeros(0),
            jnp.zeros(n), jnp.full(n, -1.0), jnp.full(n, 1.0),
            jnp.zeros(0), jnp.zeros(0), n, 0, cfg.osqp_reg,
        )
        eigvals = jnp.linalg.eigvalsh(Q)
        assert float(jnp.min(eigvals)) >= -1e-10

    def test_box_bounds_encoded(self, cfg):
        """l and u correctly encode ub - x and lb - x for box rows."""
        n = 2
        x = jnp.array([0.5, -0.3])
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)

        _, _, _, l, u = form_qp_matrices(
            jnp.eye(n), jnp.zeros(n), jnp.zeros((0, n)), jnp.zeros(0),
            x, lb, ub, jnp.zeros(0), jnp.zeros(0), n, 0, cfg.osqp_reg,
        )
        assert jnp.allclose(l, lb - x)
        assert jnp.allclose(u, ub - x)

    def test_equality_constraint_l_equals_u(self, cfg):
        """Equality constraint lhs == rhs produces l_i == u_i in output."""
        n, n_g = 2, 1
        rhs_val = 0.3
        _, _, _, l, u = form_qp_matrices(
            jnp.eye(n), jnp.zeros(n),
            jnp.ones((n_g, n)), jnp.zeros(n_g),
            jnp.zeros(n),
            jnp.full(n, -5.0), jnp.full(n, 5.0),
            jnp.array([rhs_val]), jnp.array([rhs_val]),
            n, n_g, cfg.osqp_reg,
        )
        # General constraint row: l[0] == u[0] == rhs_val - g_val == rhs_val
        assert jnp.allclose(l[0], u[0])


# ---------------------------------------------------------------------------
# TestAdmmQP — unit tests for the raw ADMM solver
# ---------------------------------------------------------------------------


class TestAdmmQP:
    """Tests operating directly on admm_qp for fine-grained coverage."""

    def _run(self, cfg, Q, c, A, l, u, n_iter=2000):
        return admm_qp(
            Q, c, A, l, u,
            rho_init=cfg.admm_rho,
            sigma=cfg.admm_sigma,
            alpha=cfg.admm_alpha,
            n_iter=n_iter,
            rho_min=cfg.admm_rho_min,
            rho_max=cfg.admm_rho_max,
            rho_update_interval=cfg.admm_rho_update_interval,
            mu=cfg.admm_mu,
            tau=cfg.admm_tau,
            adaptive_rho=cfg.admm_adaptive_rho,
        )

    def test_unconstrained_recovers_exact_solution(self, cfg):
        """min ½‖d‖² + cᵀd  with wide box → d* = -c (unconstrained minimum)."""
        n = 3
        Q = jnp.eye(n)
        c = jnp.array([1.0, -2.0, 0.5])
        # Very wide box so constraints never active
        A = jnp.eye(n)
        l = jnp.full(n, -1e9)
        u = jnp.full(n,  1e9)

        d, y = self._run(cfg, Q, c, A, l, u)
        assert jnp.allclose(d, -c, atol=1e-4)

    def test_pure_inequality_constraint(self, cfg):
        """min ½‖d‖²  s.t.  d[0] ≤ 0.3  →  d* = [0.3, 0] when pushed by c."""
        n = 2
        Q = jnp.eye(n)
        c = jnp.array([-1.0, 0.0])   # pulls d[0] to +∞
        # Constraint: d[0] ≤ 0.3  (upper bound row)
        A = jnp.eye(n)
        l = jnp.full(n, -1e9)
        u = jnp.array([0.3, 1e9])

        d, y = self._run(cfg, Q, c, A, l, u)
        assert float(d[0]) <= 0.3 + 1e-4
        assert jnp.allclose(d[0], 0.3, atol=1e-3)

    def test_equality_constraint_satisfied(self, cfg):
        """min ½‖d‖²  s.t.  d[0] + d[1] = 1  →  d* = [0.5, 0.5]."""
        n = 2
        Q = jnp.eye(n)
        c = jnp.zeros(n)
        # Equality row: l[0] = u[0] = 1
        A_eq = jnp.array([[1.0, 1.0]])
        A_box = jnp.eye(n)
        A = jnp.concatenate([A_eq, A_box], axis=0)
        l = jnp.array([1.0, -1e9, -1e9])   # equality + wide box
        u = jnp.array([1.0,  1e9,  1e9])

        d, y = self._run(cfg, Q, c, A, l, u)
        assert jnp.allclose(d, jnp.array([0.5, 0.5]), atol=1e-3)
        # Constraint satisfaction
        assert jnp.allclose(A_eq @ d, jnp.array([1.0]), atol=1e-3)

    def test_equality_multiplier_sign(self, cfg):
        """Multiplier for a binding equality is non-zero with correct sign.

        min ½‖d‖² + [-1, 0]d  s.t.  d[0] = 0
        KKT: d + c + y[0]*[1,0] = 0  →  y[0] = -(d[0] + c[0]) = -c[0] = 1.
        """
        n = 2
        Q = jnp.eye(n)
        c = jnp.array([-1.0, 0.0])
        A_eq = jnp.array([[1.0, 0.0]])
        A_box = jnp.eye(n)
        A = jnp.concatenate([A_eq, A_box], axis=0)
        l = jnp.array([0.0, -1e9, -1e9])
        u = jnp.array([0.0,  1e9,  1e9])

        d, y = self._run(cfg, Q, c, A, l, u)
        assert jnp.allclose(d, jnp.zeros(n), atol=1e-3)
        # y[0] should be positive (upper==lower bound, pulls d[0] toward 0)
        assert float(y[0]) > 0.0

    def test_mixed_equality_and_inequality(self, cfg):
        """min ½‖d - [1,1]‖²  s.t.  d[0] + d[1] = 1,  d[0] ≤ 0.8.

        Equality forces d[0]+d[1]=1; inequality is slack at d*=[0.5,0.5].
        """
        n = 2
        target = jnp.array([1.0, 1.0])
        Q = jnp.eye(n)
        c = -target                         # min ½‖d-t‖² = min ½‖d‖² - tᵀd + const
        A_eq  = jnp.array([[1.0,  1.0]])
        A_ineq = jnp.array([[1.0,  0.0]])
        A_box = jnp.eye(n)
        A = jnp.concatenate([A_eq, A_ineq, A_box], axis=0)
        l = jnp.array([1.0,  -1e9, -1e9, -1e9])
        u = jnp.array([1.0,   0.8,  1e9,  1e9])

        d, y = self._run(cfg, Q, c, A, l, u)
        # Equality satisfied
        assert jnp.allclose(A_eq @ d, jnp.array([1.0]), atol=1e-3)
        # Inequality slack (d[0] ≈ 0.5 < 0.8)
        assert float(d[0]) <= 0.8 + 1e-3

    def test_sigma_regularisation_ill_conditioned_hessian(self, cfg):
        """σ regularisation keeps solve stable for near-singular Q."""
        n = 3
        # Q has one near-zero eigenvalue
        v = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
        Q = jnp.eye(n) * 1e-8 + jnp.outer(v, v)   # near-singular
        c = jnp.array([0.1, -0.1, 0.0])
        A = jnp.eye(n)
        l = jnp.full(n, -1e9)
        u = jnp.full(n,  1e9)

        d, y = self._run(cfg, Q, c, A, l, u)
        assert jnp.all(jnp.isfinite(d)), "NaN/Inf from ill-conditioned Q"

    def test_adaptive_rho_outperforms_fixed_on_scaled_problem(self, cfg, cfg_fixed_rho):
        """Adaptive ρ should reach smaller primal residual than fixed ρ on
        a badly scaled problem within the same iteration budget."""
        n = 4
        # Badly scaled Q: eigenvalues span 1e-3 to 1e3
        scales = jnp.array([1e-3, 1e-1, 1e1, 1e3])
        Q = jnp.diag(scales)
        c = jnp.ones(n)
        A = jnp.eye(n)
        l = jnp.full(n, -10.0)
        u = jnp.full(n,  10.0)

        def primal_residual(d, A, l, u):
            Ad = A @ d
            return jnp.max(jnp.maximum(jnp.maximum(l - Ad, Ad - u), jnp.zeros_like(Ad)))

        d_adaptive, _ = self._run(cfg,            Q, c, A, l, u, n_iter=500)
        d_fixed,    _ = self._run(cfg_fixed_rho,  Q, c, A, l, u, n_iter=500)

        r_adaptive = primal_residual(d_adaptive, A, l, u)
        r_fixed    = primal_residual(d_fixed,    A, l, u)
        assert float(r_adaptive) <= float(r_fixed) + 1e-6, (
            f"Adaptive ρ primal residual {r_adaptive:.2e} should not exceed "
            f"fixed ρ residual {r_fixed:.2e} on a scaled problem"
        )


# ---------------------------------------------------------------------------
# TestSolveQPSubproblem — integration tests via the full SQP wrapper
# ---------------------------------------------------------------------------


class TestSolveQPSubproblem:
    def test_box_qp_optimal_direction(self, cfg):
        """min ½‖d‖² + d·ones  s.t.  -1 ≤ d ≤ 1  →  d* = -1 (clamped at lb)."""
        n = 2
        d, lam_g = solve_qp_subproblem(
            jnp.eye(n), jnp.ones(n),
            jnp.zeros((0, n)), jnp.zeros(0),
            jnp.zeros(n),
            jnp.full(n, -1.0), jnp.full(n, 1.0),
            jnp.zeros(0), jnp.zeros(0),
            n, 0, cfg,
        )
        assert d.shape == (n,)
        assert lam_g.shape == (0,)
        assert jnp.allclose(d, jnp.full(n, -1.0), atol=1e-3)

    def test_inequality_constraint_satisfied(self, cfg):
        """Constrained direction satisfies the linearised constraint."""
        n, n_g = 2, 1
        jac_g = jnp.array([[-1.0, 0.0]])   # constraint: -d[0] ≤ -0.5
        rhs   = jnp.array([-0.5])
        lhs   = jnp.array([-jnp.inf])

        d, lam_g = solve_qp_subproblem(
            jnp.eye(n), jnp.zeros(n),
            jac_g, jnp.zeros(n_g),
            jnp.zeros(n),
            jnp.full(n, -5.0), jnp.full(n, 5.0),
            lhs, rhs, n, n_g, cfg,
        )
        assert d.shape == (n,)
        assert lam_g.shape == (n_g,)
        assert float(jac_g[0] @ d) <= float(rhs[0]) + 1e-3

    def test_equality_constraint_satisfied(self, cfg):
        """min ½‖d‖²  s.t.  d[0] + d[1] = 1  (lhs == rhs)  →  d* = [0.5, 0.5]."""
        n, n_g = 2, 1
        jac_g = jnp.array([[1.0, 1.0]])
        eq_val = jnp.array([1.0])

        d, lam_g = solve_qp_subproblem(
            jnp.eye(n), jnp.zeros(n),
            jac_g, jnp.zeros(n_g),
            jnp.zeros(n),
            jnp.full(n, -5.0), jnp.full(n, 5.0),
            eq_val, eq_val,      # lhs == rhs → equality
            n, n_g, cfg,
        )
        assert jnp.allclose(d, jnp.array([0.5, 0.5]), atol=1e-3)
        assert jnp.allclose(jac_g @ d, eq_val, atol=1e-3)

    def test_multiplier_upper_bound_positive(self, cfg):
        """Active upper bound should give a positive KKT multiplier."""
        n, n_g = 1, 1
        # min ½d²  s.t.  d ≤ -0.5  (upper bound forces d to be negative)
        jac_g = jnp.array([[1.0]])
        lhs   = jnp.array([-jnp.inf])
        rhs   = jnp.array([-0.5])

        _, lam_g = solve_qp_subproblem(
            jnp.eye(n), jnp.zeros(n),
            jac_g, jnp.zeros(n_g),
            jnp.zeros(n),
            jnp.full(n, -5.0), jnp.full(n, 5.0),
            lhs, rhs, n, n_g, cfg,
        )
        assert float(lam_g[0]) >= 0.0, "Upper-bound multiplier must be ≥ 0"

    def test_multiplier_lower_bound_negative(self, cfg):
        """Active lower bound should give a negative KKT multiplier."""
        n, n_g = 1, 1
        # min ½d²  s.t.  d ≥ 0.5  (lower bound forces d to be positive)
        jac_g = jnp.array([[1.0]])
        lhs   = jnp.array([0.5])
        rhs   = jnp.array([jnp.inf])

        _, lam_g = solve_qp_subproblem(
            jnp.eye(n), jnp.zeros(n),
            jac_g, jnp.zeros(n_g),
            jnp.zeros(n),
            jnp.full(n, -5.0), jnp.full(n, 5.0),
            lhs, rhs, n, n_g, cfg,
        )
        assert float(lam_g[0]) <= 0.0, "Lower-bound multiplier must be ≤ 0"

    def test_returns_finite(self, cfg):
        """No NaN/Inf in output for a trivial problem."""
        n = 3
        d, _ = solve_qp_subproblem(
            jnp.eye(n), jnp.zeros(n),
            jnp.zeros((0, n)), jnp.zeros(0),
            jnp.zeros(n),
            jnp.full(n, -1.0), jnp.full(n, 1.0),
            jnp.zeros(0), jnp.zeros(0),
            n, 0, cfg,
        )
        assert jnp.all(jnp.isfinite(d))

    def test_vmap_compatible(self, cfg):
        """solve_qp_subproblem must be vmappable over a batch of problems."""
        n, batch = 2, 5
        key = jax.random.PRNGKey(42)
        H_batch    = jnp.stack([jnp.eye(n)] * batch)
        grad_batch = jax.random.normal(key, (batch, n))
        x_batch    = jnp.zeros((batch, n))
        lb = jnp.full(n, -2.0)
        ub = jnp.full(n,  2.0)

        def _solve(H, g, x):
            d, _ = solve_qp_subproblem(
                H, g, jnp.zeros((0, n)), jnp.zeros(0),
                x, lb, ub, jnp.zeros(0), jnp.zeros(0),
                n, 0, cfg,
            )
            return d

        d_batch = jax.vmap(_solve)(H_batch, grad_batch, x_batch)
        assert d_batch.shape == (batch, n)
        assert jnp.all(jnp.isfinite(d_batch))

    def test_vmap_matches_sequential(self, cfg):
        """Batched vmap results must match individual sequential solves."""
        n, batch = 2, 3
        key = jax.random.PRNGKey(7)
        grads = jax.random.normal(key, (batch, n))
        lb = jnp.full(n, -2.0)
        ub = jnp.full(n,  2.0)

        def _solve(g):
            d, _ = solve_qp_subproblem(
                jnp.eye(n), g, jnp.zeros((0, n)), jnp.zeros(0),
                jnp.zeros(n), lb, ub, jnp.zeros(0), jnp.zeros(0),
                n, 0, cfg,
            )
            return d

        d_vmap = jax.vmap(_solve)(grads)
        d_seq  = jnp.stack([_solve(grads[i]) for i in range(batch)])
        assert jnp.allclose(d_vmap, d_seq, atol=1e-6)
