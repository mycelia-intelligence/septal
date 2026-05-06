"""
QP subproblem formation and solution for the SQP outer loop.

OSQP-style ADMM  (Stellato et al. 2017, arxiv.org/abs/1711.08013)
------------------------------------------------------------------
Solves  min_{d}  ½ dᵀ Q d + cᵀ d  s.t.  l ≤ A d ≤ u  via ADMM with:

  • Two-sided constraint form (no row-doubling).  Equality constraints
    (l_i = u_i) are handled exactly by the box projection Π_{[l,u]}.
  • Proximal regularisation σI on the x-update guarantees a positive-definite
    system matrix even when Q is only PSD.
  • Over-relaxation (α ≈ 1.6, per paper §3.4) applied to Ax before projection.
  • Equality-row ρ scaling: rows where l_i = u_i (equality constraints) are
    assigned an effective ρ of rho * admm_rho_eq_scale.  Without this boost,
    ADMM convergence on equality rows degenerates to dual-ascent O(1/k).
  • Warm-starting: the ADMM dual variable y is initialised from the previous
    SQP multiplier estimate rather than zero, halving the number of inner
    iterations needed on subsequent SQP steps.
  • Adaptive ρ: every ``rho_update_interval`` iterations ρ is scaled by τ
    when the primal/dual residual ratio exceeds μ (paper §3.5).

ADMM iterates (k → k+1)  — with per-row effective rho ρᵢ = ρ · eq_scale_i
---------------------------------------------------------------------------
  x^{k+1} = (Q + σI + Aᵀ diag(ρ_vec) A)⁻¹ (σx^k - c + Aᵀ(ρ_vec ⊙ z^k - y^k))
  ã        = α Ax^{k+1} + (1-α) z^k                   (over-relaxation)
  z^{k+1} = clip(ã + y^k / ρ_vec,  l,  u)             (row-wise projection)
  y^{k+1} = y^k + ρ_vec ⊙ (ã - z^{k+1})               (dual update)

At convergence  y ≈ λ  (KKT multipliers for  l ≤ Ad ≤ u).
Positive λ_i when upper bound active; negative when lower bound active.

Constraint layout
-----------------
The stacked constraint matrix A has:
  rows 0 : n_g   — linearised general constraints  lhs - g_k ≤ J_k d ≤ rhs - g_k
  rows n_g : m   — box step constraints            lb  - x_k ≤    d  ≤ ub  - x_k

where m = n_g + n.  At the SQP level only y[:n_g] (general-constraint
multipliers) is returned; the box multipliers y[n_g:] are discarded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from septal.jax.sqp.schema import SQPConfig

_INF_VAL = 1e10   # finite stand-in for ±∞ in constraint bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_bounds(l: jnp.ndarray, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Replace ±inf with ±_INF_VAL so all ADMM arithmetic is finite."""
    l_s = jnp.where(jnp.isinf(l) & (l < 0), -_INF_VAL, l)
    u_s = jnp.where(jnp.isinf(u) & (u > 0), _INF_VAL, u)
    return l_s, u_s


# ---------------------------------------------------------------------------
# Ruiz diagonal equilibration
# ---------------------------------------------------------------------------


def ruiz_equilibration(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    l: jnp.ndarray,
    u: jnp.ndarray,
    n_iter: int,
    eps: float = 1e-8,
) -> tuple:
    """Ruiz diagonal equilibration for ``min ½xᵀQx + cᵀx  s.t.  l ≤ Ax ≤ u``.

    Finds diagonal D (n×n, column/variable scaling) and E (m×m, row/constraint
    scaling) so that the scaled matrices

        Q_s = D Q D,   A_s = E A D,   c_s = D c,   l_s = E l,   u_s = E u

    have approximately unit ∞-norm on every row and column.  This makes the
    base ADMM penalty ρ equally effective for every constraint row — the key
    fix for the hs021 failure where the general constraint row [10, -1] has
    10× larger ∞-norm than the box rows [1,0] and [0,1].

    After solving the scaled QP (primal x̃, dual ỹ), recover originals via::

        x = d * x̃     (column unscale)
        y = e * ỹ     (row unscale)

    Parameters
    ----------
    Q : (n, n)   Symmetric PSD cost (already regularised).
    c : (n,)     Linear cost.
    A : (m, n)   Constraint matrix.
    l, u : (m,)  Constraint bounds.
    n_iter :     Fixed Ruiz iterations; 10 is typically sufficient.
    eps :        ∞-norm floor to avoid division by zero.

    Returns
    -------
    Q_s, c_s, A_s, l_s, u_s : scaled QP data.
    d : (n,)  Cumulative column scale.
    e : (m,)  Cumulative row scale.
    """
    d0 = jnp.ones(Q.shape[0], dtype=Q.dtype)
    e0 = jnp.ones(A.shape[0], dtype=Q.dtype)

    def _step(carry, _):
        Q_s, A_s, d, e = carry

        # --- row scaling of A (makes ‖A_s[i,:]‖∞ → 1) -------------------
        row_norms = jnp.maximum(jnp.max(jnp.abs(A_s), axis=1), eps)
        rs = 1.0 / jnp.sqrt(row_norms)                  # (m,)
        A_s = A_s * rs[:, None]
        e = e * rs

        # --- column scaling of [Q; A] (makes ‖col‖∞ → 1) ----------------
        col_norms_A = jnp.max(jnp.abs(A_s), axis=0)     # (n,)
        col_norms_Q = jnp.max(jnp.abs(Q_s), axis=0)     # (n,) — Q symmetric
        col_norms = jnp.maximum(jnp.maximum(col_norms_A, col_norms_Q), eps)
        cs = 1.0 / jnp.sqrt(col_norms)                  # (n,)
        A_s = A_s * cs[None, :]
        Q_s = Q_s * cs[:, None] * cs[None, :]
        d = d * cs

        return (Q_s, A_s, d, e), None

    (Q_s, A_s, d, e), _ = jax.lax.scan(
        _step, (Q, A, d0, e0), None, length=n_iter
    )

    c_s = c * d
    l_s = l * e
    u_s = u * e
    return Q_s, c_s, A_s, l_s, u_s, d, e


# ---------------------------------------------------------------------------
# QP matrix formation
# ---------------------------------------------------------------------------


def form_qp_matrices(
    hessian: jnp.ndarray,
    grad_f: jnp.ndarray,
    jac_g: jnp.ndarray,
    g_val: jnp.ndarray,
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    n: int,
    n_g: int,
    reg: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build OSQP-form QP matrices ``(Q, c, A, l, u)`` for
    ``min ½dᵀQd + cᵀd  s.t.  l ≤ Ad ≤ u``.

    The constraint matrix stacks general (linearised) constraints above
    box constraints on the step *d*::

        A = [ J_k ]    l = [ lhs - g_k ]    u = [ rhs - g_k ]
            [ I_n ]        [ lb  - x_k ]        [ ub  - x_k ]

    Parameters
    ----------
    hessian : (n, n)   BFGS Hessian approximation.
    grad_f  : (n,)     Objective gradient ∇f(x_k, p).
    jac_g   : (n_g, n) Constraint Jacobian.  Pass ``zeros((0, n))`` if n_g=0.
    g_val   : (n_g,)   Constraint value g(x_k, p).  Pass ``zeros(0)`` if n_g=0.
    x       : (n,)     Current SQP iterate.
    lb, ub  : (n,)     Decision-variable bounds.
    lhs,rhs : (n_g,)   Two-sided constraint bounds.
    n, n_g  : int      Problem dimensions.
    reg     : float    Small diagonal regularisation added to Q.

    Returns
    -------
    Q : (n, n),  c : (n,),  A : (n_g+n, n),  l : (n_g+n,),  u : (n_g+n,)
    """
    Q = 0.5 * (hessian + hessian.T) + reg * jnp.eye(n)
    c = grad_f.reshape(n)

    if n_g > 0:
        l_gc = lhs.reshape(n_g) - g_val.reshape(n_g)
        u_gc = rhs.reshape(n_g) - g_val.reshape(n_g)
        l_gc, u_gc = _safe_bounds(l_gc, u_gc)
        A = jnp.concatenate([jac_g.reshape(n_g, n), jnp.eye(n)], axis=0)
        l_all = jnp.concatenate([l_gc, lb - x])
        u_all = jnp.concatenate([u_gc, ub - x])
    else:
        A = jnp.eye(n)
        l_all = lb - x
        u_all = ub - x

    l_all, u_all = _safe_bounds(l_all, u_all)
    return Q, c, A, l_all, u_all


# ---------------------------------------------------------------------------
# OSQP-style ADMM solver
# ---------------------------------------------------------------------------


def admm_qp(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    l: jnp.ndarray,
    u: jnp.ndarray,
    rho_init: float,
    sigma: float,
    alpha: float,
    n_iter: int,
    rho_min: float,
    rho_max: float,
    rho_update_interval: int,
    mu: float,
    tau: float,
    adaptive_rho: bool,
    eq_scale: jnp.ndarray,
    y_warm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """OSQP-style ADMM QP solver for  ``min ½dᵀQd + cᵀd  s.t.  l ≤ Ad ≤ u``.

    Fully vmappable and JIT-compilable via ``lax.scan`` with a fixed
    ``n_iter`` iteration budget.

    Parameters
    ----------
    Q        : (n, n)  Symmetric PSD cost matrix (already regularised).
    c        : (n,)    Linear cost vector.
    A        : (m, n)  Constraint matrix (general + box rows stacked).
    l, u     : (m,)    Constraint bounds.  ``l_i == u_i`` encodes equality.
    rho_init : float   Initial ADMM base penalty parameter ρ₀.
    sigma    : float   Proximal regularisation on x (σ > 0).
    alpha    : float   Over-relaxation parameter α ∈ (0, 2).  Paper: 1.6.
    n_iter   : int     Fixed number of ADMM iterations.
    rho_min  : float   Lower clip for adaptive ρ.
    rho_max  : float   Upper clip for adaptive ρ.
    rho_update_interval : int  Iterations between ρ checks.
    mu       : float   Residual ratio threshold (paper default: 10).
    tau      : float   ρ scale factor on update (paper default: 2).
    adaptive_rho : bool  Enable adaptive ρ.
    eq_scale : (m,)    Per-row ρ multiplier.  Set to ``admm_rho_eq_scale``
                       for equality rows (``l_i == u_i``), 1.0 elsewhere.
                       Equality rows converge as dual-ascent without boosting.
    y_warm   : (m,)    Warm-start dual variables.  Pass ``zeros(m)`` for a
                       cold start; pass the previous SQP multiplier estimates
                       to halve inner iterations on subsequent SQP steps.

    Returns
    -------
    d : (n,)   Primal solution (step direction).
    y : (m,)   Dual variables — KKT multipliers for ``l ≤ Ad ≤ u``.
               Positive when upper bound active; negative when lower bound.
    z : (m,)   Auxiliary ADMM variable at final iteration.  Used by the
               polishing step to classify the active constraint set via the
               OSQP criterion (Stellato et al. 2020, §3.2).
    """
    n = Q.shape[0]

    # Constant parts of the system matrix --------------------------------
    # With per-row eq_scale: M = Q + σI + ρ · Aᵀ diag(eq_scale) A
    #   AtA_scaled = Aᵀ diag(eq_scale) A = (A * eq_scale[:,None])ᵀ A
    Q_sigma = Q + sigma * jnp.eye(n)
    AtA_scaled = (A * eq_scale[:, None]).T @ A   # (n, n) — precomputed once

    rho0 = jnp.array(rho_init)
    M0 = Q_sigma + rho0 * AtA_scaled

    # ADMM variables: x=step direction, z=auxiliary, y=dual (warm-started)
    x0 = jnp.zeros(n)
    z0 = jnp.zeros_like(l)
    y0 = y_warm                     # warm-start from previous SQP multipliers

    # Scan state: (x, z, y, rho, M, z_prev, iter)
    init_state = (x0, z0, y0, rho0, M0, z0, jnp.array(0, dtype=jnp.int32))

    def admm_step(
        state: tuple, _: None
    ) -> tuple[tuple, None]:
        x, z, y, rho, M, z_prev, k = state

        # Per-row effective rho  ρ_vec = ρ · eq_scale
        rho_vec = rho * eq_scale

        # ---- x-update -----------------------------------------------
        # Solve (Q + σI + ρ Aᵀ diag(eq_scale) A) x_new
        #           = σx - c + Aᵀ(ρ_vec ⊙ z - y)
        rhs_x = sigma * x - c + A.T @ (rho_vec * z - y)
        x_new = jnp.linalg.solve(M, rhs_x)

        # ---- over-relaxation ----------------------------------------
        Ax_new = A @ x_new
        a_tilde = alpha * Ax_new + (1.0 - alpha) * z

        # ---- z-update: row-wise box projection onto [l, u] ----------
        z_new = jnp.clip(a_tilde + y / rho_vec, l, u)

        # ---- y-update: dual variable with per-row rho ---------------
        y_new = y + rho_vec * (a_tilde - z_new)

        # ---- adaptive ρ (Python-level branch — static at trace time) -
        k_new = k + 1
        if adaptive_rho:
            should_update = (jnp.mod(k_new, rho_update_interval) == 0)

            r_prim = jnp.linalg.norm(Ax_new - z_new)
            # Dual residual uses unscaled rho to keep the adaptive logic
            # independent of the equality-row boost factor.
            r_dual = jnp.linalg.norm(rho * (A.T @ (z_new - z_prev)))

            rho_up   = jnp.minimum(rho * tau, rho_max)
            rho_down = jnp.maximum(rho / tau, rho_min)
            rho_candidate = jnp.where(
                r_prim > mu * r_dual, rho_up,
                jnp.where(r_dual > mu * r_prim, rho_down, rho),
            )
            rho_new = jnp.where(should_update, rho_candidate, rho)
            # M changes only when rho changes; eq_scale is absorbed in AtA_scaled.
            M_new   = jnp.where(should_update, Q_sigma + rho_new * AtA_scaled, M)
        else:
            rho_new = rho
            M_new   = M

        return (x_new, z_new, y_new, rho_new, M_new, z_new, k_new), None

    (x_final, z_final, y_final, _, _, _, _), _ = jax.lax.scan(
        admm_step, init_state, None, length=n_iter
    )

    return x_final, y_final, z_final


# ---------------------------------------------------------------------------
# KKT residual helper (used for polishing fallback comparison)
# ---------------------------------------------------------------------------


def _kkt_residual_norm(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    l: jnp.ndarray,
    u: jnp.ndarray,
    d: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Scalar KKT quality metric for primal+dual residuals (scaled space).

    Computes ‖Qd + c + Aᵀy‖₂  +  ‖Ad − clip(Ad + y, l, u)‖₂.
    Used only to decide whether the polished solution is better than ADMM.
    """
    r_stat = Q @ d + c + A.T @ y
    Ad = A @ d
    r_feas = Ad - jnp.clip(Ad + y, l, u)
    return jnp.linalg.norm(r_stat) + jnp.linalg.norm(r_feas)


# ---------------------------------------------------------------------------
# OSQP-style polishing step  (Stellato et al. 2020, Eq 30)
# ---------------------------------------------------------------------------


def polish_qp(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    l: jnp.ndarray,
    u: jnp.ndarray,
    d_admm: jnp.ndarray,
    y_admm: jnp.ndarray,
    z_admm: jnp.ndarray,
    sigma: float,
    eps: float,
    n_refine: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """OSQP-style polishing step (Eq 30, Stellato et al. 2020).

    After ADMM converges to ``(d_admm, y_admm, z_admm)``, this step guesses
    the active constraint set and solves a reduced KKT system to obtain a
    high-accuracy primal/dual pair.

    Active-set classification (OSQP §3.2, using both z and y):
      • lower-active row i: ``z[i] − l[i] < −y[i]``  OR  ``l[i] == u[i]``
      • upper-active row i: ``u[i] − z[i] < y[i]``  AND  ``l[i] != u[i]``
      • inactive: all others  →  ŷ[i] = 0

    The reduced KKT system (Eq 30) is embedded in a fixed-size
    ``(n + m) × (n + m)`` system so that the array shapes are static
    (required for JAX JIT / vmap compatibility)::

        [ Q + σI          A^T · diag(is_act)       ] [ x̂ ]   [ −c          ]
        [ diag(is_act)·A  diag((1−is_act)·ε)       ] [ ŷ ] = [ is_act·b_A  ]

    For active row i:   ``A[i,:]·x̂ = b_A[i]``  (constraint enforced exactly)
    For inactive row i: ``ε·ŷ[i] = 0``          (dual forced to ≈ 0)

    This is equivalent to Eq 30: inactive duals are zero so they don't enter
    the stationarity equation, and active constraints are satisfied exactly.

    Iterative refinement (``n_refine`` steps via ``lax.fori_loop``) corrects
    the regularisation error introduced by ``σ`` and ``ε``.

    Parameters
    ----------
    Q, c, A, l, u : Ruiz-scaled QP data (same space as d_admm / y_admm).
    d_admm : (n,)  ADMM primal solution (scaled).
    y_admm : (m,)  ADMM dual solution (scaled).
    z_admm : (m,)  ADMM auxiliary variable at final iteration (scaled).
    sigma :        Regularisation added to (1,1) block diagonal.
    eps :          Regularisation placed on inactive dual diagonal rows.
    n_refine :     Iterative refinement steps (0 = disabled).

    Returns
    -------
    d_pol : (n,)   Polished primal (scaled; caller unscales with d_scale).
    y_pol : (m,)   Polished dual   (scaled; caller unscales with e_scale).
    """
    n = Q.shape[0]

    # --- Active-set classification (OSQP §3.2 criterion) ------------------
    is_eq = l == u                                          # (m,) bool
    lower_act = ((z_admm - l) < -y_admm) | is_eq           # lower bound active
    upper_act = ((u - z_admm) < y_admm) & ~is_eq           # upper bound active
    is_act = lower_act | upper_act                          # (m,) bool

    is_act_f = is_act.astype(Q.dtype)                       # float mask

    # b_A: the bound value for active rows (l if lower-active, u otherwise)
    b_A = jnp.where(lower_act, l, u)                       # (m,)

    # --- Fixed-size (n+m)×(n+m) KKT matrix assembly ----------------------
    # top-left:  Q + σI
    # top-right: A^T · diag(is_act)   — only active cols contribute to stat.
    # bot-left:  diag(is_act) · A     — only active rows enforced
    # bot-right: diag((1-is_act)·ε)   — tiny diag forces inactive ŷ ≈ 0
    top_left  = Q + sigma * jnp.eye(n, dtype=Q.dtype)       # (n, n)
    top_right = A.T * is_act_f[None, :]                     # (n, m)
    bot_left  = A * is_act_f[:, None]                       # (m, n)
    bot_right = jnp.diag((1.0 - is_act_f) * eps)           # (m, m)

    top = jnp.concatenate([top_left, top_right], axis=1)   # (n, n+m)
    bot = jnp.concatenate([bot_left, bot_right], axis=1)   # (m, n+m)
    KKT = jnp.concatenate([top, bot], axis=0)              # (n+m, n+m)

    rhs = jnp.concatenate([-c, jnp.where(is_act, b_A, 0.0)])  # (n+m,)  avoid 0*inf=NaN

    # --- Solve + iterative refinement ------------------------------------
    sol = jnp.linalg.solve(KKT, rhs)

    def _refine(_, sol_):
        residual = rhs - KKT @ sol_
        return sol_ + jnp.linalg.solve(KKT, residual)

    sol = jax.lax.fori_loop(0, n_refine, _refine, sol)

    return sol[:n], sol[n:]


# ---------------------------------------------------------------------------
# High-level wrapper used by the SQP outer loop
# ---------------------------------------------------------------------------


def solve_qp_subproblem(
    hessian: jnp.ndarray,
    grad_f: jnp.ndarray,
    jac_g: jnp.ndarray,
    g_val: jnp.ndarray,
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    n: int,
    n_g: int,
    cfg: SQPConfig,
    lam_prev: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the SQP QP subproblem via OSQP-style ADMM.

    Fully vmappable (no jaxopt dependency).

    Parameters
    ----------
    lam_prev : (n_g,), optional
        Multiplier estimate from the previous SQP iteration.  Used to
        warm-start the ADMM dual variable ``y``, significantly reducing
        the number of inner iterations needed.  Pass ``None`` for a cold
        start (e.g. on the first SQP iteration).

    Returns
    -------
    d     : (n,)    Primal search direction.
    lam_g : (n_g,)  Dual variables for the general constraints.
                    Positive when upper bound active, negative when lower.
    """
    Q, c, A, l_all, u_all = form_qp_matrices(
        hessian, grad_f, jac_g, g_val, x, lb, ub, lhs, rhs, n, n_g, cfg.osqp_reg
    )

    m = A.shape[0]  # = n_g + n (or just n when n_g == 0)

    # --- Ruiz diagonal equilibration (before eq_scale / warm-start) --------
    # Scale A rows/cols and Q cols to unit ∞-norm so that the base ADMM ρ
    # is equally effective for every constraint row.  After the scaled solve
    # we unscale: d = d_scale * d_s,  y = e_scale * y_s.
    if cfg.admm_ruiz_iter > 0:
        Q, c, A, l_all, u_all, d_scale, e_scale = ruiz_equilibration(
            Q, c, A, l_all, u_all, n_iter=cfg.admm_ruiz_iter
        )
    else:
        d_scale = jnp.ones(n, dtype=Q.dtype)
        e_scale = jnp.ones(m, dtype=Q.dtype)

    # Build per-row equality scale: equality rows (l_i == u_i) get a higher
    # rho to avoid the O(1/k) dual-ascent convergence rate on those rows.
    # After Ruiz, l_s[i]==u_s[i] iff l[i]==u[i] (positive scaling preserves ==).
    if n_g > 0:
        is_eq_gc = l_all[:n_g] == u_all[:n_g]   # equality among general constraints
        is_eq = jnp.concatenate([is_eq_gc, jnp.zeros(n, dtype=jnp.bool_)])
    else:
        is_eq = jnp.zeros(m, dtype=jnp.bool_)
    eq_scale = jnp.where(is_eq, cfg.admm_rho_eq_scale, 1.0)

    # Build warm-start dual: pad lam_prev, then unscale for the scaled problem.
    # Original dual y_orig = e_scale * y_scaled  =>  y_scaled = y_orig / e_scale.
    if lam_prev is not None and n_g > 0:
        y_warm_orig = jnp.concatenate([lam_prev, jnp.zeros(n)])
    else:
        y_warm_orig = jnp.zeros(m)
    y_warm = y_warm_orig / jnp.maximum(e_scale, 1e-30)

    d_s, y_s, z_s = admm_qp(
        Q, c, A, l_all, u_all,
        rho_init=cfg.admm_rho,
        sigma=cfg.admm_sigma,
        alpha=cfg.admm_alpha,
        n_iter=cfg.admm_n_iter,
        rho_min=cfg.admm_rho_min,
        rho_max=cfg.admm_rho_max,
        rho_update_interval=cfg.admm_rho_update_interval,
        mu=cfg.admm_mu,
        tau=cfg.admm_tau,
        adaptive_rho=cfg.admm_adaptive_rho,
        eq_scale=eq_scale,
        y_warm=y_warm,
    )

    # --- OSQP-style polishing (static Python branch — disabled at trace time
    #     when cfg.admm_polish=False, same pattern as admm_adaptive_rho) ----
    if cfg.admm_polish:
        # Active-set flag: gating inside the branch keeps these 4 operations
        # out of the XLA graph entirely when admm_polish=False, which matters
        # for vmap compile time (nested scans under vmap amplify body size).
        # For problems with no finite-bound active constraints (e.g. unconstrained
        # problems where all bounds are ±∞), polishing reduces to a pure Newton
        # step -(Q+σI)⁻¹c which diverges for non-convex Q — hence the gate.
        _is_eq_pre  = l_all == u_all
        _lower_pre  = ((z_s - l_all) < -y_s) | _is_eq_pre
        _upper_pre  = ((u_all - z_s) < y_s) & ~_is_eq_pre
        _has_active = (_lower_pre | _upper_pre).any()

        d_pol_s, y_pol_s = polish_qp(
            Q, c, A, l_all, u_all,
            d_s, y_s, z_s,
            sigma=cfg.admm_polish_reg,
            eps=cfg.admm_polish_reg,
            n_refine=cfg.admm_polish_refine,
        )

        # Sanitise before any jnp.where: on some XLA backends (macOS/Accelerate)
        # jnp.where(False, NaN_array, x) can still corrupt x because XLA's
        # select evaluates both branches arithmetically.  Replace NaN/Inf in the
        # polished output with the ADMM solution so the select is always safe.
        d_pol_s = jnp.where(jnp.isfinite(d_pol_s), d_pol_s, d_s)
        y_pol_s = jnp.where(jnp.isfinite(y_pol_s), y_pol_s, y_s)

        # Gate 1 — original polished output was finite (checked before sanitise).
        pol_finite = jnp.isfinite(d_pol_s).all() & jnp.isfinite(y_pol_s).all()

        # Gate 2 — OSQP two-residual criterion (Stellato et al. 2020, §3.4).
        #
        # The polished stationarity residual ‖Qd̂+c+Aᵀŷ‖ is ≈ σ‖d̂‖ ≈ 0 by
        # construction of the KKT system, so it cannot discriminate between a
        # correct and an incorrect active-set guess.  The primal feasibility
        # residual ‖Ad̂ − clip(Ad̂, l, u)‖ IS informative: for a correct active
        # set, inactive constraints are in the interior so it is ≈ 0; for a
        # wrong active set, violated inactive constraints give a positive value.
        #
        # We compare against the ADMM primal residual ‖Ad − z‖ (z_s is the
        # ADMM auxiliary variable at the final iteration, already returned above).
        # Polishing is accepted only when it at least matches ADMM on feasibility.
        admm_prim_res = jnp.linalg.norm(A @ d_s - z_s)
        pol_Ad = A @ d_pol_s
        pol_prim_res = jnp.linalg.norm(pol_Ad - jnp.clip(pol_Ad, l_all, u_all))
        pol_better = pol_prim_res <= admm_prim_res + 1e-10

        use_pol = pol_finite & pol_better & _has_active
        d_s = jnp.where(use_pol, d_pol_s, d_s)
        y_s = jnp.where(use_pol, y_pol_s, y_s)

    # Unscale primal and dual back to the original problem.
    d = d_scale * d_s
    y = e_scale * y_s

    # y[:n_g] are KKT multipliers for the general constraints.
    # y[n_g:] are box-constraint multipliers (discarded at the SQP level).
    lam_g = y[:n_g] if n_g > 0 else jnp.zeros(0)

    # Guard against NaN/Inf from a numerically ill-conditioned QP.
    d     = jnp.where(jnp.isfinite(d).all(),     d,     jnp.zeros(n))
    lam_g = jnp.where(jnp.isfinite(lam_g).all(), lam_g, jnp.zeros(n_g))

    return d, lam_g
