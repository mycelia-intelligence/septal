"""
Line-search filter step acceptance (Wächter & Biegler 2005).

Reference
---------
A. Wächter and L. T. Biegler, "Line Search Filter Methods for Nonlinear
Programming: Motivation and Global Convergence", SIAM J. Optim. 16(1),
pp. 1–31, 2005.  Equation numbers below refer to that paper.

The filter replaces the L1 penalty merit function.  A trial point is judged
on the pair ``(θ, f)`` where ``θ(x)`` is the L1 constraint violation and
``f(x)`` the objective.  The filter ``F_k`` is a set of such pairs; a trial
point is *acceptable* when it is not dominated by any stored pair, i.e. for
every ``(θ_j, f_j)`` in ``F_k``::

    θ(x) < θ_j   or   f(x) < f_j                                   (2.10)

Fixed-size storage
------------------
``lax.scan`` / ``jax.vmap`` need static shapes, so the filter is stored as
two arrays of length ``filter_size``.  Empty slots hold ``+inf`` in both
coordinates and never block a finite trial.  Augmentation first removes
entries dominated by the new one, then writes into the first empty slot
or, if the filter is full, evicts the entry with the largest ``θ``.

The initial filter ``{θ ≥ θ_max}`` of the paper is kept as a separate
scalar ``theta_max`` so it can never be evicted.

Map of equations to functions
-----------------------------
* (2.10) acceptability            → :func:`filter_acceptable`
* (2.11) sufficient decrease,
  (2.13) switching, (2.14) Armijo → :func:`judge_trial` (:func:`switching_condition`)
* (2.16) minimum step size        → :func:`min_step_size`
* (2.17) filter augmentation      → :func:`augment_filter`
* Alg. I step 4 (backtracking)    → :func:`filter_line_search`
* Second-order correction (Wächter & Biegler 2006, Math. Prog. 106, §2.4)
                                  → :func:`soc_correction`
* §3.2 feasibility restoration    → :func:`restoration_direction`,
                                    :func:`restoration_line_search`
  (the multi-iteration phase, its κ_resto exit rule and stall detection live
  in ``solver.py``)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from septal.jax.sqp.line_search import constraint_violation
from septal.jax.sqp.qp_subproblem import solve_qp_subproblem
from septal.jax.sqp.schema import SQPConfig

# ---------------------------------------------------------------------------
# Filter storage
# ---------------------------------------------------------------------------


def empty_filter(size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(filter_theta, filter_f)`` with every slot empty (``+inf``)."""
    inf = jnp.full(size, jnp.inf, dtype=jnp.float64)
    return inf, inf


def filter_acceptable(
    theta: jnp.ndarray,
    f: jnp.ndarray,
    filter_theta: jnp.ndarray,
    filter_f: jnp.ndarray,
    theta_max: jnp.ndarray | float,
) -> jnp.ndarray:
    """Is the pair ``(theta, f)`` acceptable to the filter?  (W&B 2005, eq. 2.10)

    Acceptable when ``theta < theta_max`` and, for every stored entry ``j``,
    ``theta < θ_j`` or ``f < f_j``.  Empty slots (``θ_j = +inf``) are
    masked out explicitly so no ``inf − inf`` arithmetic occurs.
    """
    occupied = jnp.isfinite(filter_theta)
    dominated = (theta >= filter_theta) & (f >= filter_f) & occupied
    return (theta < theta_max) & ~jnp.any(dominated)


def augment_filter(
    filter_theta: jnp.ndarray,
    filter_f: jnp.ndarray,
    theta_k: jnp.ndarray,
    f_k: jnp.ndarray,
    gamma_theta: float,
    gamma_f: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add the current iterate to the filter with margins.  (W&B 2005, eq. 2.17)

    The stored entry is ``((1 − γ_θ)·θ_k, f_k − γ_f·θ_k)``.  Entries dominated
    by the new one are cleared; the new entry is written into the first empty
    slot, or evicts the largest-``θ`` entry when the filter is full.
    """
    theta_new = (1.0 - gamma_theta) * theta_k
    f_new = f_k - gamma_f * theta_k

    occupied = jnp.isfinite(filter_theta)
    dominated = occupied & (filter_theta >= theta_new) & (filter_f >= f_new)
    filter_theta = jnp.where(dominated, jnp.inf, filter_theta)
    filter_f = jnp.where(dominated, jnp.inf, filter_f)

    empty = ~jnp.isfinite(filter_theta)
    first_empty = jnp.argmax(empty)  # first True, or 0 when none
    largest_theta = jnp.argmax(filter_theta)
    idx = jnp.where(jnp.any(empty), first_empty, largest_theta)

    return filter_theta.at[idx].set(theta_new), filter_f.at[idx].set(f_new)


# ---------------------------------------------------------------------------
# Sufficient-decrease tests
# ---------------------------------------------------------------------------


def switching_condition(
    alpha: jnp.ndarray,
    grad_f_d: jnp.ndarray,
    theta_k: jnp.ndarray,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Switching condition (W&B 2005, eq. 2.13).

    With ``m(α) = α·∇fᵀd`` the paper requires ``m(α) < 0`` and
    ``(−m(α))^{s_f} · α^{1−s_f} > δ · θ_k^{s_θ}``.  The left-hand side
    simplifies to ``α · (−∇fᵀd)^{s_f}``, which is what is evaluated here.
    When it holds the trial must satisfy the Armijo condition on ``f``
    (f-type step); otherwise the sufficient-decrease test (2.11) applies.
    """
    descent = grad_f_d < 0.0
    lhs = alpha * jnp.power(jnp.maximum(-grad_f_d, 0.0), cfg.filter_s_f)
    rhs = cfg.filter_delta * jnp.power(theta_k, cfg.filter_s_theta)
    return descent & (lhs > rhs)


def min_step_size(
    grad_f_d: jnp.ndarray,
    theta_k: jnp.ndarray,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Minimum trial step length ``α_min`` (W&B 2005, eq. 2.16).

    Backtracking below this value cannot produce an acceptable point, so the
    line search is abandoned and the feasibility restoration phase entered.
    For a descent direction::

        α_min = γ_α · min{ γ_θ,  γ_f·θ_k / (−∇fᵀd),  δ·θ_k^{s_θ} / (−∇fᵀd)^{s_f} }

    otherwise ``α_min = γ_α · γ_θ``.  At a feasible point (``θ_k = 0``) with a
    descent direction this is zero: restoration is never needed there.
    """
    descent = grad_f_d < 0.0
    # Finite, strictly positive divisor in every branch so no 0/0 appears
    # (the non-descent branch is discarded by the final jnp.where).
    neg_gd = jnp.where(descent, -grad_f_d, 1.0)
    term_f = cfg.filter_gamma_f * theta_k / neg_gd
    term_sw = (
        cfg.filter_delta
        * jnp.power(theta_k, cfg.filter_s_theta)
        / jnp.power(neg_gd, cfg.filter_s_f)
    )
    descent_min = jnp.minimum(cfg.filter_gamma_theta, jnp.minimum(term_f, term_sw))
    return cfg.filter_gamma_alpha * jnp.where(
        descent, descent_min, cfg.filter_gamma_theta
    )


# ---------------------------------------------------------------------------
# Backtracking line search on the filter
# ---------------------------------------------------------------------------


def _eval_trial(
    x: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    p: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    n_g: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(θ, f)`` at the box-clipped trial point ``clip(x + α·d, lb, ub)``."""
    x_t = jnp.clip(x + alpha * d, lb, ub)
    f_t = jnp.asarray(objective(x_t, p)).reshape(())
    if n_g > 0 and constraints is not None:
        g_t = constraints(x_t, p).reshape(n_g)
        theta_t = constraint_violation(g_t, lhs, rhs)
    else:
        theta_t = jnp.zeros((), dtype=jnp.float64)
    return theta_t, f_t


def judge_trial(
    theta_t: jnp.ndarray,
    f_t: jnp.ndarray,
    alpha: jnp.ndarray,
    grad_f_d: jnp.ndarray,
    theta_k: jnp.ndarray,
    f_k: jnp.ndarray,
    filter_theta: jnp.ndarray,
    filter_f: jnp.ndarray,
    theta_max: jnp.ndarray,
    cfg: SQPConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accept or reject one trial pair ``(θ_t, f_t)`` (W&B 2005, Alg. I steps 4.3–4.4).

    The trial must be acceptable to the filter (2.10) and then satisfy either

    * **f-type**: the switching condition (2.13) holds and the Armijo condition
      ``f_t ≤ f_k + η_f·α·∇fᵀd`` (2.14) holds — the filter is *not* augmented
      afterwards; or
    * **h-type**: the switching condition fails and the sufficient-decrease
      test ``θ_t ≤ (1−γ_θ)θ_k  or  f_t ≤ f_k − γ_f·θ_k`` (2.11) holds.

    A trial for which the switching condition holds but Armijo fails is
    rejected outright (it is not demoted to the h-type test).

    Returns
    -------
    accepted, is_f_type : bool scalars
    """
    in_filter = filter_acceptable(theta_t, f_t, filter_theta, filter_f, theta_max)
    switching = switching_condition(alpha, grad_f_d, theta_k, cfg)
    armijo = f_t <= f_k + cfg.filter_eta_f * alpha * grad_f_d
    suff_dec = (theta_t <= (1.0 - cfg.filter_gamma_theta) * theta_k) | (
        f_t <= f_k - cfg.filter_gamma_f * theta_k
    )
    f_type = switching & armijo
    h_type = ~switching & suff_dec
    return in_filter & (f_type | h_type), in_filter & f_type


def filter_line_search(
    x: jnp.ndarray,
    d: jnp.ndarray,
    p: jnp.ndarray,
    theta_k: jnp.ndarray,
    f_k: jnp.ndarray,
    grad_f_d: jnp.ndarray,
    filter_theta: jnp.ndarray,
    filter_f: jnp.ndarray,
    theta_max: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    n_g: int,
    cfg: SQPConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backtracking line search with filter acceptance (W&B 2005, Alg. I step 4).

    Starting from ``α = cfg.line_search_alpha0`` and shrinking by
    ``cfg.line_search_beta``, each trial ``clip(x + α·d, lb, ub)`` is judged by
    :func:`judge_trial`.  The loop stops when a trial is accepted, when ``α``
    drops below :func:`min_step_size` (2.16) or after ``cfg.max_line_search``
    trials.

    Returns
    -------
    alpha : scalar
        Accepted step length, ``0`` when no trial was accepted.
    accepted : bool scalar
        ``False`` means the caller should enter feasibility restoration.
    is_f_type : bool scalar
        ``True`` for an f-type step (the filter is **not** augmented).
    """
    beta = cfg.line_search_beta
    alpha0 = jnp.asarray(cfg.line_search_alpha0, dtype=x.dtype)
    alpha_min = min_step_size(grad_f_d, theta_k, cfg)

    def _judge(alpha: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        theta_t, f_t = _eval_trial(
            x, d, alpha, p, objective, constraints, lhs, rhs, lb, ub, n_g
        )
        return judge_trial(
            theta_t,
            f_t,
            alpha,
            grad_f_d,
            theta_k,
            f_k,
            filter_theta,
            filter_f,
            theta_max,
            cfg,
        )

    def cond_fn(carry: tuple) -> jnp.ndarray:
        alpha, i, accepted, _ = carry
        return (~accepted) & (i < cfg.max_line_search) & (alpha >= alpha_min)  # type: ignore[no-any-return]

    def body_fn(carry: tuple) -> tuple:
        alpha, i, _, _ = carry
        accepted, f_type = _judge(alpha)
        alpha_next = jnp.where(accepted, alpha, alpha * beta)
        return alpha_next, i + 1, accepted, f_type

    init = (alpha0, jnp.array(0, dtype=jnp.int32), jnp.array(False), jnp.array(False))
    alpha, _, accepted, f_type = jax.lax.while_loop(cond_fn, body_fn, init)
    alpha = jnp.where(accepted, alpha, jnp.zeros_like(alpha0))
    return alpha, accepted, accepted & f_type


# ---------------------------------------------------------------------------
# Second-order correction
# ---------------------------------------------------------------------------


def soc_correction(
    x: jnp.ndarray,
    d: jnp.ndarray,
    p: jnp.ndarray,
    theta_k: jnp.ndarray,
    f_k: jnp.ndarray,
    grad_f_d: jnp.ndarray,
    g_val: jnp.ndarray,
    jac_g: jnp.ndarray,
    hessian: jnp.ndarray,
    grad_f: jnp.ndarray,
    lam_prev: jnp.ndarray,
    filter_theta: jnp.ndarray,
    filter_f: jnp.ndarray,
    theta_max: jnp.ndarray,
    objective: Callable,
    constraints: Callable,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    n: int,
    n_g: int,
    cfg: SQPConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Second-order correction steps (Wächter & Biegler 2006, §2.4, step A-5.3).

    Counteracts the Maratos effect: when the full step ``x + α₀·d`` increases
    the constraint violation because of curvature, the QP is re-solved with
    the constraint values shifted by the linearisation error observed at the
    trial point,

        g_soc = g_k + [ g(x_t) − g_k − J_k (x_t − x_k) ],

    and the corrected trial ``clip(x + α₀·d_soc, lb, ub)`` is judged with the
    **original** ``α₀`` and ``∇fᵀd`` (:func:`judge_trial`).  Up to
    ``cfg.soc_max_iter`` corrections are tried, continuing only while the
    corrected trial's θ shrinks to at most ``cfg.soc_kappa`` × the previous
    trial's θ.  The caller decides whether to invoke this at all (first trial
    rejected and ``θ(x + α₀·d) ≥ θ_k``).

    Returns
    -------
    accepted, is_f_type : bool scalars
    d_soc : (n,)   last corrected direction (finite even when rejected)
    lam_soc : (n_g,)  multipliers of the last corrected QP
    """
    alpha0 = jnp.asarray(cfg.line_search_alpha0, dtype=x.dtype)
    x_first = jnp.clip(x + alpha0 * d, lb, ub)
    theta_first = constraint_violation(constraints(x_first, p).reshape(n_g), lhs, rhs)

    def _lin_error(x_t: jnp.ndarray) -> jnp.ndarray:
        return constraints(x_t, p).reshape(n_g) - g_val - jac_g @ (x_t - x)  # type: ignore[no-any-return]

    def cond_fn(carry: tuple) -> jnp.ndarray:
        k, _, _, _, _, accepted, keep_going = carry
        return (~accepted) & keep_going & (k < cfg.soc_max_iter)  # type: ignore[no-any-return]

    def body_fn(carry: tuple) -> tuple:
        k, x_t, theta_prev, _, _, _, _ = carry
        g_soc = g_val + _lin_error(x_t)
        d_soc, lam_soc = solve_qp_subproblem(
            hessian,
            grad_f,
            jac_g,
            g_soc,
            x,
            lb,
            ub,
            lhs,
            rhs,
            n,
            n_g,
            cfg,
            lam_prev=lam_prev,
        )
        x_soc = jnp.clip(x + alpha0 * d_soc, lb, ub)
        theta_soc, f_soc = _eval_trial(
            x, d_soc, alpha0, p, objective, constraints, lhs, rhs, lb, ub, n_g
        )
        accepted, f_type = judge_trial(
            theta_soc,
            f_soc,
            alpha0,
            grad_f_d,
            theta_k,
            f_k,
            filter_theta,
            filter_f,
            theta_max,
            cfg,
        )
        keep_going = theta_soc <= cfg.soc_kappa * theta_prev
        return k + 1, x_soc, theta_soc, d_soc, lam_soc, accepted, keep_going

    init = (
        jnp.array(0, dtype=jnp.int32),
        x_first,
        theta_first,
        d,
        lam_prev,
        jnp.array(False),
        jnp.array(True),
    )
    _, _, _, d_soc, lam_soc, accepted, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    # f_type is recomputed from the final trial so that a rejected loop reports False.
    theta_last, f_last = _eval_trial(
        x, d_soc, alpha0, p, objective, constraints, lhs, rhs, lb, ub, n_g
    )
    _, f_type = judge_trial(
        theta_last,
        f_last,
        alpha0,
        grad_f_d,
        theta_k,
        f_k,
        filter_theta,
        filter_f,
        theta_max,
        cfg,
    )
    return accepted, accepted & f_type, d_soc, lam_soc


# ---------------------------------------------------------------------------
# Feasibility restoration phase
# ---------------------------------------------------------------------------


def restoration_direction(
    x: jnp.ndarray,
    p: jnp.ndarray,
    g_val: jnp.ndarray,
    jac_g: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    n: int,
    n_g: int,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Gauss–Newton feasibility restoration step (W&B 2005, §3.2 / Alg. I step 9).

    Minimises the linearised squared violation of the currently violated
    constraints while keeping the satisfied ones linearly feasible and the
    iterate inside the box::

        min_d  ½ ‖ W (r + J d) ‖²  +  ½ ζ ‖d‖²
        s.t.   lhs_i − g_i ≤ J_i d ≤ rhs_i − g_i   for satisfied rows i
               lb − x ≤ d ≤ ub − x

    where ``r_i = max(g_i − rhs_i, 0) − max(lhs_i − g_i, 0)`` is the signed
    violation and ``W = diag(r ≠ 0)``.  Violated rows get their bounds
    relaxed to ``±inf`` so the quadratic objective, not the constraints,
    drives them toward feasibility — this keeps the QP feasible even when the
    linearised constraints are inconsistent.  The QP is solved with the same
    ADMM + polishing pipeline as the main SQP subproblem.
    """
    r = jnp.maximum(g_val - rhs, 0.0) - jnp.maximum(lhs - g_val, 0.0)
    w = (r != 0.0).astype(jnp.float64)
    jac_w = jac_g * w[:, None]
    hess_r = jac_w.T @ jac_g + cfg.restoration_reg * jnp.eye(n, dtype=jnp.float64)
    grad_r = jac_w.T @ r
    lhs_rel = jnp.where(w > 0.0, -jnp.inf, lhs)
    rhs_rel = jnp.where(w > 0.0, jnp.inf, rhs)
    d_r, _ = solve_qp_subproblem(
        hess_r, grad_r, jac_g, g_val, x, lb, ub, lhs_rel, rhs_rel, n, n_g, cfg
    )
    return d_r


def restoration_line_search(
    x: jnp.ndarray,
    d: jnp.ndarray,
    p: jnp.ndarray,
    theta_k: jnp.ndarray,
    filter_theta: jnp.ndarray,
    filter_f: jnp.ndarray,
    theta_max: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    n_g: int,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Backtracking on the restoration direction.

    A trial ``clip(x + α·d, lb, ub)`` is accepted when it reduces the
    violation by the filter margin, ``θ(α) ≤ (1 − γ_θ)·θ_k`` **and**
    ``θ(α) < θ_k`` strictly, and is acceptable to the filter — the exit
    condition of the restoration phase in W&B 2005.  The strict inequality
    matters when restoration is entered at a feasible point (``θ_k = 0``,
    possible only with an inexact QP step): no trial can qualify, ``α = 0``
    is returned and the caller treats it as a stall instead of accepting a
    zero step forever.  Returns ``α``, or ``0`` if no trial within
    ``cfg.max_line_search`` qualifies.
    """
    beta = cfg.line_search_beta
    alpha0 = jnp.asarray(cfg.line_search_alpha0, dtype=x.dtype)
    theta_suff = (1.0 - cfg.filter_gamma_theta) * theta_k

    def _ok(alpha: jnp.ndarray) -> jnp.ndarray:
        theta_t, f_t = _eval_trial(
            x, d, alpha, p, objective, constraints, lhs, rhs, lb, ub, n_g
        )
        in_filter = filter_acceptable(theta_t, f_t, filter_theta, filter_f, theta_max)
        return in_filter & (theta_t <= theta_suff) & (theta_t < theta_k)

    def cond_fn(carry: tuple) -> jnp.ndarray:
        _, i, accepted = carry
        return (~accepted) & (i < cfg.max_line_search)  # type: ignore[no-any-return]

    def body_fn(carry: tuple) -> tuple:
        alpha, i, _ = carry
        accepted = _ok(alpha)
        return jnp.where(accepted, alpha, alpha * beta), i + 1, accepted

    init = (alpha0, jnp.array(0, dtype=jnp.int32), jnp.array(False))
    alpha, _, accepted = jax.lax.while_loop(cond_fn, body_fn, init)
    return jnp.where(accepted, alpha, jnp.zeros_like(alpha0))
