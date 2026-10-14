"""
L1 exact penalty merit function and backtracking line search.

This is the ``step_acceptance="merit"`` path.  The default ``"filter"`` path
(Wächter & Biegler 2005) lives in :mod:`septal.jax.sqp.filter` and reuses
:func:`constraint_violation` from here as its θ measure.

Merit function
--------------
    φ(x; ρ) = f(x, p) + ρ · Σᵢ [max(gᵢ(x,p) - rhsᵢ, 0) + max(lhsᵢ - gᵢ(x,p), 0)]

This is the standard L1 exact penalty.  For ρ > ‖λ*‖_∞ the merit function
is exact (local minima of the NLP are local minima of the merit function).

Directional derivative estimate (Nocedal & Wright §18.3)
---------------------------------------------------------
For the SQP search direction d_k:

    D_φ(x_k; d_k) ≈ ∇f_kᵀ d_k  −  ρ · v(x_k)

where v(x_k) = Σᵢ max violations.  This is the standard sufficient descent
condition used to check acceptability of the penalty parameter.

Penalty update
--------------
The penalty is updated non-decreasingly when the iterate is infeasible
(standard exact-penalty condition: ρ > ‖λ‖_∞ ensures exactness).  When the
iterate is already feasible, the penalty is allowed to shrink slowly by a
factor ``penalty_decrease_factor`` toward max(‖λ‖_∞ + ε, 0).  This prevents
the penalty from growing without bound at feasible non-optimal points, which
would otherwise make the merit function indifferent to objective improvement.

Line search
-----------
Armijo backtracking via ``jax.lax.while_loop`` (JIT- and vmap-compatible):

    α ← α · β  until  φ(x + α·d) ≤ φ(x) + c·α·D_φ

If the directional derivative D_φ ≥ 0 (non-descent direction, which can
happen when the ADMM inner solve has not converged), the line search is
skipped and α = 0 is returned to avoid taking a damaging step.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from septal.jax.sqp.schema import SQPConfig

# ---------------------------------------------------------------------------
# L1 merit helpers
# ---------------------------------------------------------------------------


def constraint_violation(
    g_val: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """L1 constraint violation: Σᵢ max(gᵢ-rhsᵢ, 0) + max(lhsᵢ-gᵢ, 0)."""
    upper_viol = jnp.sum(jnp.maximum(g_val - rhs, 0.0))
    lower_viol = jnp.sum(jnp.maximum(lhs - g_val, 0.0))
    return upper_viol + lower_viol


def l1_merit(
    f_val: jnp.ndarray,
    g_val: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    penalty: jnp.ndarray,
    n_g: int,
) -> jnp.ndarray:
    """Evaluate the L1 exact penalty merit function φ(x; ρ).

    Parameters
    ----------
    f_val:
        Objective value ``f(x, p)``, scalar.
    g_val:
        Constraint value ``g(x, p)``, shape ``(n_g,)``.
        Pass ``jnp.zeros(0)`` when there are no general constraints.
    lhs, rhs:
        Constraint bounds, each of shape ``(n_g,)``.
    penalty:
        Penalty parameter ``ρ``, scalar.
    n_g:
        Number of general constraints.

    Returns
    -------
    jnp.ndarray
        Scalar merit value.
    """
    if n_g > 0:
        viol = constraint_violation(g_val, lhs, rhs)
        return f_val + penalty * viol
    return f_val


def merit_directional_deriv(
    grad_f: jnp.ndarray,
    d: jnp.ndarray,
    g_val: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    penalty: jnp.ndarray,
    n_g: int,
) -> jnp.ndarray:
    """Estimate the directional derivative of the L1 merit along the SQP step.

    Uses the standard SQP approximation:
        D_φ(x; d) ≈ ∇f(x)ᵀ d  −  ρ · v(x)

    This is negative (descent) when ρ ≥ ‖λ‖_∞.
    """
    gd = grad_f @ d
    if n_g > 0:
        viol = constraint_violation(g_val, lhs, rhs)
        return gd - penalty * viol
    return gd


def update_penalty(
    lam: jnp.ndarray,
    penalty: jnp.ndarray,
    n_g: int,
    eps: float,
    feasibility: jnp.ndarray,
    decrease_factor: float,
) -> jnp.ndarray:
    """Penalty update with feasibility-aware decrease.

    When the iterate is infeasible (feasibility > eps) the penalty grows
    non-decreasingly: ρ_{k+1} = max(‖λ_k‖_∞ + ε, ρ_k).

    When the iterate is already feasible the penalty is allowed to shrink
    slowly: ρ_{k+1} = max(‖λ_k‖_∞ + ε, decrease_factor · ρ_k).  This
    prevents ρ from growing without bound at feasible non-optimal iterates,
    which would otherwise dominate the merit function and stall objective
    improvement.

    Parameters
    ----------
    lam:
        Current multiplier estimates, shape ``(n_g,)``.
    penalty:
        Current penalty parameter, scalar.
    n_g:
        Number of general constraints.
    eps:
        Small margin: target = ‖λ‖_∞ + eps.
    feasibility:
        Current primal feasibility residual (L∞ constraint violation), scalar.
        Compared against ``eps`` to decide increase vs decrease.
    decrease_factor:
        Multiplicative factor in [0, 1) for slow penalty reduction when
        feasible.  Use 1.0 to disable (reverts to non-decreasing behaviour).
    """
    if n_g > 0:
        lam_inf = jnp.max(jnp.abs(lam))
        target = lam_inf + eps
        penalty_up = jnp.maximum(target, penalty)
        penalty_down = jnp.maximum(target, decrease_factor * penalty)
        return jnp.where(feasibility < eps, penalty_down, penalty_up)
    return penalty


# ---------------------------------------------------------------------------
# Backtracking line search
# ---------------------------------------------------------------------------


def backtracking_line_search(
    x: jnp.ndarray,
    d: jnp.ndarray,
    p: jnp.ndarray,
    reference_merit: jnp.ndarray,
    dir_deriv: jnp.ndarray,
    penalty: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    n_g: int,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Armijo backtracking line search on the L1 merit function.

    Starts with ``α = cfg.line_search_alpha0`` and multiplies by
    ``cfg.line_search_beta`` until

        φ(x + α·d; ρ)  ≤  reference_merit + c · α · D_φ(x; d)

    or until ``cfg.max_line_search`` trials are exhausted.

    ``reference_merit`` is typically the maximum merit over the last M
    iterates (non-monotone line search, Grippo et al. 1986).  Passing
    the current merit recovers standard monotone Armijo.

    If ``dir_deriv ≥ 0`` (non-descent direction — can occur when the ADMM
    inner solve stalls and returns an inaccurate step), the search is skipped
    and ``α = 0`` is returned so that no update is taken this iteration.

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    d:
        Search direction, shape ``(n,)``.
    p:
        Parameter vector, shape ``(m,)``.
    reference_merit:
        Reference merit level for the Armijo condition.  For non-monotone
        line search, pass ``jnp.max(state.merit_window)``; for standard
        monotone, pass the current merit ``state.merit``.
    dir_deriv:
        Directional derivative estimate at ``x`` along ``d``, scalar.
    penalty:
        Penalty parameter ``ρ``, scalar.
    objective:
        ``f(x, p) -> scalar``.
    constraints:
        ``g(x, p) -> (n_g,)`` or ``None``.
    lhs, rhs:
        Constraint bounds.
    n_g:
        Number of general constraints.
    cfg:
        Solver configuration.

    Returns
    -------
    jnp.ndarray
        Accepted step length ``α``, scalar.  Returns 0 when dir_deriv ≥ 0.
    """
    beta = cfg.line_search_beta
    c = cfg.line_search_c
    alpha0 = jnp.array(cfg.line_search_alpha0, dtype=x.dtype)

    def _merit_at(alpha: jnp.ndarray) -> jnp.ndarray:
        x_new = x + alpha * d
        f_new = objective(x_new, p)
        if n_g > 0 and constraints is not None:
            g_new = constraints(x_new, p)
            return l1_merit(f_new, g_new, lhs, rhs, penalty, n_g)
        return jnp.asarray(f_new)

    def cond_fn(state: tuple) -> jnp.ndarray:
        alpha, i = state
        trial_merit = _merit_at(alpha)
        armijo = trial_merit <= reference_merit + c * alpha * dir_deriv
        # Continue if Armijo fails AND we have more trials
        return (~armijo) & (i < cfg.max_line_search)  # type: ignore[no-any-return]

    def body_fn(state: tuple) -> tuple:
        alpha, i = state
        return (alpha * beta, i + 1)

    alpha_ls, _ = jax.lax.while_loop(cond_fn, body_fn, (alpha0, 0))

    # If the direction is not a merit-function descent direction (dir_deriv >= 0),
    # taking any step risks increasing the merit.  Return alpha=0 to skip this
    # iteration rather than taking a damaging micro-step.
    is_descent = dir_deriv < 0.0
    return jnp.where(is_descent, alpha_ls, jnp.zeros_like(alpha0))
