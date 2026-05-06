"""
KKT residual computation and convergence check for the SQP solver.

Convergence is declared when both:

    ‖proj_{[lb,ub]}(x - ∇_xL) - x‖_∞  ≤  ε_stat   (stationarity, projected gradient)
    max(max(g-rhs, 0), max(lhs-g, 0))_∞             ≤  ε_feas   (primal feasibility)

The projected-gradient form of stationarity correctly handles active box
constraints: when x[i]=lb[i] the Lagrangian gradient may be positive (pointing
into the feasible region), which is NOT a KKT violation.  Using the raw
‖∇_xL‖_∞ would incorrectly flag such iterates as non-stationary.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from septal.jax.sqp.schema import SQPConfig


def kkt_residuals(
    x: jnp.ndarray,
    lam: jnp.ndarray,
    p: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    constraint_lhs: jnp.ndarray | None,
    constraint_rhs: jnp.ndarray | None,
    n_g: int,
    lb: jnp.ndarray | None = None,
    ub: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute KKT stationarity and primal feasibility residuals.

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    p:
        Parameter vector, shape ``(m,)``.
    objective:
        ``f(x, p) -> scalar``.
    constraints:
        ``g(x, p) -> (n_g,)`` or ``None``.
    constraint_lhs, constraint_rhs:
        Constraint bounds, each of shape ``(n_g,)`` or ``None``.
    n_g:
        Number of general constraints.
    lb, ub:
        Decision-variable bounds, each shape ``(n,)``.  When provided, the
        stationarity metric uses the projected-gradient residual
        ``‖proj_{[lb,ub]}(x - ∇_xL) - x‖_∞`` which correctly handles active
        box constraints.  Pass ``None`` to fall back to ``‖∇_xL‖_∞``.

    Returns
    -------
    stationarity : jnp.ndarray
        Scalar projected-gradient (or raw) stationarity residual.
    feasibility : jnp.ndarray
        Scalar constraint violation in the L∞ norm.
    """
    grad_f = jax.grad(objective)(x, p)

    if n_g > 0 and constraints is not None:
        g_val = constraints(x, p).reshape(n_g)
        jac_g = jax.jacfwd(constraints)(x, p).reshape(n_g, x.shape[0])
        grad_lag = grad_f + jac_g.T @ lam

        rhs = jnp.asarray(constraint_rhs).reshape(n_g)
        lhs = jnp.asarray(constraint_lhs).reshape(n_g)
        feas = jnp.maximum(
            jnp.max(jnp.maximum(g_val - rhs, 0.0)),
            jnp.max(jnp.maximum(lhs - g_val, 0.0)),
        )
    else:
        grad_lag = grad_f
        feas = jnp.zeros(())

    if lb is not None and ub is not None:
        # Projected-gradient stationarity: accounts for active box constraints.
        # At a KKT point x* with active lb[i]: grad_lag[i] >= 0 is not a violation.
        # proj(x - grad_lag, lb, ub) - x collapses to 0 in that case.
        proj_grad = jnp.clip(x - grad_lag, lb, ub) - x
        stationarity = jnp.max(jnp.abs(proj_grad))
    else:
        stationarity = jnp.max(jnp.abs(grad_lag))
    return stationarity, feas


def is_converged(
    stationarity: jnp.ndarray,
    feasibility: jnp.ndarray,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Return ``True`` iff both KKT tolerances are satisfied."""
    return (stationarity <= cfg.tol_stationarity) & (feasibility <= cfg.tol_feasibility)
