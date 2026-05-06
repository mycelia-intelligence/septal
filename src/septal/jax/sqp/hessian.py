"""
Damped BFGS Hessian approximation (Powell, 1978).

The standard BFGS update is:

    H_{k+1} = H_k - (H_k s_k s_kᵀ H_k) / (s_kᵀ H_k s_k)
                  + (r_k r_kᵀ) / (s_kᵀ r_k)

where s_k = x_{k+1} - x_k and r_k is the (possibly damped) curvature vector.

Powell damping modifies y_k so that the updated Hessian remains positive
definite, which is required for the QP subproblem to be convex:

    θ_k = 1                         if  s_kᵀ y_k >= 0.2 · s_kᵀ H_k s_k
          0.8 s_kᵀ H_k s_k          otherwise
          ─────────────────────
          s_kᵀ H_k s_k - s_kᵀ y_k

    r_k = θ_k y_k + (1 - θ_k) H_k s_k

Condition-number reset
----------------------
After each BFGS update the spectral condition number of H is checked via
``jnp.linalg.eigvalsh``.  If cond(H) > ``max_cond`` the matrix is reset to
``mean(diag(H)) * I``.  This prevents the Hessian approximation from becoming
ill-conditioned on problems with curved equality manifolds where the full-space
Lagrangian Hessian is indefinite (e.g. hs006, hs039).

All operations are pure JAX — no Python control flow on traced values.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def bfgs_update(
    H: jnp.ndarray,
    s: jnp.ndarray,
    y: jnp.ndarray,
    skip_tol: float = 1e-10,
    max_cond: float = 1e8,
) -> jnp.ndarray:
    """Damped BFGS rank-2 Hessian update with condition-number safeguard.

    Parameters
    ----------
    H:
        Current Hessian approximation, shape ``(n, n)``.  Must be
        symmetric positive definite.
    s:
        Primal step ``x_{k+1} - x_k``, shape ``(n,)``.
    y:
        Lagrangian gradient difference ``∇L_{k+1} - ∇L_k``, shape ``(n,)``.
    skip_tol:
        If ``sᵀs < skip_tol`` the update is skipped (step was numerically zero).
    max_cond:
        If ``cond(H_new) > max_cond`` the Hessian is reset to
        ``mean(diag(H_new)) * I`` to prevent ill-conditioning.

    Returns
    -------
    jnp.ndarray
        Updated Hessian, shape ``(n, n)``.
    """
    Hs = H @ s
    sHs = s @ Hs
    sy = s @ y

    # Powell damping: blend y towards H@s so that sᵀr > 0
    theta = jnp.where(sy >= 0.2 * sHs, 1.0, 0.8 * sHs / (sHs - sy + 1e-30))
    r = theta * y + (1.0 - theta) * Hs
    sr = s @ r

    # BFGS rank-2 update
    H_new = H - jnp.outer(Hs, Hs) / (sHs + 1e-30) + jnp.outer(r, r) / (sr + 1e-30)

    # Skip update if the step was too small (avoids numerical blow-up)
    skip = (s @ s) < skip_tol
    H_new = jnp.where(skip, H, H_new)

    # Symmetrise to counteract floating-point drift
    H_new = 0.5 * (H_new + H_new.T)

    # Condition-number safeguard: reset to scaled identity when H becomes
    # ill-conditioned.  This protects against Hessian deterioration on
    # curved constraint manifolds where the Lagrangian Hessian is indefinite
    # on the full space (positive only on the constraint tangent space).
    eigs = jnp.linalg.eigvalsh(H_new)  # sorted ascending, O(n³)
    eig_max = eigs[-1]
    eig_min = jnp.maximum(eigs[0], 1e-30)
    cond = eig_max / eig_min
    reset_scale = jnp.mean(jnp.diag(H_new))
    H_new = jnp.where(cond > max_cond, reset_scale * jnp.eye(H.shape[0]), H_new)

    return H_new


def regularised_lagrangian_hessian(
    x: jnp.ndarray,
    lam: jnp.ndarray,
    p: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    n_g: int,
    hess_reg_delta: float = 1e-4,
    hess_reg_min: float = 1e-8,
) -> jnp.ndarray:
    """Exact Lagrangian Hessian with adaptive PD regularisation.

    Computes ``∇²_xx L(x, λ, p)`` via ``jax.hessian`` then applies an
    additive shift ``tau * I`` to ensure all eigenvalues are at least
    ``hess_reg_delta``.  This guarantees the QP subproblem receives a
    positive-definite Hessian regardless of the iterate's non-convexity.

    Parameters
    ----------
    x, lam, p, objective, constraints, n_g:
        See :func:`lagrangian_hessian`.
    hess_reg_delta:
        Target minimum eigenvalue.  Shift is applied only when the
        smallest eigenvalue is below this threshold.
    hess_reg_min:
        Minimum shift always added (even when all eigenvalues exceed
        ``hess_reg_delta``), preventing an exactly-zero shift that could
        leave borderline eigenvalues unprotected.

    Returns
    -------
    jnp.ndarray, shape ``(n, n)``
        Symmetrised, PD-regularised Lagrangian Hessian.
    """
    H = lagrangian_hessian(x, lam, p, objective, constraints, n_g)
    H = 0.5 * (H + H.T)  # symmetrise against AD drift
    eigs = jnp.linalg.eigvalsh(H)  # ascending order, O(n³)
    lam_min = eigs[0]
    # Shift: tau = max(hess_reg_min, -lam_min + hess_reg_delta)
    # Applied only when lam_min < hess_reg_delta; otherwise tau = hess_reg_min.
    tau_needed = jnp.maximum(hess_reg_min, -lam_min + hess_reg_delta)
    tau = jnp.where(lam_min < hess_reg_delta, tau_needed, hess_reg_min)
    return H + tau * jnp.eye(H.shape[0], dtype=H.dtype)


def lagrangian_hessian(
    x: jnp.ndarray,
    lam: jnp.ndarray,
    p: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    n_g: int,
) -> jnp.ndarray:
    """Compute the exact Lagrangian Hessian ∇²_xx L(x, λ, p) via ``jax.hessian``.

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    p:
        Parameter vector, shape ``(m,)``.
    objective:
        JAX callable ``f(x, p) -> scalar``.
    constraints:
        JAX callable ``g(x, p) -> (n_g,)`` or ``None``.
    n_g:
        Number of general constraints.

    Returns
    -------
    jnp.ndarray, shape ``(n, n)``
        ``∇²f(x, p) + Σᵢ λᵢ ∇²gᵢ(x, p)``.
    """

    def lagrangian(x_: jnp.ndarray) -> jnp.ndarray:
        f = objective(x_, p)
        if n_g > 0 and constraints is not None:
            g = constraints(x_, p)
            return f + lam @ g  # type: ignore[no-any-return]
        return f  # type: ignore[no-any-return]

    return jax.hessian(lagrangian)(x)  # type: ignore[no-any-return]


def lagrangian_grad(
    x: jnp.ndarray,
    lam: jnp.ndarray,
    p: jnp.ndarray,
    objective: Callable,
    constraints: Callable | None,
    n_g: int,
) -> jnp.ndarray:
    """Compute the Lagrangian gradient ∇_x L(x, λ, p).

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    p:
        Parameter vector, shape ``(m,)``.
    objective:
        JAX callable ``f(x, p) -> scalar``.
    constraints:
        JAX callable ``g(x, p) -> (n_g,)`` or ``None``.
    n_g:
        Number of general constraints.

    Returns
    -------
    jnp.ndarray, shape ``(n,)``
        ``∇f(x, p) + J_g(x, p)ᵀ λ``.
    """
    grad_f = jax.grad(objective)(x, p)
    if n_g > 0 and constraints is not None:
        jac_g = jax.jacfwd(constraints)(x, p)  # (n_g, n)
        return grad_f + jac_g.T @ lam  # type: ignore[no-any-return]
    return grad_f  # type: ignore[no-any-return]
