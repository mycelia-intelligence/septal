"""
Data structures for the JAX-based SQP solver.

Parametric NLP standard form
-----------------------------
    min_{x}  f(x, p)
    s.t.     lhs <= g(x, p) <= rhs
             lb  <=  x      <= ub

where p is a fixed parameter vector.  The batch use-case solves the same
structural problem for N different values of p simultaneously via jax.vmap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------


@dataclass
class ParametricNLPProblem:
    """Parametric NLP: min f(x, p) s.t. lhs <= g(x, p) <= rhs, lb <= x <= ub.

    Parameters
    ----------
    objective:
        JAX function ``f(x, p) -> scalar``.
    bounds:
        ``[lb, ub]`` — list/tuple of two JAX/NumPy arrays, each of shape
        ``(n_decision,)``.
    n_decision:
        Dimension of ``x``.
    n_params:
        Dimension of ``p``.  Use 0 for parameter-free problems.
    constraints:
        Optional JAX function ``g(x, p) -> (n_constraints,)``.
    constraint_lhs:
        Lower bound on ``g(x, p)``; shape ``(n_constraints,)``.
        Use ``-jnp.inf`` entries for one-sided upper bounds.
    constraint_rhs:
        Upper bound on ``g(x, p)``; shape ``(n_constraints,)``.
        Typically ``0`` for constraints of the form ``g(x, p) <= 0``.
    n_constraints:
        Number of general constraints.  Inferred from *constraint_lhs* /
        *constraint_rhs* if not supplied.
    """

    objective: Callable
    bounds: list[Any]
    n_decision: int
    n_params: int
    constraints: Callable | None = None
    constraint_lhs: Any | None = None
    constraint_rhs: Any | None = None
    n_constraints: int | None = None

    def __post_init__(self) -> None:
        if self.n_constraints is None:
            if self.constraint_lhs is not None:
                self.n_constraints = len(jnp.asarray(self.constraint_lhs).reshape(-1))
            elif self.constraint_rhs is not None:
                self.n_constraints = len(jnp.asarray(self.constraint_rhs).reshape(-1))
            else:
                self.n_constraints = 0

    @property
    def lb(self) -> Any:
        return self.bounds[0]

    @property
    def ub(self) -> Any:
        return self.bounds[1]

    @property
    def has_constraints(self) -> bool:
        return self.constraints is not None and (self.n_constraints or 0) > 0


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@dataclass
class SQPConfig:
    """Hyper-parameters for the JAX SQP solver.

    Parameters
    ----------
    max_iter:
        Maximum number of SQP outer iterations.
    tol_stationarity:
        KKT stationarity convergence tolerance.
    tol_feasibility:
        KKT primal feasibility convergence tolerance.
    line_search_beta:
        Backtracking factor (0 < beta < 1).  Step is multiplied by *beta*
        at each rejected trial.
    line_search_c:
        Armijo sufficient-decrease constant (typically 1e-4).
    line_search_alpha0:
        Initial trial step length.
    max_line_search:
        Maximum backtracking iterations.
    penalty_eps:
        Additive margin above ``‖λ‖_∞`` when updating the penalty parameter.
    penalty_init:
        Initial penalty parameter ``ρ₀``.
    bfgs_init_scale:
        Diagonal scale of the initial Hessian approximation ``H₀ = scale·I``.
    bfgs_skip_tol:
        Skip BFGS update when ``|sᵀy| < tol * ‖s‖²`` (prevents rank-deficiency).
    osqp_tol:
        OSQP convergence tolerance (combined primal/dual).
    osqp_max_iter:
        Maximum OSQP iterations per QP subproblem.
    osqp_reg:
        Small regularisation added to QP Hessian diagonal for PSD guarantee.
    """

    max_iter: int = 100
    tol_stationarity: float = 1e-6
    tol_feasibility: float = 1e-6
    line_search_beta: float = 0.5
    line_search_c: float = 1e-4
    line_search_alpha0: float = 1.0
    max_line_search: int = 30
    penalty_eps: float = 1e-3
    penalty_init: float = 1.0
    bfgs_init_scale: float = 1.0
    bfgs_skip_tol: float = 1e-10
    osqp_tol: float = 1e-7  # kept for API compatibility (unused by ADMM)
    osqp_max_iter: int = 4000  # kept for API compatibility (unused by ADMM)
    osqp_reg: float = 1e-7  # Hessian diagonal regularisation
    admm_rho: float = 1.0  # Initial ADMM penalty parameter ρ₀
    admm_n_iter: int = 500  # fixed ADMM iterations (scan-based, vmappable)
    admm_sigma: float = 1e-6  # Proximal regularisation on x-update (σ > 0)
    admm_alpha: float = 1.6  # Over-relaxation parameter (α ∈ (0,2); paper: 1.6)
    admm_rho_min: float = 1e-6  # Lower clip for adaptive ρ
    admm_rho_max: float = 1e6  # Upper clip for adaptive ρ
    admm_rho_update_interval: int = 25  # Iterations between ρ updates
    admm_adaptive_rho: bool = True  # Enable adaptive ρ updating
    admm_mu: float = 10.0  # Residual ratio threshold for ρ update (paper: 10)
    admm_tau: float = 2.0  # ρ scaling factor on update (paper: 2)
    admm_ruiz_iter: int = 10  # Ruiz diagonal equilibration iterations run
    # before ADMM.  Scales A rows/cols and Q
    # cols to unit ∞-norm so ρ is equally
    # effective for all constraint rows.
    # Set 0 to disable.
    admm_polish: bool = True  # OSQP-style polishing after ADMM (Eq 30,
    # Stellato et al. 2020).  Guesses the active
    # constraint set from the ADMM dual y and z,
    # then solves a reduced KKT system exactly.
    # Falls back to the ADMM result if the
    # polished solution is worse or non-finite.
    admm_polish_reg: float = 1e-8  # Regularisation for the polishing KKT system.
    # Added to the (1,1) block as σI (keeps Q PD)
    # and to inactive dual rows as ε (forces ŷᵢ≈0).
    admm_polish_refine: int = 3  # Iterative refinement steps after the polish
    # solve: Δsol = solve(KKT, rhs − KKT·sol);
    # sol += Δsol.  Corrects the σ/ε error.
    # Set 0 to disable refinement.
    admm_rho_eq_scale: float = 100.0  # Extra ρ multiplier for equality rows (l==u).
    # Equality rows converge as dual-ascent O(1/k)
    # without this boost; 100× drives them faster.
    bfgs_max_cond: float = 1e8  # Hessian condition-number reset threshold.
    # When cond(H) > threshold, H is reset to
    # mean(diag(H)) * I to prevent ill-conditioning.
    penalty_decrease_factor: float = 0.999  # Multiplicative shrink applied to the
    # penalty when the iterate is already
    # feasible, preventing penalty blow-up
    # at feasible non-optimal points.
    use_exact_hessian: bool = True  # Use exact Lagrangian Hessian via
    # jax.hessian instead of BFGS.  More
    # accurate on curved constraint manifolds
    # (e.g. hs006, hs039) but costs O(n²)
    # AD passes per iteration.

    # ── Non-convex robustness ────────────────────────────────────────────────

    hess_reg_delta: float = 1e-4  # Minimum eigenvalue target for the exact
    # Lagrangian Hessian (use_exact_hessian=True
    # path only).  When the Lagrangian Hessian
    # has eigenvalues below this threshold, an
    # additive shift tau*I is applied to make Q
    # positive definite before the QP subproblem.
    # Set 0 to disable.
    hess_reg_min: float = 1e-8  # Minimum regularisation always added to the
    # exact Hessian (prevents exactly-zero shift
    # when hess_reg_delta == 0).

    nonmonotone_window: int = 5  # Non-monotone line search memory (Grippo et
    # al. 1986).  The Armijo condition uses the
    # maximum merit over the last M iterates as
    # its reference level, allowing temporary
    # increases that help escape saddle points.
    # M=1 recovers the standard monotone Armijo.

    stagnation_patience: int = 30  # Number of consecutive near-zero steps
    # before triggering a stagnation reset.
    stagnation_alpha_tol: float = 1e-6  # Step-length threshold: alpha < tol
    # counts as a stagnated iteration.
    stagnation_reset_hessian: bool = (
        True  # Reset H to bfgs_init_scale * I on stagnation.
    )
    stagnation_reset_penalty: bool = (
        True  # Deflate penalty to penalty_init on stagnation.
    )


# ---------------------------------------------------------------------------
# Solver state  (JAX pytree via NamedTuple)
# ---------------------------------------------------------------------------


class SQPState(NamedTuple):
    """Internal SQP iterate.  All fields are JAX arrays so the state is a
    valid JAX pytree compatible with ``lax.scan`` and ``jax.vmap``.

    Parameters
    ----------
    x:
        Current primal iterate, shape ``(n,)``.
    params_p:
        Parameter vector frozen into the state, shape ``(m,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    hessian:
        BFGS Hessian approximation, shape ``(n, n)``.
    grad_lag:
        Lagrangian gradient ``∇_x L(x, λ, p)`` at current iterate, shape ``(n,)``.
    f_val:
        Current objective value ``f(x, p)``, scalar.
    penalty:
        L1 merit penalty parameter ``ρ``, scalar.
    merit:
        Current L1 merit function value ``φ(x; ρ)``, scalar.
    stationarity:
        KKT stationarity residual at current iterate, scalar.
    feasibility:
        KKT primal feasibility residual at current iterate, scalar.
    iteration:
        Current iteration counter (integer scalar).
    converged:
        Boolean scalar — ``True`` when KKT tolerances are satisfied.
    """

    x: jnp.ndarray
    params_p: jnp.ndarray
    lam: jnp.ndarray
    hessian: jnp.ndarray
    grad_lag: jnp.ndarray
    f_val: jnp.ndarray
    penalty: jnp.ndarray
    merit: jnp.ndarray
    stationarity: jnp.ndarray
    feasibility: jnp.ndarray
    iteration: jnp.ndarray
    converged: jnp.ndarray
    merit_window: jnp.ndarray  # shape (nonmonotone_window,) — rolling buffer of
    # recent merit values for non-monotone line search.
    stagnation_count: jnp.ndarray  # integer scalar — consecutive near-zero steps.
    alpha_last: jnp.ndarray  # scalar — step length accepted on last iteration.


# ---------------------------------------------------------------------------
# Solve result
# ---------------------------------------------------------------------------


@dataclass
class SQPResult:
    """Result returned by the JAX SQP solver.

    Parameters
    ----------
    success:
        ``True`` if the solver converged within ``max_iter`` iterations.
    objective:
        Optimal objective value ``f(x*, p)``.
    decision_variables:
        Optimal primal solution ``x*``, shape ``(n,)`` or ``(N, n)`` for batches.
    multipliers:
        Optimal dual variables ``λ*`` for general constraints.
    constraints:
        Value of ``g(x*, p)``; ``None`` when the problem has no constraints.
    iterations:
        Number of SQP outer iterations performed.
    kkt_stationarity:
        Final KKT stationarity residual.
    kkt_feasibility:
        Final KKT primal feasibility residual.
    timing:
        Wall-clock solve time in seconds.
    message:
        Human-readable status string.
    """

    success: Any
    objective: Any
    decision_variables: Any
    multipliers: Any = None
    constraints: Any | None = None
    iterations: Any = 0
    kkt_stationarity: Any = 0.0
    kkt_feasibility: Any = 0.0
    timing: float = 0.0
    message: str = ""
