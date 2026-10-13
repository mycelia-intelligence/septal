"""
Core JAX SQP solver.

Two variants are provided:

sqp_solve_single(problem, x0, p, cfg)
    Uses ``jax.lax.while_loop`` — exits as soon as KKT tolerances are met.
    JIT-compilable but **not vmappable** (variable iteration count).
    Suitable for single-problem solves.

sqp_solve_scan(problem, x0, p, cfg)
    Uses ``jax.lax.scan`` — always runs ``cfg.max_iter`` steps with a
    ``converged`` mask that freezes already-converged iterates.
    Fully **vmappable** and GPU-efficient for batched parametric solves.

make_solver(problem, cfg) -> solve_fn
    Returns a JIT-compiled ``solve_fn(x0, p)`` built with the scan-based
    variant.  Pass it to ``jax.vmap`` to batch over parameters.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from septal.jax.sqp.convergence import is_converged
from septal.jax.sqp.filter import (
    _eval_trial,
    augment_filter,
    empty_filter,
    filter_line_search,
    restoration_direction,
    restoration_line_search,
    soc_correction,
)
from septal.jax.sqp.hessian import (
    bfgs_update,
    lagrangian_grad,
    regularised_lagrangian_hessian,
)
from septal.jax.sqp.line_search import (
    backtracking_line_search,
    constraint_violation,
    l1_merit,
    merit_directional_deriv,
    update_penalty,
)
from septal.jax.sqp.qp_subproblem import solve_qp_subproblem
from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig, SQPResult, SQPState

# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------


def init_sqp_state(
    x0: jnp.ndarray,
    p: jnp.ndarray,
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
) -> SQPState:
    """Build the initial ``SQPState`` from a starting guess and parameter.

    Parameters
    ----------
    x0:
        Initial primal guess, shape ``(n,)``.
    p:
        Parameter vector, shape ``(m,)``.  Pass ``jnp.zeros(0)`` for
        parameter-free problems.
    problem:
        Parametric NLP definition.
    cfg:
        Solver configuration.

    Returns
    -------
    SQPState
    """
    n = problem.n_decision
    n_g = problem.n_constraints or 0
    x0 = jnp.asarray(x0, dtype=jnp.float64).reshape(n)
    p = (
        jnp.asarray(p, dtype=jnp.float64).reshape(problem.n_params)
        if problem.n_params > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )

    f0 = jnp.asarray(problem.objective(x0, p)).reshape(())
    lam0 = jnp.zeros(n_g, dtype=jnp.float64)
    H0 = cfg.bfgs_init_scale * jnp.eye(n, dtype=jnp.float64)
    gl0 = lagrangian_grad(x0, lam0, p, problem.objective, problem.constraints, n_g)

    lhs = (
        jnp.asarray(problem.constraint_lhs, dtype=jnp.float64).reshape(n_g)
        if n_g > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )
    rhs = (
        jnp.asarray(problem.constraint_rhs, dtype=jnp.float64).reshape(n_g)
        if n_g > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )
    g0 = (
        problem.constraints(x0, p).reshape(n_g)
        if n_g > 0 and problem.constraints is not None
        else jnp.zeros(0, dtype=jnp.float64)
    )

    merit0 = l1_merit(
        f0, g0, lhs, rhs, jnp.array(cfg.penalty_init, dtype=jnp.float64), n_g
    )
    lb_arr = jnp.asarray(problem.lb, dtype=jnp.float64).reshape(n)
    ub_arr = jnp.asarray(problem.ub, dtype=jnp.float64).reshape(n)

    # Projected-gradient stationarity (dual-free at k=0 since lam0=0).
    grad_f0 = jax.grad(problem.objective)(x0, p)
    stat0 = jnp.max(jnp.abs(jnp.clip(x0 - grad_f0, lb_arr, ub_arr) - x0))

    # Initial feasibility
    if n_g > 0 and problem.constraints is not None:
        feas0 = jnp.maximum(
            jnp.max(jnp.maximum(g0 - rhs, 0.0)),
            jnp.max(jnp.maximum(lhs - g0, 0.0)),
        )
        theta0 = constraint_violation(g0, lhs, rhs)
    else:
        feas0 = jnp.zeros((), dtype=jnp.float64)
        theta0 = jnp.zeros((), dtype=jnp.float64)

    # Filter initialisation (W&B 2005): F_0 = {θ ≥ θ_max}, θ_max = c·max(1, θ(x₀)).
    filter_theta0, filter_f0 = empty_filter(cfg.filter_size)
    theta_max0 = cfg.filter_theta_max_factor * jnp.maximum(1.0, theta0)

    return SQPState(
        x=x0,
        params_p=p,
        lam=lam0,
        hessian=H0,
        grad_lag=gl0,
        f_val=f0,
        penalty=jnp.array(cfg.penalty_init, dtype=jnp.float64),
        merit=merit0,
        stationarity=stat0,
        feasibility=feas0,
        iteration=jnp.array(0, dtype=jnp.int32),
        converged=is_converged(stat0, feas0, cfg),
        merit_window=jnp.full(cfg.nonmonotone_window, merit0, dtype=jnp.float64),
        stagnation_count=jnp.array(0, dtype=jnp.int32),
        alpha_last=jnp.array(1.0, dtype=jnp.float64),
        filter_theta=filter_theta0,
        filter_f=filter_f0,
        theta_max=theta_max0,
        n_restoration=jnp.array(0, dtype=jnp.int32),
        n_soc=jnp.array(0, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Single SQP step (scan-body signature: (state, _) -> (state, None))
# ---------------------------------------------------------------------------


def make_sqp_step(
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
) -> Callable[[SQPState, None], tuple[SQPState, None]]:
    """Return a scan-compatible SQP step function closed over *problem* and *cfg*.

    The returned function is pure JAX — no Python conditionals on traced
    values — so it can be used inside ``lax.scan`` and ``lax.while_loop``.
    Already-converged states are masked out (identity update).
    """
    n = problem.n_decision
    n_g = problem.n_constraints or 0
    lb = jnp.asarray(problem.lb, dtype=jnp.float64).reshape(n)
    ub = jnp.asarray(problem.ub, dtype=jnp.float64).reshape(n)
    lhs = (
        jnp.asarray(problem.constraint_lhs, dtype=jnp.float64).reshape(n_g)
        if n_g > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )
    rhs = (
        jnp.asarray(problem.constraint_rhs, dtype=jnp.float64).reshape(n_g)
        if n_g > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )

    def _do_step(state: SQPState) -> SQPState:
        x = state.x
        p = state.params_p

        # 1. Gradients and constraint evaluation
        f_val, grad_f = jax.value_and_grad(problem.objective)(x, p)
        f_val = jnp.asarray(f_val).reshape(())
        grad_f = grad_f.reshape(n)

        if n_g > 0 and problem.constraints is not None:
            g_val = problem.constraints(x, p).reshape(n_g)
            jac_g = jax.jacfwd(problem.constraints)(x, p).reshape(n_g, n)
        else:
            g_val = jnp.zeros(0, dtype=jnp.float64)
            jac_g = jnp.zeros((0, n), dtype=jnp.float64)

        # 2. Solve QP subproblem — warm-start ADMM dual from previous multipliers
        d, lam_new = solve_qp_subproblem(
            state.hessian,
            grad_f,
            jac_g,
            g_val,
            x,
            lb,
            ub,
            lhs,
            rhs,
            n,
            n_g,
            cfg,
            lam_prev=state.lam,
        )

        # 3. Penalty update — decrease slowly when already feasible, increase otherwise
        penalty_new = update_penalty(
            lam_new,
            state.penalty,
            n_g,
            cfg.penalty_eps,
            state.feasibility,
            cfg.penalty_decrease_factor,
        )

        # 4. Step acceptance — static branch on cfg.step_acceptance.
        theta_k = (
            constraint_violation(g_val, lhs, rhs)
            if n_g > 0
            else jnp.zeros((), dtype=jnp.float64)
        )
        filter_theta_new, filter_f_new = state.filter_theta, state.filter_f
        used_restoration = jnp.array(False)

        if cfg.step_acceptance == "filter":
            # 4a. Wächter–Biegler line-search filter (Alg. I, step 4).
            grad_f_d = grad_f @ d
            alpha_f, accepted, f_type = filter_line_search(
                x,
                d,
                p,
                theta_k,
                f_val,
                grad_f_d,
                state.filter_theta,
                state.filter_f,
                state.theta_max,
                problem.objective,
                problem.constraints,
                lhs,
                rhs,
                lb,
                ub,
                n_g,
                cfg,
            )

            # 4b. Second-order correction (Wächter & Biegler 2006, step A-5.3):
            #     when the *first* trial was rejected and did not reduce θ, re-solve
            #     the QP with the constraint linearisation error folded in.  An
            #     accepted SOC trial replaces the backtracking result.
            d_used, lam_used, soc_accepted = d, lam_new, jnp.array(False)
            if cfg.soc_max_iter > 0 and n_g > 0 and problem.constraints is not None:
                alpha0 = jnp.asarray(cfg.line_search_alpha0, dtype=x.dtype)
                theta_first, _ = _eval_trial(
                    x,
                    d,
                    alpha0,
                    p,
                    problem.objective,
                    problem.constraints,
                    lhs,
                    rhs,
                    lb,
                    ub,
                    n_g,
                )
                first_rejected = (~accepted) | (alpha_f < alpha0)
                need_soc = first_rejected & (theta_first >= theta_k)

                def _soc(_: None) -> tuple:
                    return soc_correction(
                        x,
                        d,
                        p,
                        theta_k,
                        f_val,
                        grad_f_d,
                        g_val,
                        jac_g,
                        state.hessian,
                        grad_f,
                        lam_new,
                        state.filter_theta,
                        state.filter_f,
                        state.theta_max,
                        problem.objective,
                        problem.constraints,  # type: ignore[arg-type]  # guarded above
                        lhs,
                        rhs,
                        lb,
                        ub,
                        n,
                        n_g,
                        cfg,
                    )

                def _no_soc(_: None) -> tuple:
                    return jnp.array(False), jnp.array(False), d, lam_new

                soc_accepted, soc_f_type, d_soc, lam_soc = jax.lax.cond(
                    need_soc, _soc, _no_soc, None
                )
                alpha_f = jnp.where(soc_accepted, alpha0, alpha_f)
                d_used = jnp.where(soc_accepted, d_soc, d)
                lam_used = jnp.where(soc_accepted, lam_soc, lam_new)
                f_type = jnp.where(soc_accepted, soc_f_type, f_type)
                accepted = accepted | soc_accepted

            # 4c. Feasibility restoration when no trial step was acceptable.
            #     lax.cond keeps the second QP off the hot path for
            #     sqp_solve_single; under jax.vmap it lowers to a select and
            #     BOTH branches execute for every batch element.
            if n_g > 0 and problem.constraints is not None:

                def _restore(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                    d_r = restoration_direction(
                        x, p, g_val, jac_g, lhs, rhs, lb, ub, n, n_g, cfg
                    )
                    alpha_r = restoration_line_search(
                        x,
                        d_r,
                        p,
                        theta_k,
                        state.filter_theta,
                        state.filter_f,
                        state.theta_max,
                        problem.objective,
                        problem.constraints,
                        lhs,
                        rhs,
                        lb,
                        ub,
                        n_g,
                        cfg,
                    )
                    return alpha_r, d_r

                def _keep(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                    return alpha_f, d_used

                alpha, d_used = jax.lax.cond(accepted, _keep, _restore, None)
                used_restoration = ~accepted
            else:
                alpha = alpha_f

            # 4d. Augment the filter after an h-type step or on restoration
            #     (Alg. I step 6 / eq. 2.17).  f-type steps leave it unchanged.
            augment = (accepted & ~f_type) | used_restoration
            aug_theta, aug_f = augment_filter(
                state.filter_theta,
                state.filter_f,
                theta_k,
                f_val,
                cfg.filter_gamma_theta,
                cfg.filter_gamma_f,
            )
            filter_theta_new = jnp.where(augment, aug_theta, state.filter_theta)
            filter_f_new = jnp.where(augment, aug_f, state.filter_f)
        else:
            # 4e. L1 merit, non-monotone Armijo backtracking.
            #     Reference level: max merit over the last M iterates.
            reference_merit = jnp.max(state.merit_window)
            dir_deriv = merit_directional_deriv(
                grad_f, d, g_val, lhs, rhs, penalty_new, n_g
            )
            alpha = backtracking_line_search(
                x,
                d,
                p,
                reference_merit,
                dir_deriv,
                penalty_new,
                problem.objective,
                problem.constraints,
                lhs,
                rhs,
                n_g,
                cfg,
            )
            d_used, lam_used, soc_accepted = d, lam_new, jnp.array(False)

        # 5. Primal update — project onto the variable box. The QP encodes the
        #    box only as step inequalities solved approximately by ADMM, so the
        #    step can drift marginally outside [lb, ub]; clipping the iterate
        #    guarantees the returned solution is bound-feasible and makes the
        #    projected-gradient stationarity residual below exact.
        x_new = x + alpha * d_used
        x_new = jnp.clip(x_new, lb, ub)

        # 6. New function values
        f_new = jnp.asarray(problem.objective(x_new, p)).reshape(())
        if n_g > 0 and problem.constraints is not None:
            g_new = problem.constraints(x_new, p).reshape(n_g)
        else:
            g_new = jnp.zeros(0, dtype=jnp.float64)
        merit_new = l1_merit(f_new, g_new, lhs, rhs, penalty_new, n_g)

        # 7. Lagrangian gradient at new iterate (for BFGS and stationarity)
        grad_lag_new = lagrangian_grad(
            x_new, lam_used, p, problem.objective, problem.constraints, n_g
        )

        # 8. Hessian update: exact Lagrangian Hessian (jax.hessian) or damped BFGS.
        #    Exact path: regularise to PD via eigenvalue shift (inertia correction).
        if cfg.use_exact_hessian:
            H_new = regularised_lagrangian_hessian(
                x_new,
                lam_used,
                p,
                problem.objective,
                problem.constraints,
                n_g,
                cfg.hess_reg_delta,
                cfg.hess_reg_min,
            )
        else:
            s = x_new - x
            y = grad_lag_new - state.grad_lag
            H_new = bfgs_update(
                state.hessian, s, y, cfg.bfgs_skip_tol, cfg.bfgs_max_cond
            )

        # 9. Stationarity: projected-gradient KKT residual.
        stat_new = jnp.max(jnp.abs(jnp.clip(x_new - grad_lag_new, lb, ub) - x_new))
        if n_g > 0:
            feas_new = jnp.maximum(
                jnp.max(jnp.maximum(g_new - rhs, 0.0)),
                jnp.max(jnp.maximum(lhs - g_new, 0.0)),
            )
        else:
            feas_new = jnp.zeros((), dtype=jnp.float64)
        conv_new = is_converged(stat_new, feas_new, cfg)

        # 10. Non-monotone merit window: roll left, insert new merit at end.
        merit_window_new = jnp.roll(state.merit_window, -1).at[-1].set(merit_new)

        # 11. Stagnation detection and reset.
        #     Count consecutive iterations where the step was essentially zero.
        is_stagnating = alpha < cfg.stagnation_alpha_tol
        stagnation_new = jnp.where(
            is_stagnating,
            state.stagnation_count + 1,
            jnp.array(0, dtype=jnp.int32),
        )
        do_reset = stagnation_new >= cfg.stagnation_patience

        # When stagnating: reset Hessian to scaled identity and deflate penalty.
        H_reset = cfg.bfgs_init_scale * jnp.eye(n, dtype=H_new.dtype)
        H_final = jnp.where(do_reset & cfg.stagnation_reset_hessian, H_reset, H_new)
        penalty_reset = jnp.array(cfg.penalty_init, dtype=jnp.float64)
        penalty_final = jnp.where(
            do_reset & cfg.stagnation_reset_penalty, penalty_reset, penalty_new
        )
        stagnation_final = jnp.where(
            do_reset, jnp.array(0, dtype=jnp.int32), stagnation_new
        )
        # A reset also forgets the filter history (a stale filter can block
        # every step from the reset Hessian).
        filter_theta_final = jnp.where(do_reset, jnp.inf, filter_theta_new)
        filter_f_final = jnp.where(do_reset, jnp.inf, filter_f_new)

        return SQPState(
            x=x_new,
            params_p=p,
            lam=lam_used,
            hessian=H_final,
            grad_lag=grad_lag_new,
            f_val=f_new,
            penalty=penalty_final,
            merit=merit_new,
            stationarity=stat_new,
            feasibility=feas_new,
            iteration=state.iteration + 1,
            converged=conv_new,
            merit_window=merit_window_new,
            stagnation_count=stagnation_final,
            alpha_last=alpha,
            filter_theta=filter_theta_final,
            filter_f=filter_f_final,
            theta_max=state.theta_max,
            n_restoration=state.n_restoration + used_restoration.astype(jnp.int32),
            n_soc=state.n_soc + soc_accepted.astype(jnp.int32),
        )

    def sqp_step(state: SQPState, _: None) -> tuple[SQPState, None]:
        new_state = _do_step(state)
        # Mask: freeze state once converged
        final_state = jax.tree.map(
            lambda n_val, o_val: jnp.where(state.converged, o_val, n_val),
            new_state,
            state,
        )
        # Ensure converged flag is monotone (True stays True)
        return (
            final_state._replace(converged=state.converged | new_state.converged),
            None,
        )

    return sqp_step


# ---------------------------------------------------------------------------
# Full solvers
# ---------------------------------------------------------------------------


def sqp_solve_scan(
    problem: ParametricNLPProblem,
    x0: jnp.ndarray,
    p: jnp.ndarray,
    cfg: SQPConfig,
) -> SQPState:
    """SQP solve using ``lax.scan`` — vmappable over parameters.

    Runs exactly ``cfg.max_iter`` steps; converged iterates are frozen
    after convergence via the masked update in :func:`make_sqp_step`.

    Parameters
    ----------
    problem:
        Parametric NLP definition (static — not a JAX array).
    x0:
        Initial guess, shape ``(n,)``.
    p:
        Parameter vector, shape ``(m,)``.
    cfg:
        Solver configuration (static).

    Returns
    -------
    SQPState
        Final solver state after ``cfg.max_iter`` iterations.
    """
    state = init_sqp_state(x0, p, problem, cfg)
    step_fn = make_sqp_step(problem, cfg)
    final_state, _ = jax.lax.scan(step_fn, state, None, length=cfg.max_iter)
    return final_state


def sqp_solve_single(
    problem: ParametricNLPProblem,
    x0: jnp.ndarray,
    p: jnp.ndarray,
    cfg: SQPConfig,
) -> SQPState:
    """SQP solve using ``lax.while_loop`` — exits early on convergence.

    JIT-compilable but **not directly vmappable**.  Use
    :func:`sqp_solve_scan` (or :func:`make_solver`) for batched solves.

    Parameters
    ----------
    problem, x0, p, cfg:
        See :func:`sqp_solve_scan`.

    Returns
    -------
    SQPState
        Final solver state.
    """
    state = init_sqp_state(x0, p, problem, cfg)
    step_fn = make_sqp_step(problem, cfg)

    def cond_fn(s: SQPState) -> jnp.ndarray:
        return (~s.converged) & (s.iteration < cfg.max_iter)

    def body_fn(s: SQPState) -> SQPState:
        new_s, _ = step_fn(s, None)
        return new_s

    return jax.lax.while_loop(cond_fn, body_fn, state)


def select_initial_points(
    problem: ParametricNLPProblem,
    p: jnp.ndarray,
    n_select: int,
    n_candidates: int = 256,
    penalty: float = 10.0,
    seed: int = 42,
    inf_fallback: float = 1e3,
) -> jnp.ndarray:
    """Rank Sobol candidates by L1 merit and return the best ``n_select``.

    Generates ``n_candidates`` points from a scrambled Sobol sequence within
    the problem's decision bounds, evaluates the L1 merit function

        φ(x) = f(x, p) + penalty · Σᵢ [max(gᵢ(x,p) − rhsᵢ, 0)
                                        + max(lhsᵢ − gᵢ(x,p), 0)]

    at each candidate via ``jax.vmap``, and returns the ``n_select`` points
    with the lowest merit value (best first).

    Parameters
    ----------
    problem:
        Parametric NLP definition.
    p:
        Fixed parameter vector, shape ``(n_params,)``.  A representative
        value should be used when this is called for a batch (e.g. the
        batch mean).
    n_select:
        Number of points to return.  Must be ≤ ``n_candidates``.
    n_candidates:
        Total Sobol candidates to evaluate.  Should be a power of 2 for
        best Sobol uniformity.  Default 256.
    penalty:
        L1 penalty weight ρ for constraint violation used in ranking.
        Should be ≥ the expected optimal dual variable magnitude.
        Default 10.0.
    seed:
        Sobol scramble seed.  Default 42.
    inf_fallback:
        Range used to replace infinite bounds when generating the Sobol
        sequence.  If lb[i] = −∞ but ub[i] is finite, the effective lower
        bound is ``ub[i] − inf_fallback``, and vice versa.  Default 1e3.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_select, n_decision)`` — best candidates ranked by L1
        merit (lowest merit = index 0).
    """
    n = problem.n_decision
    n_g = problem.n_constraints or 0

    # ── Clamp infinite bounds so Sobol stays in a finite hypercube ──────────
    lb_np = np.asarray(problem.lb, dtype=np.float64).reshape(n)
    ub_np = np.asarray(problem.ub, dtype=np.float64).reshape(n)
    lb_safe = np.where(
        np.isfinite(lb_np),
        lb_np,
        np.where(np.isfinite(ub_np), ub_np - inf_fallback, -inf_fallback),
    )
    ub_safe = np.where(
        np.isfinite(ub_np),
        ub_np,
        np.where(np.isfinite(lb_np), lb_np + inf_fallback, inf_fallback),
    )

    # ── Sobol generation (host-side NumPy) ───────────────────────────────────
    sobol = qmc.Sobol(d=n, scramble=True, seed=seed).random(n_candidates)
    candidates = jnp.asarray(
        lb_safe + (ub_safe - lb_safe) * sobol, dtype=jnp.float64
    )  # (n_candidates, n)

    # ── L1 merit ingredients ─────────────────────────────────────────────────
    lhs = (
        jnp.asarray(problem.constraint_lhs, dtype=jnp.float64).reshape(n_g)
        if n_g > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )
    rhs = (
        jnp.asarray(problem.constraint_rhs, dtype=jnp.float64).reshape(n_g)
        if n_g > 0
        else jnp.zeros(0, dtype=jnp.float64)
    )
    rho = jnp.array(penalty, dtype=jnp.float64)
    p_arr = jnp.asarray(p, dtype=jnp.float64)

    # ── Evaluate merit at all candidates via vmap ────────────────────────────
    def _merit(x: jnp.ndarray) -> jnp.ndarray:
        f_val = jnp.asarray(problem.objective(x, p_arr)).reshape(())
        if n_g > 0 and problem.constraints is not None:
            g_val = problem.constraints(x, p_arr).reshape(n_g)
        else:
            g_val = jnp.zeros(0, dtype=jnp.float64)
        return l1_merit(f_val, g_val, lhs, rhs, rho, n_g)

    merits = jax.vmap(_merit)(candidates)  # (n_candidates,)

    # ── Return the n_select candidates with lowest merit ─────────────────────
    idx = jnp.argsort(merits)[:n_select]
    return candidates[idx]  # (n_select, n)


def make_batch_solver(
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
    mode: str = "vmap",
) -> Callable[[jnp.ndarray, jnp.ndarray], SQPState]:
    """Return a batched solver that parallelises over a leading batch dimension.

    Parameters
    ----------
    problem:
        Parametric NLP (captured in closure, treated as static).
    cfg:
        Solver configuration (captured in closure, treated as static).
    mode:
        ``"vmap"`` — vectorises the batch on one device via
        ``jax.jit(jax.vmap(...))``.  Supports any batch size and is safe
        to call from inside an outer ``jax.pmap`` (vmap composes with pmap).

        ``"pmap"`` — distributes instances across XLA devices via
        ``jax.pmap``.  Each device runs one instance of
        ``sqp_solve_scan``.  Batch size along axis-0 must equal
        ``jax.local_device_count()`` (or a multiple thereof when chunked
        externally).  **Cannot be called from inside another pmap context**
        (nested pmap is not supported by JAX).

    Returns
    -------
    Callable
        ``solve_batch(x0_batch, p_batch) -> SQPState``

        * ``x0_batch`` : ``(N, n)``  for vmap  (N = any batch size),
                         ``(D, n)``  for pmap  (D = device count).
        * ``p_batch``  : ``(N, n_params)`` / ``(D, n_params)``
                         correspondingly.  Use ``jnp.zeros((N, 0))``
                         for parameter-free problems.

    Notes
    -----
    The default mode is ``"vmap"``.  Use ``"pmap"`` only at the outermost
    call site — never inside an existing ``jax.pmap`` context.

    For ``pmap`` with N instances and D devices, call this function once
    to get ``solve_batch``, then chunk::

        D = jax.local_device_count()
        x0_chunks = x0_batch.reshape(N // D, D, n)
        p_chunks  = p_batch.reshape(N // D, D, n_params)
        states = [solve_batch(x0_chunks[i], p_chunks[i])
                  for i in range(N // D)]

    To simulate multiple CPU devices set the environment variable
    *before* importing JAX::

        XLA_FLAGS=--xla_force_host_platform_device_count=8 python script.py
    """

    def _solve_single(x0: jnp.ndarray, p: jnp.ndarray) -> SQPState:
        return sqp_solve_scan(problem, x0, p, cfg)

    if mode == "vmap":

        @jax.jit
        def _solve_vmap(x0_batch: jnp.ndarray, p_batch: jnp.ndarray) -> SQPState:
            return jax.vmap(_solve_single)(x0_batch, p_batch)

        return _solve_vmap
    elif mode == "pmap":
        return jax.pmap(_solve_single)  # type: ignore[no-any-return]
    else:
        raise ValueError(f"mode must be 'vmap' or 'pmap', got {mode!r}")


def make_solver(
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
) -> Callable[[jnp.ndarray, jnp.ndarray], SQPState]:
    """Return a JIT-compiled, vmappable ``solve_fn(x0, p) -> SQPState``.

    The returned callable uses the scan-based solver and can be passed
    directly to ``jax.vmap`` for batched parametric solves::

        solve_one = make_solver(problem, cfg)
        results   = jax.vmap(solve_one)(x0_batch, params_batch)

    Parameters
    ----------
    problem:
        Parametric NLP (captured in closure, treated as static).
    cfg:
        Solver configuration (captured in closure, treated as static).

    Returns
    -------
    Callable
        JIT-compiled ``(x0, p) -> SQPState``.
    """

    @jax.jit
    def _solve(x0: jnp.ndarray, p: jnp.ndarray) -> SQPState:
        return sqp_solve_scan(problem, x0, p, cfg)

    return _solve


# ---------------------------------------------------------------------------
# State → SQPResult conversion
# ---------------------------------------------------------------------------


def state_to_result(
    state: SQPState,
    problem: ParametricNLPProblem,
    timing: float = 0.0,
) -> SQPResult:
    """Convert a final ``SQPState`` to a user-facing ``SQPResult``."""
    n_g = problem.n_constraints or 0

    g_final = None
    if n_g > 0 and problem.constraints is not None:
        g_final = problem.constraints(state.x, state.params_p)

    msg = "converged" if bool(state.converged) else "max_iter reached"
    return SQPResult(
        success=bool(state.converged),
        objective=state.f_val,
        decision_variables=state.x,
        multipliers=state.lam,
        constraints=g_final,
        iterations=int(state.iteration),
        kkt_stationarity=float(state.stationarity),
        kkt_feasibility=float(state.feasibility),
        timing=timing,
        message=msg,
    )


def batch_state_to_result(
    state: SQPState,
    problem: ParametricNLPProblem,
    timing: float = 0.0,
) -> SQPResult:
    """Convert a batched ``SQPState`` (from vmap) to a batched ``SQPResult``.

    All array fields will have a leading batch dimension ``N``.
    """
    n_g = problem.n_constraints or 0

    g_final = None
    if n_g > 0 and problem.constraints is not None:
        g_final = jax.vmap(lambda x, p: problem.constraints(x, p))(  # type: ignore[misc]
            state.x, state.params_p
        )

    msg = "batch solve complete"
    return SQPResult(
        success=state.converged,
        objective=state.f_val,
        decision_variables=state.x,
        multipliers=state.lam,
        constraints=g_final,
        iterations=state.iteration,
        kkt_stationarity=state.stationarity,
        kkt_feasibility=state.feasibility,
        timing=timing,
        message=msg,
    )
