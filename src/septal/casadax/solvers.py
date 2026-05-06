"""
NLP solver implementations.

Two backends are provided:

CasadiSolver
    Multi-start IPOPT via CasADi. Accepts pre-built CasADi-compatible
    objective and constraint callbacks (see :mod:`septal.casadax.callbacks`).

JaxSolver
    Bound-constrained multi-start L-BFGS-B via JAXopt. No constraint support;
    intended for box-constrained feasibility sub-problems.

Both classes expose a common ``solve(initial_guesses) -> SolveResult`` interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from casadi import MX, Function, nlpsol
from jax import jacfwd, jit, lax
from jaxopt import LBFGSB  # type: ignore[import-not-found]

from septal.casadax.schema import SolveResult
from septal.casadax.utilities import generate_initial_guess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal IPOPT helpers
# ---------------------------------------------------------------------------


def _ipopt_no_gcons(
    objective: Any,
    bounds: list[Any],
    initial_guess: Any,
) -> tuple:
    """Single-start IPOPT solve — unconstrained (box bounds only)."""
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]

    x = MX.sym("x", n_d, 1)
    F = Function("F", [x], [objective(x)])
    nlp = {"x": x, "f": F(x)}

    opts = {
        "ipopt": {"hessian_approximation": "limited-memory"},
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 150,
    }
    solver = nlpsol("solver", "ipopt", nlp, opts)
    solution = solver(x0=np.hstack(initial_guess), lbx=lb, ubx=ub)

    del nlp, F, x, lb, ub, opts
    return solver, solution


def _ipopt_gcons(
    objective: Any,
    constraints: Any,
    bounds: list[Any],
    initial_guess: Any,
    lhs: Any,
    rhs: Any,
) -> tuple:
    """Single-start IPOPT solve — with general inequality constraints."""
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]

    x = MX.sym("x", n_d, 1)
    F = Function("F", [x], [objective(x)])
    G = Function("G", [x], [constraints(x)])

    lbg = np.array(lhs)
    ubg = np.array(rhs)
    nlp = {"x": x, "f": F(x), "g": G(x)}

    opts = {
        "ipopt": {"hessian_approximation": "limited-memory"},
        "ipopt.print_level": 1,
        "print_time": 0,
        "ipopt.max_iter": 150,
    }
    solver = nlpsol("solver", "ipopt", nlp, opts)
    solution = solver(x0=np.hstack(initial_guess), lbx=lb, ubx=ub, lbg=lbg, ubg=ubg)

    del nlp, F, G, x, lb, ub, lbg, ubg, opts
    return solver, solution


# ---------------------------------------------------------------------------
# Internal JAXopt helper
# ---------------------------------------------------------------------------


def _lbfgsb_single(
    init,
    xs: Any,
    objective_func: Callable,
    bounds_: tuple,
    tol: float,
    objective_params: tuple = (),
) -> tuple:
    """Single L-BFGS-B solve from one initial point (scan-compatible)."""
    lbfgsb = LBFGSB(
        fun=objective_func,
        maxiter=200,
        use_gamma=True,
        verbose=False,
        linesearch="backtracking",
        decrease_factor=0.8,
        maxls=100,
        tol=tol,
    )
    problem = lbfgsb.run(xs, bounds_, *objective_params)
    return None, problem


def _assess_lbfgsb_solution(
    init,
    xs: Any,
    objective_func: Callable,
    args: tuple,
) -> tuple:
    return None, (objective_func(xs, *args), jacfwd(objective_func)(xs, *args))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSolver(ABC):
    """Common interface for septal solvers."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def solve(self, initial_guesses: Any) -> SolveResult:
        raise NotImplementedError

    @abstractmethod
    def initial_guess(self) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CasADi / IPOPT solver
# ---------------------------------------------------------------------------


class CasadiSolver(BaseSolver):
    """Multi-start IPOPT solver wrapping CasADi.

    Parameters
    ----------
    cfg:
        Config object with at least ``n_starts`` and ``max_solution_time``.
    objective_fn:
        CasADi-compatible callable (e.g. produced by :func:`~septal.casadax.callbacks.casadify_reverse`).
    bounds:
        ``[lb, ub]`` arrays, each of shape ``(n_decision,)``.
    constraints_fn:
        Optional CasADi-compatible constraint callable.
    constraint_lhs:
        Lower bound on ``g(x)``.
    constraint_rhs:
        Upper bound on ``g(x)``.
    """

    def __init__(
        self,
        cfg: Any,
        objective_fn: Callable,
        bounds: list[Any],
        constraints_fn: Callable | None = None,
        constraint_lhs: Any | None = None,
        constraint_rhs: Any | None = None,
    ) -> None:
        super().__init__(cfg)
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.constraints_fn = constraints_fn
        self.constraint_lhs = constraint_lhs
        self.constraint_rhs = constraint_rhs
        self.n_d = len(bounds[0])

    def initial_guess(self) -> Any:
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)

    def solve(self, initial_guesses: Any) -> SolveResult:
        n_starts = initial_guesses.shape[0]
        solutions = []

        has_cons = self.constraints_fn is not None

        for i in range(n_starts):
            x0 = np.array(initial_guesses[i, :]).squeeze()
            try:
                if has_cons:
                    solver, sol = _ipopt_gcons(
                        self.objective_fn,
                        self.constraints_fn,
                        self.bounds,
                        x0,
                        self.constraint_lhs,
                        self.constraint_rhs,
                    )
                else:
                    solver, sol = _ipopt_no_gcons(self.objective_fn, self.bounds, x0)
                if solver.stats()["success"]:
                    solutions.append((solver, sol))
                    if np.array(sol["f"]) <= 0:
                        break
            except Exception as exc:
                logger.warning("IPOPT start %d failed: %s", i, exc)

        return self._digest(solutions)

    def _digest(self, solutions: list) -> SolveResult:
        if not solutions:
            return SolveResult(
                success=False,
                objective=np.inf,
                decision_variables=None,
                message="All starts failed",
            )

        min_idx = np.argmin(np.vstack([s[1]["f"] for s in solutions]))
        solver_opt, sol_opt = solutions[min_idx]
        stats = solver_opt.stats()
        success: bool = bool(stats.get("success", False))
        t_wall = sum(v for k, v in stats.items() if "t_wall_" in k)
        objective = float(np.array(sol_opt["f"]).reshape(-1)[0])
        constraints = np.array(sol_opt["g"]).reshape(-1) if "g" in sol_opt else None
        decision = np.array(sol_opt["x"])
        message: str = str(stats.get("return_status", ""))

        if not success:
            if constraints is not None:
                objective = float(
                    np.maximum(objective, np.max(np.abs(constraints)))
                )
            logger.warning("CasadiSolver: %s", message)

        if (
            t_wall >= getattr(self.cfg, "max_solution_time", float("inf"))
            and not success
        ):
            logger.warning("CasadiSolver: max_solution_time exceeded (%.1f s)", t_wall)

        return SolveResult(
            success=success,
            objective=objective,
            decision_variables=decision,
            constraints=constraints,
            message=message,
            timing=t_wall,
            n_solves=len(solutions),
        )


# ---------------------------------------------------------------------------
# JAXopt / L-BFGS-B solver
# ---------------------------------------------------------------------------


class JaxSolver(BaseSolver):
    """Multi-start bound-constrained L-BFGS-B solver via JAXopt.

    Parameters
    ----------
    cfg:
        Config with ``n_starts`` and ``jax_opt_options.error_tol``.
    objective_fn:
        JAX callable ``f(x, *objective_args) -> scalar``.
    bounds:
        ``[lb, ub]`` arrays.
    objective_args:
        Extra arguments forwarded to *objective_fn* (JAX tracers are ok).
    """

    def __init__(
        self,
        cfg: Any,
        objective_fn: Callable,
        bounds: list[Any],
        objective_args: tuple = (),
    ) -> None:
        super().__init__(cfg)
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.objective_args = objective_args
        self.n_d = len(bounds[0])

    def initial_guess(self) -> Any:
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)

    @property
    def _tol(self) -> float:
        try:
            return float(self.cfg.jax_opt_options.error_tol)
        except AttributeError:
            return 1e-4

    def solve(self, initial_guesses: Any) -> SolveResult:
        tol = self._tol
        bounds_ = (self.bounds[0], self.bounds[1])
        obj_fn = self.objective_fn
        args = self.objective_args

        partial_solver = jit(
            partial(
                _lbfgsb_single,
                objective_func=obj_fn,
                bounds_=bounds_,
                tol=tol,
            )
        )

        _, (_, solutions) = lax.scan(
            lambda carry, x: (
                None,
                partial_solver(init=None, xs=x, objective_params=args),
            ),
            init=None,
            xs=initial_guesses,
        )

        assess = partial(_assess_lbfgsb_solution, objective_func=obj_fn, args=args)
        _, assessment = lax.scan(assess, init=None, xs=solutions.params)  # type: ignore[attr-defined, var-annotated]

        cond = solutions.state.error <= jnp.array([tol]).squeeze()  # type: ignore[attr-defined]
        mask = jnp.asarray(cond)
        obj_vals = assessment[0]
        grad_norms = jnp.linalg.norm(assessment[1], axis=1).squeeze()

        blended = (
            jnp.where(mask, obj_vals, jnp.minimum(obj_vals, grad_norms)),
            jnp.where(mask, grad_norms, jnp.inf),
        )

        arg_min = jnp.argmin(blended[0], axis=0)
        min_obj = blended[0][arg_min]
        stationary_err = solutions.state.error[arg_min]  # type: ignore[attr-defined]
        success = bool(jnp.linalg.norm(stationary_err) <= tol)

        return SolveResult(
            success=success,
            objective=float(min_obj),
            decision_variables=np.array(solutions.params[arg_min]),  # type: ignore[attr-defined]
            message="converged" if success else "stationary error above tolerance",
        )

    def get_status(self, stationary_error: Any) -> bool:
        return bool(jnp.linalg.norm(stationary_error) <= self._tol)
