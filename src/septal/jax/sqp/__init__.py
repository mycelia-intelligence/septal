"""
septal.jax.sqp — JAX-based SQP solver for parametric NLPs.

Exports the public API needed to formulate and solve parametric NLPs:

    from septal.jax.sqp import (
        ParametricNLPProblem,
        SQPConfig,
        SQPResult,
        SQPState,
        ParametricSQPFactory,
    )

Float64 precision is enabled globally for this module on import so that the
SQP solver, BFGS Hessian updates and ADMM inner iterations all operate in
double precision.  JAX defaults to float32; without x64 enabled the BFGS
Hessian can become indefinite from accumulated rounding error.
"""

import jax

jax.config.update("jax_enable_x64", True)

from septal.jax.sqp.convergence import is_converged, kkt_residuals
from septal.jax.sqp.factory import ParametricSQPFactory
from septal.jax.sqp.hessian import bfgs_update, lagrangian_grad
from septal.jax.sqp.line_search import (
    backtracking_line_search,
    constraint_violation,
    l1_merit,
    update_penalty,
)
from septal.jax.sqp.qp_subproblem import form_qp_matrices, solve_qp_subproblem
from septal.jax.sqp.schema import (
    ParametricNLPProblem,
    SQPConfig,
    SQPResult,
    SQPState,
)
from septal.jax.sqp.solver import (
    batch_state_to_result,
    init_sqp_state,
    make_solver,
    sqp_solve_scan,
    sqp_solve_single,
    state_to_result,
)

__all__ = [
    # Schema
    "ParametricNLPProblem",
    "SQPConfig",
    "SQPResult",
    "SQPState",
    # Factory
    "ParametricSQPFactory",
    # Solver
    "make_solver",
    "sqp_solve_scan",
    "sqp_solve_single",
    "init_sqp_state",
    "state_to_result",
    "batch_state_to_result",
    # Convergence
    "kkt_residuals",
    "is_converged",
    # Hessian
    "bfgs_update",
    "lagrangian_grad",
    # Line search
    "l1_merit",
    "backtracking_line_search",
    "update_penalty",
    "constraint_violation",
    # QP subproblem
    "form_qp_matrices",
    "solve_qp_subproblem",
]
