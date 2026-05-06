"""
Standard NLP schema for septal.

Defines the input problem (NLPProblem) and output result (SolveResult)
dataclasses used across all solver backends.

NLP standard form:
    min  f(x)
    s.t. lhs <= g(x) <= rhs
         lb  <=  x   <= ub
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class NLPProblem:
    """Describes a nonlinear programme in standard form.

    Parameters
    ----------
    objective:
        JAX function ``f(x) -> scalar`` (or CasADi-compatible callback).
        Will be wrapped for the chosen backend automatically.
    bounds:
        ``[lb, ub]`` — list/tuple of two JAX/NumPy arrays, each of shape
        ``(n_decision,)``.
    constraints:
        Optional JAX function ``g(x) -> (n_g,)`` vector.  Paired with
        *constraint_lhs* / *constraint_rhs* to form ``lhs <= g(x) <= rhs``.
    constraint_lhs:
        Lower bound on ``g(x)``; shape ``(n_g,)`` or ``(n_g, 1)``.
        ``-inf`` entries are unconstrained from below.
    constraint_rhs:
        Upper bound on ``g(x)``; shape ``(n_g,)`` or ``(n_g, 1)``.
        Typically ``0`` for inequality constraints ``g(x) <= 0``.
    n_decision:
        Dimension of ``x``.  Inferred from *bounds* if not supplied.
    n_starts:
        Number of multi-start initial guesses to generate / evaluate.
    """

    objective: Callable
    bounds: list[Any]
    constraints: Callable | None = None
    constraint_lhs: Any | None = None
    constraint_rhs: Any | None = None
    n_decision: int | None = None
    n_starts: int = 5

    def __post_init__(self) -> None:
        if self.n_decision is None:
            self.n_decision = len(self.bounds[0])

    # Convenience ----------------------------------------------------------------

    @property
    def lb(self) -> Any:
        return self.bounds[0]

    @property
    def ub(self) -> Any:
        return self.bounds[1]

    @property
    def has_constraints(self) -> bool:
        return self.constraints is not None


@dataclass
class SolveResult:
    """Result returned by every septal solver.

    Parameters
    ----------
    success:
        ``True`` if the solver converged to a (locally) optimal feasible point.
    objective:
        Optimal objective value ``f(x*)``.
    decision_variables:
        Optimal primal solution ``x*``.
    constraints:
        Value of ``g(x*)``; ``None`` when the problem has no constraints.
    message:
        Solver status string (e.g. IPOPT's *return_status*).
    timing:
        Wall-clock time spent in the solver (seconds).
    n_solves:
        Number of multi-start solves that returned feasible solutions.
    """

    success: bool
    objective: Any
    decision_variables: Any
    constraints: Any | None = None
    message: str = ""
    timing: float = 0.0
    n_solves: int = 0
