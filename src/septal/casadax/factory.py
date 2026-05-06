"""
SolverFactory — creates the right solver backend from a configuration key.

Supported solver types
----------------------
``"general_constrained_nlp"``
    CasADi / IPOPT multi-start (via :class:`~septal.casadax.solvers.CasadiSolver`).

``"box_constrained_nlp"``
    JAXopt L-BFGS-B multi-start (via :class:`~septal.casadax.solvers.JaxSolver`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from septal.casadax.schema import NLPProblem, SolveResult
from septal.casadax.solvers import BaseSolver, CasadiSolver, JaxSolver


class SolverFactory:
    """Factory that instantiates and holds a solver for a given NLP.

    Parameters
    ----------
    cfg:
        Sub-config for the solver (must contain ``n_starts``, ``max_solution_time``,
        ``jax_opt_options.error_tol``, etc.).
    solver_type:
        One of ``"general_constrained_nlp"`` or ``"box_constrained_nlp"``.
    """

    def __init__(self, cfg: Any, solver_type: str) -> None:
        self.cfg = cfg
        self.solver_type = solver_type
        self.solver: BaseSolver | None = None
        self.objective_func: Callable | None = None
        self.constraints_func: Callable | None = None
        self.bounds: list[Any] | None = None
        self.constraint_lhs: Any | None = None
        self.constraint_rhs: Any | None = None

    # ------------------------------------------------------------------
    # Class-level constructor (preferred entry point)
    # ------------------------------------------------------------------

    @classmethod
    def from_problem(
        cls, cfg: Any, solver_type: str, problem: NLPProblem
    ) -> SolverFactory:
        """Build a factory and construct the solver from an :class:`~septal.casadax.schema.NLPProblem`.

        Parameters
        ----------
        cfg:
            Solver configuration.
        solver_type:
            ``"general_constrained_nlp"`` or ``"box_constrained_nlp"``.
        problem:
            The NLP to solve.

        Returns
        -------
        SolverFactory
            Factory with :attr:`solver` ready to call.
        """
        factory = cls(cfg, solver_type)
        factory.load_objective(problem.objective)
        factory.load_bounds(problem.bounds)
        factory.load_constraints(
            problem.constraints, problem.constraint_lhs, problem.constraint_rhs
        )
        factory.construct_solver()
        return factory

    @classmethod
    def from_method(
        cls,
        cfg: Any,
        solver_type: str,
        objective_func: Callable,
        bounds: list[Any],
        constraints_func: Callable | None = None,
        constraint_lhs: Any | None = None,
        constraint_rhs: Any | None = None,
    ) -> SolverFactory:
        """Build a factory from individual components (legacy / low-level entry).

        Parameters
        ----------
        cfg:
            Solver configuration.
        solver_type:
            Backend type.
        objective_func:
            Pre-built objective (CasADi callback or JAX callable).
        bounds:
            ``[lb, ub]``.
        constraints_func:
            Optional pre-built constraint callback.
        constraint_lhs, constraint_rhs:
            Bounds on ``g(x)``.

        Returns
        -------
        SolverFactory
        """
        factory = cls(cfg, solver_type)
        factory.load_objective(objective_func)
        factory.load_bounds(bounds)
        factory.load_constraints(constraints_func, constraint_lhs, constraint_rhs)
        factory.construct_solver()
        return factory

    # ------------------------------------------------------------------
    # Builder steps
    # ------------------------------------------------------------------

    def load_objective(self, objective_func: Callable) -> None:
        self.objective_func = objective_func

    def load_bounds(self, bounds: list[Any]) -> None:
        self.bounds = bounds

    def load_constraints(
        self,
        constraints_func: Callable | None,
        lhs: Any | None = None,
        rhs: Any | None = None,
    ) -> None:
        self.constraints_func = constraints_func
        self.constraint_lhs = lhs
        self.constraint_rhs = rhs

    def construct_solver(self) -> None:
        if self.objective_func is None:
            raise ValueError("objective_func must be set before construct_solver().")
        if self.bounds is None:
            raise ValueError("bounds must be set before construct_solver().")

        if self.solver_type == "general_constrained_nlp":
            self.solver = CasadiSolver(
                self.cfg,
                self.objective_func,
                self.bounds,
                constraints_fn=self.constraints_func,
                constraint_lhs=self.constraint_lhs,
                constraint_rhs=self.constraint_rhs,
            )
        elif self.solver_type == "box_constrained_nlp":
            self.solver = JaxSolver(self.cfg, self.objective_func, self.bounds)
        else:
            raise NotImplementedError(
                f"Solver type '{self.solver_type}' is not implemented."
            )

    # ------------------------------------------------------------------
    # Solve interface
    # ------------------------------------------------------------------

    def initial_guess(self) -> Any:
        assert self.solver is not None, "Call construct_solver() first."
        return self.solver.initial_guess()

    def solve(self, initial_guesses: Any) -> SolveResult:
        assert self.solver is not None, "Call construct_solver() first."
        return self.solver.solve(initial_guesses)

    def __call__(self, initial_guesses: Any) -> SolveResult:
        return self.solve(initial_guesses)
