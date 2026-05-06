"""
General-purpose utilities for septal.

These are backend-agnostic helpers used by both solver implementations.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc


def generate_initial_guess(n_starts: int, n_d: int, bounds: list[Any]) -> Any:
    """Generate Sobol-sequence initial guesses within *bounds*.

    Parameters
    ----------
    n_starts:
        Number of initial points to generate.
    n_d:
        Dimension of the decision variable.
    bounds:
        ``[lb, ub]`` — each array-like of shape ``(n_d,)``.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_starts, n_d)``.
    """
    n_d = len(bounds[0])
    lb = jnp.array(bounds[0])
    ub = jnp.array(bounds[1])
    sobol = qmc.Sobol(d=n_d, scramble=True, seed=42).random(n_starts)
    return lb + (ub - lb) * sobol


def unpack_results(
    solutions: list, fallback_solver: Any, fallback_solution: Any
) -> tuple:
    """Pick the best solution from a multi-start result list.

    Parameters
    ----------
    solutions:
        List of ``(solver, solution)`` pairs from successful IPOPT runs.
    fallback_solver:
        Solver object to use when *solutions* is empty.
    fallback_solution:
        Solution object to use when *solutions* is empty.

    Returns
    -------
    (solver_stats, solution, n_successful)
    """
    try:
        min_idx = np.argmin(np.vstack([s[1]["f"] for s in solutions]))
        solver_opt, solution_opt = solutions[min_idx]
        return solver_opt.stats(), solution_opt, len(solutions)
    except Exception:
        return fallback_solver.stats(), fallback_solution, len(solutions)


def clean_up(objects: list) -> None:
    """Delete a list of objects to free memory."""
    for obj in objects:
        del obj
