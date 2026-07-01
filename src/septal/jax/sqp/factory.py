"""
High-level factory for parametric and batched SQP solves.

Usage — single solve
--------------------
    factory = ParametricSQPFactory(problem, cfg)
    result  = factory.solve(x0, p)

Usage — batched solve (GPU-accelerated)
----------------------------------------
    results = factory.solve_batch(x0_batch, params_batch)
    # results.decision_variables has shape (N, n)
    # results.success            has shape (N,)

Usage — pre-compiled batch solve
---------------------------------
    solve = factory.compile_batch(N)   # traces + compiles once
    results = solve(x0_batch, params_batch)
    results = solve(x0_batch2, params_batch2)  # fast: no recompilation
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

from septal.casadax.schema import NLPProblem
from septal.casadax.utilities import generate_initial_guess
from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig, SQPResult
from septal.jax.sqp.solver import (
    batch_state_to_result,
    make_solver,
    sqp_solve_single,
    state_to_result,
)


class ParametricSQPFactory:
    """Factory for single and batched parametric SQP solves.

    Parameters
    ----------
    problem:
        Parametric NLP problem definition.
    cfg:
        SQP solver configuration.  Defaults to :class:`~SQPConfig` if
        ``None``.
    """

    def __init__(
        self,
        problem: ParametricNLPProblem,
        cfg: SQPConfig | None = None,
    ) -> None:
        self.problem = problem
        self.cfg = cfg if cfg is not None else SQPConfig()
        # Build and JIT-compile the scan-based solve function once
        self._solve_jit: Callable = make_solver(problem, self.cfg)

    # ------------------------------------------------------------------
    # Single solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: jnp.ndarray,
        p: jnp.ndarray,
        use_while_loop: bool = False,
    ) -> SQPResult:
        """Solve a single parametric NLP instance.

        Parameters
        ----------
        x0:
            Initial guess, shape ``(n,)``.
        p:
            Parameter vector, shape ``(m,)``.
        use_while_loop:
            If ``True`` use the while-loop variant (early exit); otherwise
            use the scan-based variant (fixed iterations, vmappable).

        Returns
        -------
        SQPResult
        """
        t0 = time.perf_counter()
        if use_while_loop:
            state = sqp_solve_single(self.problem, x0, p, self.cfg)
        else:
            state = self._solve_jit(x0, p)
        timing = time.perf_counter() - t0
        return state_to_result(state, self.problem, timing=timing)

    def initial_guess(self) -> jnp.ndarray:
        """Generate a single Sobol-sequence initial guess within bounds.

        Returns
        -------
        jnp.ndarray, shape ``(n,)``
        """
        n = self.problem.n_decision
        guess = generate_initial_guess(1, n, self.problem.bounds)
        return guess.reshape(n)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Batched solve
    # ------------------------------------------------------------------

    def solve_batch(
        self,
        x0_batch: jnp.ndarray,
        params_batch: jnp.ndarray,
    ) -> SQPResult:
        """Solve N parametric NLPs simultaneously via ``jax.vmap``.

        Parameters
        ----------
        x0_batch:
            Initial guesses, shape ``(N, n)``.  Pass ``(n,)`` to broadcast
            the same guess to all N problems.
        params_batch:
            Parameter vectors, shape ``(N, m)``.

        Returns
        -------
        SQPResult
            All array fields have a leading batch dimension ``N``.
        """
        x0_batch = jnp.asarray(x0_batch)
        params_batch = jnp.asarray(params_batch)

        # Broadcast scalar initial guess if needed
        if x0_batch.ndim == 1:
            N = params_batch.shape[0]
            x0_batch = jnp.broadcast_to(x0_batch, (N, self.problem.n_decision))

        t0 = time.perf_counter()
        batched_fn = jax.vmap(self._solve_jit)
        state_batch = batched_fn(x0_batch, params_batch)
        # Force evaluation (JAX is lazy)
        jax.block_until_ready(state_batch)
        timing = time.perf_counter() - t0

        return batch_state_to_result(state_batch, self.problem, timing=timing)

    def compile_batch(self, N: int) -> Callable:
        """AOT-trace and compile the batched solver for batch size ``N``.

        Returns a callable that accepts ``(x0_batch, params_batch)`` and
        returns an :class:`~SQPResult` without re-compilation on subsequent
        calls.

        Parameters
        ----------
        N:
            Batch size to compile for.

        Returns
        -------
        Callable
            ``(x0_batch: (N, n), params_batch: (N, m)) -> SQPResult``
        """
        n = self.problem.n_decision
        m = self.problem.n_params

        batched_fn = jax.vmap(self._solve_jit)
        compiled_fn = jax.jit(batched_fn)

        # Warm-up: trace and compile with dummy arrays
        dummy_x0 = jnp.zeros((N, n))
        dummy_p = jnp.zeros((N, m)) if m > 0 else jnp.zeros((N, 0))
        compiled_fn(dummy_x0, dummy_p)  # triggers compilation

        def _run(x0_batch: jnp.ndarray, params_batch: jnp.ndarray) -> SQPResult:
            t0 = time.perf_counter()
            state_batch = compiled_fn(x0_batch, params_batch)
            jax.block_until_ready(state_batch)
            timing = time.perf_counter() - t0
            return batch_state_to_result(state_batch, self.problem, timing=timing)

        return _run

    # ------------------------------------------------------------------
    # Class method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_nlp_problem(
        cls,
        problem: NLPProblem,
        n_params: int,
        cfg: SQPConfig | None = None,
    ) -> ParametricSQPFactory:
        """Construct from an existing :class:`~septal.casadax.schema.NLPProblem`.

        The NLP's objective and constraint functions must accept a second
        argument ``p`` (the parameter vector), even if they ignore it.  This
        is the standard way to convert an existing parametric JAX function.

        Parameters
        ----------
        problem:
            Non-parametric NLP.  Its ``objective`` and ``constraints``
            callables must accept ``(x, p)`` signatures.
        n_params:
            Dimension of the parameter vector.
        cfg:
            SQP configuration.

        Returns
        -------
        ParametricSQPFactory
        """
        n_decision = problem.n_decision
        assert n_decision is not None, (
            "NLPProblem.n_decision must be set before conversion"
        )
        param_problem = ParametricNLPProblem(
            objective=problem.objective,
            bounds=problem.bounds,
            n_decision=n_decision,
            n_params=n_params,
            constraints=problem.constraints,
            constraint_lhs=problem.constraint_lhs,
            constraint_rhs=problem.constraint_rhs,
        )
        return cls(param_problem, cfg)
