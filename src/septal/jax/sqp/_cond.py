"""
septal.jax.sqp._cond
====================
A ``lax.cond`` that does not copy constants into every ``vmap`` lane.

``jax.lax.cond`` with a batched predicate is lowered by ``jax.vmap`` to a
select, and its batching rule first broadcasts *every* operand of the cond to
the batch size.  Constants the branches close over are operands too, so a
large array captured by a user's objective or constraints (a Cholesky factor,
a data matrix) gets one private copy per lane, and every evaluation inside the
branches becomes a batched-matrix operation.

:func:`batch_aware_cond` keeps ``lax.cond`` when it is not vmapped, so a
single solve still skips the branch it does not take.  Under ``jax.vmap`` with
a batched predicate it evaluates both branches with ``jax.vmap`` over the
batched operands only and selects the outputs.  Both branches already ran for
every lane in that case; the difference is that unbatched data stays
unbatched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


def _hoist_closed_over_tracers(
    fn: Callable[..., Any], operands: tuple[jnp.ndarray, ...]
) -> tuple[Callable[..., Any], list[Any]]:
    """Closure-convert *fn*: return ``(converted, tracers)`` such that
    ``converted(*operands, *tracers) == fn(*operands)``.

    *tracers* are the JAX tracers *fn* closes over (any dtype) — the only
    closed-over values that can be batched by an enclosing ``vmap``.  Concrete
    arrays stay baked into *converted* as constants.  ``jax.closure_convert``
    is not used because it hoists only values that may carry a tangent, which
    misses integer tracers and values batched by a non-jitted ``vmap``.
    """
    closed, out_shape = jax.make_jaxpr(fn, return_shape=True)(*operands)
    out_tree = jax.tree.structure(out_shape)
    consts = list(closed.consts)
    is_tracer = [isinstance(c, jax.core.Tracer) for c in consts]
    tracers = [c for c, t in zip(consts, is_tracer, strict=True) if t]

    def converted(*args: jnp.ndarray) -> Any:
        ops, hoisted = args[: len(operands)], iter(args[len(operands) :])
        all_consts = [
            next(hoisted) if t else c for c, t in zip(consts, is_tracer, strict=True)
        ]
        out_flat = jax.core.eval_jaxpr(closed.jaxpr, all_consts, *ops)
        return jax.tree.unflatten(out_tree, out_flat)

    return converted, tracers


def batch_aware_cond(
    pred: jnp.ndarray,
    true_fn: Callable[..., Any],
    false_fn: Callable[..., Any],
    *operands: jnp.ndarray,
) -> Any:
    """``lax.cond(pred, true_fn, false_fn, *operands)`` with a vmap rule that
    leaves unbatched data unbatched.

    Parameters
    ----------
    pred : scalar bool array
    true_fn, false_fn : callables taking ``*operands`` and returning pytrees of
        arrays with identical structure, shapes and dtypes.
    *operands : arrays passed to the branches.  Tracers the branches close
        over are found and treated as further operands, so a closed-over value
        that an enclosing ``vmap`` batches is handled; concrete arrays the
        branches close over are constants and are never batched.

    Returns
    -------
    The output of the selected branch.
    """
    operands = tuple(jnp.asarray(op) for op in operands)
    n_ops = len(operands)
    true_conv, true_aux = _hoist_closed_over_tracers(true_fn, operands)
    false_conv, false_aux = _hoist_closed_over_tracers(false_fn, operands)
    n_true = len(true_aux)

    def _true(*args: jnp.ndarray) -> Any:
        return true_conv(*args[:n_ops], *args[n_ops : n_ops + n_true])

    def _false(*args: jnp.ndarray) -> Any:
        return false_conv(*args[:n_ops], *args[n_ops + n_true :])

    @custom_vmap
    def _cond(pred: jnp.ndarray, *args: jnp.ndarray) -> Any:
        return jax.lax.cond(pred, _true, _false, *args)

    @_cond.def_vmap
    def _cond_vmap(
        axis_size: int, in_batched: list, pred: jnp.ndarray, *args: jnp.ndarray
    ) -> tuple[Any, Any]:
        pred_batched, args_batched = in_batched[0], list(in_batched[1:])
        any_arg_batched = any(args_batched)
        if any_arg_batched:
            in_axes = [0 if b else None for b in args_batched]
            true_b = jax.vmap(_true, in_axes=in_axes, axis_size=axis_size)
            false_b = jax.vmap(_false, in_axes=in_axes, axis_size=axis_size)
        else:
            true_b, false_b = _true, _false

        if not pred_batched:
            # Outputs carry the batch axis exactly when some operand does
            # (nothing is batched when the rule is re-entered through jvp).
            out = jax.lax.cond(pred, true_b, false_b, *args)
            return out, jax.tree.map(lambda _: any_arg_batched, out)

        out_true = true_b(*args)
        out_false = false_b(*args)
        lane_dims = 1 if any_arg_batched else 0

        def _select(t: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
            mask = pred.reshape((axis_size,) + (1,) * (jnp.ndim(t) - lane_dims))
            return jnp.where(mask, t, f)

        out = jax.tree.map(_select, out_true, out_false)
        return out, jax.tree.map(lambda _: True, out)

    return _cond(pred, *operands, *true_aux, *false_aux)
