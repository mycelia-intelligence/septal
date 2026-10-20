"""
JAX-CasADi callback wrappers.

Wraps arbitrary JAX functions so they can be used inside CasADi NLP problems
while still benefiting from JAX's JIT compilation and automatic differentiation.

Two AD modes are provided:
- Forward mode  (JaxCallbackForward / casadify_forward)  — preferred when
  n_constraints > n_decision (tall Jacobian).
- Reverse mode  (JaxCallbackReverse / casadify_reverse) — preferred when
  n_constraints <= n_decision (wide Jacobian; e.g. scalar objective).

The choice of mode is made automatically by :func:`casadify` based on the
relative dimensions of the function.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from casadi import DM, Callback, Function, Sparsity
from jax import jit, jvp, vjp


class JaxCasADiEvaluator(Callback):
    """Base CasADi Callback that wraps a JIT-compiled JAX function.

    Performs a single dummy forward pass on construction to determine the
    output shape so CasADi can set up its sparsity patterns.

    Parameters
    ----------
    functn:
        JAX callable ``f(x)`` where ``x`` has shape ``(nd, 1)``.
    nd:
        Number of input dimensions.
    name:
        CasADi function name (must be unique within a session).
    opts:
        CasADi callback options dict.
    """

    def __init__(
        self,
        functn: Callable,
        nd: int,
        name: str = "JaxCasADiEvaluator",
        opts: dict | None = None,
    ) -> None:
        if opts is None:
            opts = {}
        super().__init__()
        self.nd = nd
        self.refs: list = []
        self._forward_pass = jit(functn)

        # Determine output shape via a dummy trace
        dummy_x = jnp.zeros((nd, 1))
        dummy_y = self._forward_pass(dummy_x)
        self.output_shape = dummy_y.shape
        self.n_out_dim = self.output_shape[0]

        self.construct(name, opts)

    # CasADi interface ----------------------------------------------------------

    def get_n_in(self) -> int:
        return 1

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_in(self, i: int) -> Sparsity:
        assert i == 0
        return Sparsity.dense(self.nd, 1)

    def get_sparsity_out(self, i: int) -> Sparsity:
        assert i == 0
        return Sparsity.dense(self.n_out_dim, 1)

    def eval(self, arg):  # type: ignore[no-untyped-def]
        x_numpy = np.array(arg[0]).reshape(self.nd, 1)
        x_jax = jnp.array(x_numpy)
        y_jax = jnp.reshape(self._forward_pass(x_jax), (self.n_out_dim, 1))
        return [DM(np.asarray(y_jax))]

    def has_reverse(self, nadj: int) -> bool:
        return nadj == 0

    def has_forward(self, nfwd: int) -> bool:
        return nfwd == 0


class JaxCallbackForward(JaxCasADiEvaluator):
    """JAX-CasADi callback using *forward-mode* AD (jvp).

    Preferred when the number of constraints exceeds the number of decision
    variables (tall Jacobian).
    """

    def __init__(
        self,
        functn: Callable,
        nd: int,
        name: str = "JaxCallbackForward",
        opts: dict | None = None,
    ) -> None:
        super().__init__(functn, nd, name, opts)
        self.counter = 0
        self.time = 0.0
        self.forward_pass = jit(functn)

        @jit
        def _fwd(primals, tangents):  # type: ignore[no-untyped-def]
            return jvp(self.forward_pass, (primals,), (tangents,))[1]

        self.forward_sensitivities = _fwd

    def eval(self, arg):  # type: ignore[no-untyped-def]
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret

    def has_forward(self, nfwd: int) -> bool:
        return nfwd == 1

    def get_forward(self, nfwd, name, inames, onames, opts):  # type: ignore[no-untyped-def]
        assert nfwd == 1
        forward_sens = self.forward_sensitivities
        nd = self.nd
        n_out_dim = self.n_out_dim

        class ForwardEvaluator(Callback):
            def __init__(self, forward_fn, nd, n_out_dim, name, opts):  # type: ignore[no-untyped-def]
                super().__init__()
                self._fwd = forward_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self) -> int:
                return 3  # x, y (unused), fwd_seed

            def get_n_out(self) -> int:
                return 1  # fwd_y

            def get_sparsity_in(self, i):  # type: ignore[no-untyped-def]
                if i == 0:
                    return Sparsity.dense(self.nd, 1)
                if i == 1:
                    return Sparsity.dense(self.n_out_dim, 1)
                if i == 2:
                    return Sparsity.dense(self.nd, 1)

            def get_sparsity_out(self, i):  # type: ignore[no-untyped-def]
                return Sparsity.dense(self.n_out_dim, 1)

            def eval(self, arg):  # type: ignore[no-untyped-def]
                x = jnp.array(arg[0]).reshape(self.nd)
                tangents = jnp.array(arg[2]).reshape(self.nd)
                sens = self._fwd(x, tangents)
                return [DM(np.array(sens))]

        fwd_cb = ForwardEvaluator(forward_sens, nd, n_out_dim, f"{name}_forward", opts)
        self.refs.append(fwd_cb)

        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        fwd_seed = self.mx_in()

        return Function(
            name,
            nominal_in + nominal_out + fwd_seed,
            fwd_cb.call(nominal_in + nominal_out + fwd_seed),
            inames,
            onames,
        )


class JaxCallbackReverse(JaxCasADiEvaluator):
    """JAX-CasADi callback using *reverse-mode* AD (vjp).

    Preferred when the number of constraints does not exceed the number of
    decision variables (wide Jacobian; e.g. scalar objective).
    """

    def __init__(
        self,
        functn: Callable,
        nd: int,
        name: str = "JaxCallbackReverse",
        opts: dict | None = None,
    ) -> None:
        super().__init__(functn, nd, name, opts)
        self.counter = 0
        self.time = 0.0
        self.forward_pass = jit(functn)

        @jit
        def _vjp(primals, tangents):  # type: ignore[no-untyped-def]
            return vjp(self._forward_pass, primals)[1](tangents)[0]

        self.reverse_pass = _vjp

    def eval(self, arg):  # type: ignore[no-untyped-def]
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret

    def has_reverse(self, nadj: int) -> bool:
        return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):  # type: ignore[no-untyped-def]
        assert nadj == 1
        reverse_pass = self.reverse_pass
        nd = self.nd
        n_out_dim = self.n_out_dim

        class ReverseEvaluator(Callback):
            def __init__(self, rev_fn, nd, n_out_dim, name, opts):  # type: ignore[no-untyped-def]
                super().__init__()
                self._rev = rev_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self) -> int:
                return 3  # x, y (unused), adj_seed

            def get_n_out(self) -> int:
                return 1  # grad_x

            def get_sparsity_in(self, i):  # type: ignore[no-untyped-def]
                if i == 0:
                    return Sparsity.dense(self.nd, 1)
                if i == 1:
                    return Sparsity.dense(self.n_out_dim, 1)
                if i == 2:
                    return Sparsity.dense(self.n_out_dim, 1)

            def get_sparsity_out(self, i):  # type: ignore[no-untyped-def]
                return Sparsity.dense(self.nd, 1)

            def eval(self, arg):  # type: ignore[no-untyped-def]
                x = jnp.array(arg[0]).reshape(self.nd, 1)
                adj_seed = jnp.array(arg[2]).reshape(self.n_out_dim, 1)
                grad = self._rev(x, adj_seed)
                return [DM(np.array(grad))]

        rev_cb = ReverseEvaluator(reverse_pass, nd, n_out_dim, f"{name}_reverse", opts)
        self.refs.append(rev_cb)

        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        rev_seed = self.mx_out()

        return Function(
            name,
            nominal_in + nominal_out + rev_seed,
            rev_cb.call(nominal_in + nominal_out + rev_seed),
            inames,
            onames,
        )


# Factory helpers ----------------------------------------------------------------


def casadify_forward(functn: Callable, nd: int) -> JaxCallbackForward:
    """Wrap a JAX function with forward-mode AD for use in CasADi."""
    return JaxCallbackForward(functn, nd)


def casadify_reverse(functn: Callable, nd: int) -> JaxCallbackReverse:
    """Wrap a JAX function with reverse-mode AD for use in CasADi."""
    return JaxCallbackReverse(functn, nd)


def casadify(functn: Callable, nd: int, n_out: int) -> JaxCasADiEvaluator:
    """Auto-select forward or reverse mode based on Jacobian shape.

    Parameters
    ----------
    functn:
        JAX function ``f(x)`` where ``x`` has shape ``(nd, 1)``.
    nd:
        Number of input dimensions.
    n_out:
        Number of output dimensions (rows of the Jacobian).

    Returns
    -------
    JaxCallbackForward or JaxCallbackReverse
    """
    if n_out > nd:
        return casadify_forward(functn, nd)
    return casadify_reverse(functn, nd)
