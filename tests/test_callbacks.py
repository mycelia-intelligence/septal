"""Tests for septal.casadax.callbacks — JAX-CasADi AD wrappers."""

import jax.numpy as jnp
import numpy as np
import pytest

from septal.casadax.callbacks import (
    JaxCallbackForward,
    JaxCallbackReverse,
    JaxCasADiEvaluator,
    casadify,
    casadify_forward,
    casadify_reverse,
)


def _scalar_fn(x):
    """f(x) = sum(x^2), shape (1,1)."""
    return jnp.sum(x ** 2).reshape(1, 1)


def _vector_fn(x):
    """g(x) = [x[0]^2, x[1]^2, x[0]*x[1]], shape (3,1)."""
    x_ = x.reshape(-1)
    return jnp.array([x_[0] ** 2, x_[1] ** 2, x_[0] * x_[1]]).reshape(3, 1)


class TestJaxCasADiEvaluator:
    def test_output_shape_scalar(self):
        cb = JaxCasADiEvaluator(_scalar_fn, nd=2)
        assert cb.n_out_dim == 1
        assert cb.nd == 2

    def test_output_shape_vector(self):
        cb = JaxCasADiEvaluator(_vector_fn, nd=2)
        assert cb.n_out_dim == 3

    def test_eval_scalar(self):
        from casadi import DM
        cb = JaxCasADiEvaluator(_scalar_fn, nd=2)
        x_in = DM([1.0, 2.0])
        result = cb.eval([x_in])
        val = float(result[0])
        assert val == pytest.approx(5.0)  # 1^2 + 2^2 = 5

    def test_eval_vector(self):
        from casadi import DM
        cb = JaxCasADiEvaluator(_vector_fn, nd=2)
        x_in = DM([2.0, 3.0])
        result = cb.eval([x_in])
        out = np.array(result[0]).flatten()
        assert out[0] == pytest.approx(4.0)   # 2^2
        assert out[1] == pytest.approx(9.0)   # 3^2
        assert out[2] == pytest.approx(6.0)   # 2*3


class TestJaxCallbackForward:
    def test_construction(self):
        cb = JaxCallbackForward(_vector_fn, nd=2)
        assert cb.n_out_dim == 3
        assert cb.has_forward(1)
        assert not cb.has_forward(0)

    def test_eval_matches_base(self):
        from casadi import DM
        cb = JaxCallbackForward(_scalar_fn, nd=2)
        x_in = DM([3.0, 4.0])
        result = cb.eval([x_in])
        val = float(result[0])
        assert val == pytest.approx(25.0)  # 9 + 16

    def test_counter_increments(self):
        from casadi import DM
        cb = JaxCallbackForward(_scalar_fn, nd=2)
        cb.eval([DM([1.0, 0.0])])
        cb.eval([DM([0.0, 1.0])])
        assert cb.counter == 2


class TestJaxCallbackReverse:
    def test_construction(self):
        cb = JaxCallbackReverse(_scalar_fn, nd=2)
        assert cb.n_out_dim == 1
        assert cb.has_reverse(1)
        assert not cb.has_reverse(0)

    def test_eval_matches_base(self):
        from casadi import DM
        cb = JaxCallbackReverse(_scalar_fn, nd=2)
        x_in = DM([1.0, 2.0])
        result = cb.eval([x_in])
        val = float(result[0])
        assert val == pytest.approx(5.0)

    def test_counter_increments(self):
        from casadi import DM
        cb = JaxCallbackReverse(_scalar_fn, nd=2)
        cb.eval([DM([0.0, 0.0])])
        assert cb.counter == 1


class TestCasadifyFactories:
    def test_casadify_forward_returns_forward_type(self):
        cb = casadify_forward(_vector_fn, nd=2)
        assert isinstance(cb, JaxCallbackForward)

    def test_casadify_reverse_returns_reverse_type(self):
        cb = casadify_reverse(_scalar_fn, nd=2)
        assert isinstance(cb, JaxCallbackReverse)

    def test_casadify_auto_forward_when_n_out_gt_n_d(self):
        # n_out=3 > n_d=2 → should pick forward
        cb = casadify(_vector_fn, nd=2, n_out=3)
        assert isinstance(cb, JaxCallbackForward)

    def test_casadify_auto_reverse_when_n_out_le_n_d(self):
        # n_out=1 <= n_d=2 → should pick reverse
        cb = casadify(_scalar_fn, nd=2, n_out=1)
        assert isinstance(cb, JaxCallbackReverse)

    def test_casadify_reverse_correct_value(self):
        from casadi import DM
        cb = casadify_reverse(_scalar_fn, nd=3)
        x_in = DM([1.0, 1.0, 1.0])
        result = cb.eval([x_in])
        assert float(result[0]) == pytest.approx(3.0)

    def test_casadify_forward_correct_value(self):
        from casadi import DM
        cb = casadify_forward(_vector_fn, nd=2)
        x_in = DM([2.0, 3.0])
        result = cb.eval([x_in])
        out = np.array(result[0]).flatten()
        assert out[0] == pytest.approx(4.0)
