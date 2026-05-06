"""Tests for septal.jax.sqp.hessian."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from septal.jax.sqp.hessian import bfgs_update, lagrangian_grad


class TestBFGSUpdate:
    def test_output_shape(self):
        n = 4
        H = jnp.eye(n)
        s = jax.random.normal(jax.random.PRNGKey(0), (n,))
        y = jax.random.normal(jax.random.PRNGKey(1), (n,)) + 0.5 * s
        H_new = bfgs_update(H, s, y)
        assert H_new.shape == (n, n)

    def test_symmetry_preserved(self):
        n = 4
        H = jnp.eye(n)
        s = jnp.array([0.1, 0.2, -0.1, 0.05])
        y = jnp.array([0.3, 0.5, 0.1, 0.2])
        H_new = bfgs_update(H, s, y)
        assert jnp.allclose(H_new, H_new.T, atol=1e-10)

    def test_positive_definite_after_update(self):
        """PD initial H, positive curvature → PD result."""
        n = 3
        H = 2.0 * jnp.eye(n)
        # s, y with positive curvature sᵀy > 0
        s = jnp.array([0.1, 0.2, 0.05])
        y = jnp.array([0.3, 0.6, 0.15])  # y = 3s → sᵀy = 3‖s‖² > 0
        H_new = bfgs_update(H, s, y)
        eigvals = jnp.linalg.eigvalsh(H_new)
        assert float(jnp.min(eigvals)) > -1e-8

    def test_damping_negative_curvature(self):
        """Damping must ensure PD even when sᵀy < 0.2 sᵀHs."""
        n = 2
        H = jnp.eye(n)
        s = jnp.array([1.0, 0.0])
        # Negative curvature: sᵀy = -0.5
        y = jnp.array([-0.5, 0.0])
        H_new = bfgs_update(H, s, y)
        eigvals = jnp.linalg.eigvalsh(H_new)
        assert float(jnp.min(eigvals)) > -1e-8

    def test_skip_on_tiny_step(self):
        """Update should be skipped (H unchanged) when ‖s‖ is tiny."""
        n = 3
        H = 5.0 * jnp.eye(n)
        s = jnp.zeros(n)  # zero step
        y = jnp.ones(n)
        H_new = bfgs_update(H, s, y, skip_tol=1e-10)
        assert jnp.allclose(H_new, H, atol=1e-12)

    def test_jit_compatible(self):
        n = 3
        H = jnp.eye(n)
        s = jnp.array([0.1, -0.1, 0.05])
        y = jnp.array([0.3, -0.2, 0.1])

        H_new = jax.jit(bfgs_update)(H, s, y)
        assert H_new.shape == (n, n)

    def test_vmap_compatible(self):
        """bfgs_update must be vmappable over batches."""
        n, batch = 3, 5
        key = jax.random.PRNGKey(42)
        H_batch = jnp.stack([jnp.eye(n)] * batch)
        s_batch = jax.random.normal(key, (batch, n)) * 0.1
        y_batch = s_batch * 2.0 + jax.random.normal(key, (batch, n)) * 0.01

        H_new_batch = jax.vmap(bfgs_update)(H_batch, s_batch, y_batch)
        assert H_new_batch.shape == (batch, n, n)


class TestLagrangianGrad:
    def test_no_constraints(self):
        """Without constraints, grad_lag = grad_f."""

        def f(x, p):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 2.0])
        p = jnp.zeros(1)
        gl = lagrangian_grad(x, jnp.zeros(0), p, f, None, 0)
        expected = jnp.array([2.0, 4.0])
        assert jnp.allclose(gl, expected, atol=1e-6)

    def test_with_constraint(self):
        """grad_lag = grad_f + J_gᵀ λ."""

        def f(x, p):
            return jnp.sum(x**2)

        def g(x, p):
            return (x[0] + x[1]).reshape(1)

        x = jnp.array([1.0, 1.0])
        p = jnp.zeros(1)
        lam = jnp.array([2.0])
        gl = lagrangian_grad(x, lam, p, f, g, 1)
        # grad_f = [2, 2]; J_g = [[1, 1]]; J_gᵀ lam = [2, 2]
        expected = jnp.array([4.0, 4.0])
        assert jnp.allclose(gl, expected, atol=1e-6)
