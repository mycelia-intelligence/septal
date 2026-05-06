"""Tests for septal.casadax.utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from septal.casadax.utilities import clean_up, generate_initial_guess, unpack_results


class TestGenerateInitialGuess:
    def test_shape(self, simple_bounds):
        samples = generate_initial_guess(n_starts=8, n_d=2, bounds=simple_bounds)
        assert samples.shape == (8, 2)

    def test_within_bounds(self, sphere_bounds):
        n = 32
        samples = generate_initial_guess(n_starts=n, n_d=3, bounds=sphere_bounds)
        assert jnp.all(samples >= sphere_bounds[0])
        assert jnp.all(samples <= sphere_bounds[1])

    def test_single_start(self, simple_bounds):
        samples = generate_initial_guess(n_starts=1, n_d=2, bounds=simple_bounds)
        assert samples.shape == (1, 2)

    def test_uses_all_of_domain(self, simple_bounds):
        """Sobol samples should spread across the domain, not collapse to one corner."""
        samples = generate_initial_guess(n_starts=16, n_d=2, bounds=simple_bounds)
        # Each dimension should have range > 0.5 (Sobol covers well)
        for dim in range(2):
            span = float(jnp.max(samples[:, dim]) - jnp.min(samples[:, dim]))
            assert span > 0.5, f"Dim {dim} has span {span:.3f} — Sobol not spreading"


class TestUnpackResults:
    def _make_fake_solver(self, f_val):
        class FakeSolver:
            def stats(self):
                return {"return_status": "ok", "success": True, "t_wall_total": 0.1}

        return FakeSolver()

    def _make_fake_solution(self, f_val):
        return {"f": np.array([[f_val]]), "x": np.zeros(2), "g": np.zeros(0)}

    def test_picks_best_objective(self):
        s1 = self._make_fake_solver(1.0)
        s2 = self._make_fake_solver(0.5)
        sol1 = self._make_fake_solution(1.0)
        sol2 = self._make_fake_solution(0.5)
        solutions = [(s1, sol1), (s2, sol2)]
        stats, best_sol, ns = unpack_results(solutions, s1, sol1)
        assert ns == 2
        assert float(best_sol["f"]) == pytest.approx(0.5)

    def test_fallback_on_empty_list(self):
        s = self._make_fake_solver(99.0)
        sol = self._make_fake_solution(99.0)
        stats, best_sol, ns = unpack_results([], s, sol)
        assert ns == 0
        assert float(best_sol["f"]) == pytest.approx(99.0)


class TestCleanUp:
    def test_clean_up_runs_without_error(self):
        a = [1, 2, 3]
        b = {"key": "val"}
        clean_up([a, b])  # should not raise
