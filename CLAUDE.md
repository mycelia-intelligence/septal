# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

```
src/septal/
├── __init__.py                     Top-level re-exports
├── casadax/
│   ├── schema.py                   NLPProblem, SolveResult
│   ├── callbacks.py                JaxCasADiEvaluator, casadify_*, JaxCallbackForward/Reverse
│   ├── solvers.py                  BaseSolver, CasadiSolver (IPOPT), JaxSolver (L-BFGS-B)
│   ├── factory.py                  SolverFactory
│   └── utilities.py                generate_initial_guess (Sobol, seed=42)
└── jax/sqp/
    ├── schema.py                   ParametricNLPProblem, SQPConfig, SQPState, SQPResult
    ├── solver.py                   sqp_solve_single, sqp_solve_scan, make_solver, state_to_result
    ├── qp_subproblem.py            form_qp_matrices, ruiz_equilibration, admm_qp, polish_qp,
    │                               solve_qp_subproblem
    ├── hessian.py                  bfgs_update, lagrangian_hessian, regularised_lagrangian_hessian,
    │                               lagrangian_grad
    ├── line_search.py              l1_merit, constraint_violation, merit_directional_deriv,
    │                               update_penalty, backtracking_line_search  ("merit" mode)
    ├── filter.py                   filter_acceptable, augment_filter, switching_condition,
    │                               min_step_size, filter_line_search, restoration_direction,
    │                               restoration_line_search  ("filter" mode, default)
    ├── convergence.py              kkt_residuals, is_converged
    └── factory.py                  ParametricSQPFactory
tests/
├── conftest.py, test_*.py          CasADi backend tests
└── sqp/
    ├── conftest.py
    └── test_*.py                   JAX SQP component and integration tests
```

## Commands

```bash
# Install
pip install -e .

# Run tests
pytest

# Run a single test file
pytest tests/sqp/test_solver.py -v

# Run benchmark (nlp-bsuite repo, not here)
# cd /path/to/nlp-bsuite && python scripts/benchmark_sqp.py
```

## SQP algorithm walkthrough

### Outer loop (`solver.py`)

`make_sqp_step(problem, cfg)` returns a scan-compatible body function. One iteration:

1. Compute `f, ∇f, g, J_g` at current `x`
2. Call `solve_qp_subproblem(...)` → `(d, lam_new)`
3. Update penalty ρ via `update_penalty` (always computed; only used by `"merit"` mode)
4. Step acceptance — static branch on `cfg.step_acceptance`:
   - `"filter"` (default): `filter_line_search` → `(α, accepted, f_type)`; if not accepted
     and `n_g > 0`, `lax.cond` into feasibility restoration (`restoration_direction` +
     `restoration_line_search`); augment the filter after an h-type step or restoration
   - `"merit"`: directional derivative `D_φ`, non-monotone Armijo backtracking → `α`
5. `x_new = clip(x + α·d, lb, ub)`
6. Evaluate `f_new, g_new, merit_new`
7. Compute `∇L` at `x_new`
8. Hessian update (BFGS or `regularised_lagrangian_hessian`)
9. KKT residuals → `stationarity, feasibility`
10. `converged = is_converged(stat, feas, cfg)`
11. Update `merit_window` (rolling buffer for non-monotone LS)
12. Stagnation detection: if `α < stagnation_alpha_tol` for `stagnation_patience` steps, reset H, ρ
    and clear the filter
13. Freeze state if `converged` (no-op masked update for scan)

`sqp_solve_scan` uses `lax.scan` (always `max_iter` steps, vmappable).
`sqp_solve_single` uses `lax.while_loop` (early exit, NOT vmappable).

### Filter line search (`filter.py`)

Wächter & Biegler, *Line Search Filter Methods for Nonlinear Programming: Motivation and
Global Convergence*, SIAM J. Optim. 16(1), 2005, Algorithm I. θ = L1 general-constraint
violation (`constraint_violation`); the box never enters θ because iterates are clipped.

Trial `x(α) = clip(x + α d)` is accepted when it is acceptable to the filter (not dominated
by any stored `(θ_j, f_j)`, and `θ < θ_max`) **and**:
- switching condition `α·(−∇fᵀd)^{s_f} > δ·θ_k^{s_θ}` holds and Armijo on f holds → **f-type**
  (filter not augmented), or
- switching fails and `θ(α) ≤ (1−γ_θ)θ_k` or `f(α) ≤ f_k − γ_f θ_k` → **h-type** (filter
  augmented with `((1−γ_θ)θ_k, f_k − γ_f θ_k)`).

Backtracking stops below `α_min` (eq. 2.16) → **restoration**: Gauss–Newton QP on the
violated rows (`H_r = JᵀWJ + ζI`, violated rows' bounds relaxed to ±inf, satisfied rows and
box kept) solved with the same `solve_qp_subproblem` pipeline, then backtracking on θ with
filter acceptability. The filter is augmented with the current iterate on entry.
`state.n_restoration` counts entries. Unconstrained problems reduce to plain Armijo on f.

Filter storage is fixed-size (`filter_size`, default 20): empty slots are `+inf`, dominated
entries are cleared on insert, and a full filter evicts the largest-θ entry. `theta_max`
(`1e4·max(1, θ(x₀))`) is a separate state scalar.

Costs: under `jax.vmap` the restoration `lax.cond` lowers to a select, so the restoration QP
runs for every batch element on every iteration. `sqp_solve_single` only pays when it fires.

### QP subproblem (`qp_subproblem.py`)

```
min  ½ dᵀ Q d + cᵀ d
s.t. l ≤ A d ≤ u
```

where:
```
Q = H_k + ε·I   (symmetrised + regularised)
c = ∇f_k
A = [J_g; I_n]
l = [lhs-g_k; lb-x_k],  u = [rhs-g_k; ub-x_k]
```

Pipeline: `form_qp_matrices` → `ruiz_equilibration` → `admm_qp` (lax.scan) → `polish_qp` → unscale.

### ADMM (`admm_qp`)

Standard OSQP ADMM iterations (Boyd et al.):
```
x-step:  (Q + σI + ρ·AᵀA) x_{k+1} = σx_k - c + Aᵀ(ρ·z_k - y_k)
z-step:  z_{k+1} = clip(α·Ax_{k+1} + (1-α)·z_k + y_k/ρ, l, u)
y-step:  y_{k+1} = y_k + ρ·(α·Ax_{k+1} + (1-α)·z_k - z_{k+1})
```
- x-step uses `jnp.linalg.solve` (factorisation inside lax.scan — constant matrix, XLA caches it)
- `eq_scale` vector gives equality rows a `admm_rho_eq_scale×` multiplier
- Warm-start: `y_warm` initialised from previous SQP multipliers
- Adaptive ρ: checked every `admm_rho_update_interval` steps via `lax.cond`

### Polishing (`polish_qp`)

Active-set classification:
- Lower-active: `z_i - l_i < -y_i` or `l_i == u_i` (equality)
- Upper-active: `u_i - z_i < y_i` and `l_i != u_i`

Reduced KKT (fixed-size embedding):
```
[ Q+σI     Aᵀ·diag(is_act)        ] [x̂]   [-c          ]
[ diag(is_act)·A  diag((1-is_act)ε)] [ŷ] = [is_act·b_A  ]
```

Acceptance gates (all three must pass):
1. `isfinite(d_pol).all() and isfinite(y_pol).all()`
2. `pol_prim_res ≤ admm_prim_res + 1e-10`
3. `(lower_act | upper_act).any()` — prevents Newton step on unconstrained problems

Critical: use `jnp.where(is_act, b_A, 0.0)` not `is_act * b_A` — `0 * inf = NaN` in JAX.

## Critical JAX patterns

### Fixed-size arrays required for lax.scan / vmap / pmap

No dynamic shapes. All ADMM and polishing operations use `(n+m)×(n+m)` fixed embeddings.
`jnp.block` does NOT exist — use `jnp.concatenate` + reshape.

### NaN propagation on macOS/Accelerate

`jnp.where(False, NaN, x)` can propagate NaN into x on Accelerate backend.
Always sanitize before selecting:
```python
d_safe = jnp.where(jnp.isfinite(d_pol), d_pol, d_admm)
d_out  = jnp.where(accept, d_safe, d_admm)
```

### SQPState must be a JAX pytree

`SQPState` is a `NamedTuple` — all fields must be JAX arrays. Never store Python scalars or
Python lists/dicts in state. This is required for `vmap`, `scan`, and `pmap` compatibility.

### lax.scan frozen-state pattern

```python
def sqp_step(state, _):
    new_state = _do_step(state)
    frozen = jax.tree.map(
        lambda new, old: jnp.where(state.converged, old, new),
        new_state, state
    )
    return frozen, None
```

### Exact Hessian and inertia correction

`lagrangian_hessian` uses `jax.hessian` — expensive (O(n²) forward passes) but accurate.
`regularised_lagrangian_hessian` adds `τ·I` where `τ = max(0, -λ_min + delta)` to ensure PD.
Use `use_exact_hessian=True` for constrained problems with curved manifolds.

## Known issues / gotchas

- **hs023 start 1/5**: exterior-of-ellipse feasible set — Sobol start is trapped outside the feasible region. Not a solver bug; multi-start with more starts or better initialisation is the remedy.
- **`0 * inf = NaN`** in polishing RHS — fixed by using `jnp.where(is_act, b_A, 0.0)`.
- **Polishing Newton-step regression** — on unconstrained problems polishing would accept a pure Newton step `-(Q+σI)⁻¹c` diverging on non-convex. Fixed by Gate 3 (`_has_active` check).
- **`jnp.linalg.solve` inside lax.scan** — works correctly because the LHS matrix `(Q + σI + ρ·AᵀA)` is constant across ADMM iterations (only RHS changes). XLA caches the factorisation.
- **Float64** must be enabled before any JAX SQP operation. `import septal.jax.sqp` does this automatically.
- **vmap trajectories are not bitwise reproducible** — the polishing acceptance gate can flip
  between batched and single LU rounding (observed on the Wächter–Biegler 2000 counterexample:
  a 0.35 difference in `d` at one iterate with `admm_polish=True`, 4e-14 without). Both paths
  converge; do not assert iteration or restoration counts across vmap/non-vmap.
- **hs063 start 5 (filter mode)**: iterate lands on the box face `x1 = x3 = 0` where the
  linearised constraints are inconsistent with the bounds; the QP is infeasible and Gauss–Newton
  restoration has a stationary point of θ on that face. Converges after 252 iterations with 10
  restorations (merit: 39). Known limitation of one-shot restoration without an interior push.

## Benchmark config (reference)

From `nlp-bsuite/scripts/benchmark_sqp.py`:

```python
SQPConfig(
    max_iter=1000,
    admm_n_iter=1000,
    tol_stationarity=1e-7,
    tol_feasibility=1e-7,
    use_exact_hessian=True,
    admm_ruiz_iter=10,
    admm_polish=True,
    penalty_init=10.0,
    admm_rho_eq_scale=100.0,
    nonmonotone_window=5,
    stagnation_patience=30,
    stagnation_reset_hessian=True,
    stagnation_reset_penalty=True,
)
```

Result: 139/140 on 28 problems × 5 Sobol starts (seed=42) in **both** `step_acceptance="filter"`
(default) and `"merit"`; the single failure is hs023 (see above). Filter is faster on hs039
(34 vs 80 iters), ackley (7 vs 22), hs071 (9 vs 15); slower on hs063 start 5.

Running the benchmark from septal's venv (nlp-bsuite is not installed in it): add
`~/Github/nlp-bsuite/scripts` and `~/Github/nlp-bsuite/src` to `sys.path`, import
`benchmark_sqp`, set `benchmark_sqp.SQP_CFG = dataclasses.replace(..., step_acceptance=...)`,
then `write_report(run_all(), path)`.
