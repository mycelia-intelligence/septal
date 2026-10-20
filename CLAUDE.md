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
    │                               min_step_size, judge_trial, filter_line_search,
    │                               soc_correction, restoration_direction,
    │                               restoration_line_search  ("filter" mode, default)
    ├── convergence.py              kkt_residuals, complementarity_residual, is_converged
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

## Releasing

Version lives in one place: `src/septal/__init__.py::__version__` (pyproject reads it via
`[tool.setuptools.dynamic]`; `docs/conf.py` greps it). To release:

1. Bump `__version__`, add a `CHANGELOG.md` entry, open a PR, merge.
2. `git tag vX.Y.Z && git push origin vX.Y.Z` on `main`.
3. `.github/workflows/release.yml` builds sdist + wheel, runs `twine check`, verifies the
   tag matches `__version__`, then publishes through PyPI trusted publishing (GitHub
   environment `pypi`, no API token). Local dry run: `python -m build && twine check dist/*`.

CI runs `black --check .` with black 22.x; keep the tree black-clean.

## SQP algorithm walkthrough

### Outer loop (`solver.py`)

`make_sqp_step(problem, cfg)` returns a scan-compatible body function. One iteration:

1. Compute `f, ∇f, g, J_g` at current `x`
2. Call `solve_qp_subproblem(...)` → `(d, lam_new)`
3. Update penalty ρ via `update_penalty` (always computed; only used by `"merit"` mode)
4. Step acceptance — static branch on `cfg.step_acceptance`:
   - `"filter"` (default): `filter_line_search` → `(α, accepted, f_type)`; if the first
     trial was rejected without reducing θ, `lax.cond` into `soc_correction` (an accepted
     correction replaces the step and multipliers); if still nothing accepted, or the
     restoration phase is active, `lax.cond` into one restoration iteration
     (`restoration_direction` + `restoration_line_search`, multipliers kept); augment the
     filter after an h-type step or on restoration entry
   - `"merit"`: directional derivative `D_φ`, non-monotone Armijo backtracking → `α`
5. `x_new = clip(x + α·d, lb, ub)`
6. Evaluate `f_new, g_new, merit_new`
7. Compute `∇L` at `x_new`
8. Hessian update (BFGS or `regularised_lagrangian_hessian`)
9. KKT residuals → `stationarity, feasibility`
10. `converged = is_converged(stat, feas, cfg)`; restoration-phase bookkeeping (exit when
    `θ_new ≤ restoration_kappa·θ_entry`, stall when `α = 0` or `restoration_max_iter` reached)
11. Update `merit_window` (rolling buffer for non-monotone LS)
12. Stagnation detection: if `α < stagnation_alpha_tol` for `stagnation_patience` steps, **or
    the restoration phase stalled** (local infeasibility), reset H, ρ and clear the filter
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

**Second-order correction** (Wächter & Biegler 2006, §2.4, step A-5.3): if the first trial
(α₀) is rejected and `θ(x+α₀d) ≥ θ_k`, re-solve the same QP with
`g_soc = g_k + [g(x_t) − g_k − J(x_t − x)]` (linearisation error at the trial), judge the
corrected full step with the **original** α₀ and ∇fᵀd, up to `soc_max_iter` (4) times while θ
shrinks by `soc_kappa` (0.99). Accepted → replaces step and multipliers, `n_soc += 1`.
Cures the Maratos effect (hs063: 252 → 35 iterations on its slow start).

Backtracking stops below `α_min` (eq. 2.16) → **feasibility restoration phase** (W&B §3.2,
IPOPT §3.3 exit rule). A mode carried in the state (`in_restoration`, `theta_resto_entry`,
`resto_count`): on entry the filter is augmented with the current iterate; each iteration
solves a Gauss–Newton QP on the violated rows (`H_r = JᵀWJ + ζI`, violated rows' bounds
relaxed to ±inf, satisfied rows and box kept, same `solve_qp_subproblem` pipeline) and
backtracks on θ requiring `θ_t ≤ (1−γ_θ)θ_k` and filter acceptability; multipliers are kept.
Exit when `θ ≤ restoration_kappa·θ_entry` (0.9). **Stall** (`α = 0`, or
`restoration_max_iter` = 20 consecutive restoration iterations) = local minimiser of the
infeasibility: `n_local_infeasible += 1`, leave the phase and reset H, ρ and the filter
immediately. `state_to_result` reports stalls in the message when not converged.
`n_restoration` counts restoration iterations. Unconstrained problems reduce to plain Armijo.

Filter storage is fixed-size (`filter_size`, default 20): empty slots are `+inf`, dominated
entries are cleared on insert, and a full filter evicts the largest-θ entry. `theta_max`
(`1e4·max(1, θ(x₀))`) is a separate state scalar.

Costs: under `jax.vmap` every `lax.cond` lowers to a select, so the SOC QPs (up to
`soc_max_iter`) and the restoration QP run for every batch element on every iteration.
`sqp_solve_single` only pays when they fire. Set `soc_max_iter=0` for cheaper batched solves.

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

- **hs023 "trapped start" (fixed)**: one Sobol start used to fail in both modes. Cause: a garbage
  QP solve left multipliers of order 1e25, the exact Lagrangian Hessian built from them was garbage,
  and the stagnation/stall reset restored H but kept the multipliers. Resets now also zero `lam`
  (and recompute `grad_lag`). Restoration additionally requires a *strict* θ decrease and a zero
  restoration step is always a stall, so a phase entered at a feasible point resets instead of
  spinning. Both modes are now 140/140.
- **`0 * inf = NaN`** in polishing RHS — fixed by using `jnp.where(is_act, b_A, 0.0)`.
- **Polishing Newton-step regression** — on unconstrained problems polishing would accept a pure Newton step `-(Q+σI)⁻¹c` diverging on non-convex. Fixed by Gate 3 (`_has_active` check).
- **`jnp.linalg.solve` inside lax.scan** — works correctly because the LHS matrix `(Q + σI + ρ·AᵀA)` is constant across ADMM iterations (only RHS changes). XLA caches the factorisation.
- **Float64** must be enabled before any JAX SQP operation. `import septal.jax.sqp` does this automatically.
- **Complementarity is part of convergence** (since e95fea8). Stationarity + feasibility alone
  accepted non-KKT points: a non-zero multiplier on an inactive constraint zeroes the projected
  Lagrangian gradient (HS013 at (0,0); HS016 at the corner (0.5,1)). Any new convergence shortcut
  must keep `complementarity_residual` in the test.
- **vmap trajectories are not bitwise reproducible** — the polishing acceptance gate can flip
  between batched and single LU rounding (observed on the Wächter–Biegler 2000 counterexample:
  a 0.35 difference in `d` at one iterate with `admm_polish=True`, 4e-14 without). Both paths
  converge; do not assert iteration or restoration counts across vmap/non-vmap.
- **hs063 start 5 (filter mode)**: iterate can land on the box face `x1 = x3 = 0`, a first- and
  second-order local minimiser of the constraint violation on the feasible cone, where the QP is
  infeasible. No descent-based restoration can escape it; the restoration-phase stall detection
  resets H and the filter immediately. With SOC the trap is avoided (35 iterations).
- **Filter permissiveness**: an h-type step may raise θ enormously (θ up to θ_max = 1e4·max(1, θ₀))
  when f drops a lot. That is per the paper; the stall reset recovers from the resulting excursions.
- **Benchmark "best start" iteration counts are misleading**: the report picks the start with the
  smallest KKT residual, which can switch between runs. Compare per-start totals from the
  "Per-Start Detail" table instead.

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

Result (nlp-bsuite @ 53fa60d, 39 problems × 5 Sobol starts, seed 42): filter **185/195**
starts, merit **186/195**; the original 28 problems are **140/140** in both modes. Σ iterations
over all starts: filter 13271, merit 13133 (dominated by the known non-converging starts below).
The report's Totals block carries these sums plus SOC/restoration/stall counts
(`scripts/benchmark_sqp.py --step-acceptance {filter,merit} --out ... --json ...`, compare two
JSON sidecars with `scripts/compare_reports.py`). Starts are Sobol points in each problem's
`effective_start_bounds` (infinite bounds → ±10 window around the HS standard start).

Known non-converging starts (both modes unless stated):
- hs013: 0/5 — no constraint qualification at x*, multipliers unbounded; expected.
- hs081: 2/5 — the other starts approach the global optimum (f ≈ 0.0541 vs f* = 0.0539498)
  but stationarity stalls near 1e-2 for 1000 iterations; hs080 (same constraints, no
  `−½(x1³+x2³+1)²` term) is 5/5. Suspect the inertia correction of the negative-curvature term;
  not investigated.
- hs020: 4/5. hs033: 4/5 in filter mode only (one start cycles through 994 restoration stalls).
- hs016: all starts converge, but to local KKT points (f = 404 vertex, f = 3.98) as often as to
  f* = 0.25; the benchmark's |f−f*| column is not a convergence criterion.

nlp-bsuite is installed editable into septal's `.venv` (`pip install -e ~/Github/nlp-bsuite
--no-deps`); run the script from `~/Github/nlp-bsuite` with `~/Github/septal/.venv/bin/python`.
