# Roadmap

_Last updated: March 2026_

## Current Status

The JAX-native SQP solver has reached high maturity:

- **139/140** converged on 28 Hock-Schittkowski problems × 5 Sobol starts (99.3%)
- **3.8× median speedup** over IPOPT on a 34-problem head-to-head benchmark
- Fully JIT-compilable and vmappable for batched parametric solves

### Implemented

| Component | Details |
|---|---|
| ADMM QP subproblem | Adaptive ρ scaling, warm-started multipliers |
| Ruiz diagonal equilibration | 10-iter scaling; removes ADMM stall from scale imbalance |
| OSQP active-set polishing | 3 acceptance gates; vmap-safe fixed-size KKT embed |
| Exact Lagrangian Hessian | `jax.hessian`; damped BFGS fallback |
| Hessian inertia correction | τI shift ensuring min eigenvalue ≥ δ |
| Non-monotone line search | Rolling merit window (M=5); standard Armijo as special case (`step_acceptance="merit"`) |
| Filter line search (default) | Wächter–Biegler 2005 Alg. I: f-/h-type steps, fixed-size filter; second-order correction (IPOPT §2.4); iterative Gauss–Newton restoration phase with local-infeasibility stall reset |
| Stagnation detection + reset | Resets Hessian and penalty after 30 stuck iterations |
| Batched solves | `sqp_solve_scan` via `lax.scan` — constant iteration count for vmap |

---

## Gaps

| Gap | Impact |
|---|---|
| Dense constraint Jacobian | Limits practical scalability to n ≈ 50–100 |
| Dense exact Hessian (O(n²)) | No path to large-n problems |
| No warm-starting | Every parametric re-solve starts cold |
| No sensitivity analysis | Cannot differentiate through the solve |
| No large-scale benchmarks | Validation only at n ≤ 10 |

---

## 6-Month Roadmap (Apr–Sep 2026)

### Month 1 — Apr: Sparse Jacobian Feasibility

Survey and prototype approaches for auto-detecting and exploiting sparsity in the constraint
Jacobian A, using JAX tracing rather than user-specified patterns.

- Survey `jax.experimental.sparse` (BCOO/BCSR) for JIT/vmap compatibility
- Evaluate `sparsify` transform and coloring / seed-vector approaches
- Prototype: evaluate J at a sample point → threshold → extract (row, col) index set
- Assess sparse KKT solve options for the polishing step
- **Deliverable**: `docs/sparse_design.md`; working prototype on 1–2 structured problems

---

### Month 2 — May: Sparse ADMM

Replace dense matrix operations in the QP subproblem with sparse equivalents when a
sparsity pattern is available.

- `detect_jacobian_sparsity(g, x0, p0) → SparsityPattern` in `sparse_utils.py`
- Store pattern in `ParametricNLPProblem` at construction time
- Sparse matrix-vector products in ADMM (`A @ x` → BCOO equivalent)
- Ruiz equilibration updated for sparse A
- Gated by `SQPConfig.use_sparse_jacobian: bool = False`
- **Deliverable**: Sparse ADMM passing full 28-problem benchmark with no regressions

---

### Month 3 — Jun: Sparse Polishing + Structured Benchmarks

Extend polishing to sparse A; validate gains on structured large-n problems.

- Sparse KKT system in OSQP polishing step
- Add scaled H&S variants (n=50, 100, 200) to nlp-bsuite
- Add MPC-like banded problem (linear tracking, horizon N=10–50)
- **Deliverable**: `reports/sparse_benchmark_report.md`; dense vs sparse scaling plot

---

### Month 4 — Jul: Sparse / L-BFGS Hessian

Scale the Hessian approximation to match sparse Jacobian gains.

- Sparse exact Hessian: detect Lagrangian Hessian sparsity; compute only non-zeros
- L-BFGS: ring buffer (history m=10), two-loop recursion, O(mn) memory
- `SQPConfig.hessian_type: Literal["bfgs", "lbfgs", "exact"]`
- Usage guidance: n < 50 → exact; n = 50–500 → L-BFGS; structured → sparse exact
- **Deliverable**: Scaling study; L-BFGS converges on n=500 problem

---

### Month 5 — Aug: Warm-Starting + API

Enable fast repeated parametric solves; clean up API for collaborator use.

- `sqp_solve_single(..., warm_state: SQPState | None = None)` — initialise from prior solve
- Warm-start demo: parametric sweep varying p, showing reduced iteration counts
- API consistency pass on `factory.py`; improved error messages
- Update README and CLAUDE.md with sparse + L-BFGS usage
- **Deliverable**: Warm-start speedup report on `parametric_tracking`

---

### Month 6 — Sep: Sensitivity Analysis + Broader Validation

Differentiate through the solve; expand problem coverage; prepare for collaborator beta.

- `∂x*/∂p` via implicit differentiation of KKT conditions (`jax.custom_vjp` wrapper)
- Verify against finite-difference reference on parametric problems
- Expand nlp-bsuite to 50+ problems (remaining Hock-Schittkowski suite)
- Final comparison report: JAX SQP (sparse) vs IPOPT at large n
- **Deliverable**: Sensitivity verification notebook; updated benchmark report

---

## Summary

| Month | Theme | Key Deliverable |
|---|---|---|
| Apr | Sparse Jacobian feasibility | `docs/sparse_design.md` |
| May | Sparse ADMM | Full benchmark with no regression |
| Jun | Sparse polishing + benchmarks | `reports/sparse_benchmark_report.md` |
| Jul | Sparse/L-BFGS Hessian | Scaling study to n=500 |
| Aug | Warm-starting + API | Warm-start speedup report |
| Sep | Sensitivity + validation | Sensitivity notebook; 50+ problems |
