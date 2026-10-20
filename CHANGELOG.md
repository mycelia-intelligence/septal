# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-20

First release on PyPI.

### Added
- Wächter–Biegler filter line search (`SQPConfig.step_acceptance="filter"`, now the
  default) with second-order correction and an iterative feasibility restoration phase
  that detects local infeasibility (`septal.jax.sqp.filter`).
- Complementarity residual in the KKT convergence test
  (`SQPConfig.tol_complementarity`, `SQPResult.kkt_complementarity`).
- Solver counters `n_soc`, `n_restoration` and `n_local_infeasible` in `SQPState`.
- `septal.__version__`, `py.typed` marker, MIT `LICENSE` file.

### Changed
- Every SQP iterate is projected onto the variable box.
- Stagnation and restoration-stall resets also zero the multipliers.
- Packaging moved to a single PEP 621 `[project]` table; the duplicate Poetry
  metadata and lock file were removed. Minimum `jax`/`jaxlib` is now 0.4.30.

### Fixed
- Stationarity and feasibility alone accepted non-KKT points with a non-zero
  multiplier on an inactive constraint (HS013, HS016).
- A garbage QP solve could leave multipliers of order 1e25 that poisoned the exact
  Hessian after a reset (HS023 trapped start).

## [0.1.1] - 2026-05

### Added
- Jenkins and GitHub Actions CI.

## [0.1.0] - 2026-04

### Added
- JAX SQP solver (`septal.jax.sqp`): ADMM QP subproblem with Ruiz equilibration and
  active-set polishing, damped BFGS or exact Lagrangian Hessian with inertia
  correction, L1 merit non-monotone line search, `vmap`/`pmap` batched solves.
- CasADi/IPOPT backend (`septal.casadax`) with JAX forward/reverse AD callbacks.

[0.2.0]: https://github.com/mycelia-intelligence/septal/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/mycelia-intelligence/septal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mycelia-intelligence/septal/releases/tag/v0.1.0
