septal
======

A lightweight Python library for solving **nonlinear programmes (NLPs)** with a
pure-JAX batched SQP solver and optional CasADi/IPOPT integration.

.. code-block:: text

   min  f(x, p)
   s.t. lhs ≤ g(x, p) ≤ rhs
        lb  ≤     x   ≤ ub

**Two backends:**

.. list-table::
   :header-rows: 1
   :widths: 25 30 20 25

   * - Backend
     - Entry point
     - AD
     - Use case
   * - **JAX SQP**
     - :class:`~septal.jax.sqp.factory.ParametricSQPFactory`
     - ``jax.grad`` / ``jax.hessian``
     - Batched parametric NLP — GPU/TPU, ``vmap``-compatible
   * - **CasADi / IPOPT**
     - :class:`~septal.casadax.factory.SolverFactory`
     - ``jvp`` / ``vjp`` callbacks
     - General constrained NLP on CPU

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Background

   casadi_ipopt
   algorithm
   benchmark
   citing

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index
