Installation
============

From PyPI
---------

::

    pip install septal

Development install
-------------------

Clone the repository and install in editable mode::

    git clone https://github.com/mycelia-intelligence/septal.git
    cd septal
    pip install -e .

This installs all runtime dependencies listed in ``pyproject.toml``.

To also install documentation build tools::

    pip install -e ".[docs]"

Or manually::

    pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

Building the docs
-----------------

From the ``docs/`` directory::

    make html

Then open ``docs/_build/html/index.html`` in a browser.

Runtime dependencies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - ``jax`` / ``jaxlib``
     - ≥ 0.4.30
     - Array operations, autodiff, JIT compilation, ``vmap``/``pmap``
   * - ``numpy``
     - ≥ 1.24
     - Array construction utilities
   * - ``scipy``
     - ≥ 1.11
     - Sobol sequence generation (``scipy.stats.qmc``)
   * - ``casadi``
     - ≥ 3.6
     - IPOPT NLP formulation (``casadax`` backend only)
   * - ``jaxopt``
     - ≥ 0.8
     - L-BFGS-B solver (``JaxSolver`` in ``casadax`` backend)

.. note::

   Importing ``septal.jax.sqp`` automatically enables JAX float64 precision via
   ``jax.config.update("jax_enable_x64", True)``.  This is required for
   numerical stability of BFGS and the KKT system.  It should be imported
   before any other JAX code in your programme.
