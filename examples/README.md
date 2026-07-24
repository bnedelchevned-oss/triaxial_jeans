# Triaxial Jeans Examples

This directory contains example scripts and Jupyter notebooks demonstrating how to use the triaxial Jeans package.

## Notebooks

- `01_basic_potentials.ipynb` - Introduction to separable potentials
- `02_jeans_equations.ipynb` - Solving Jeans equations in triaxial systems
- `03_velocity_dispersions.ipynb` - Computing velocity moments and dispersions
- `04_fitting_example.ipynb` - Fitting models to mock observational data

## Scripts

- `example_stackel_potential.py` - Working with Stäckel potentials
- `example_jeans_solution.py` - Solving Jeans equations

## Getting Started

To run the examples, first install the package:

```bash
pip install -e ..
```

Then run a notebook:

```bash
jupyter notebook 01_basic_potentials.ipynb
```

Or run a script:

```bash
python example_stackel_potential.py
```

## References

All examples follow the methodology from:

> de Zeeuw, T., Bureau, M., & Franx, M. (2002). "General solution of the Jeans equations for triaxial galaxies with separable potentials." *The Astrophysical Journal*, 343(1), 3-21.
