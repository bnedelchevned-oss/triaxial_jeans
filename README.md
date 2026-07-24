# Triaxial Jeans

A Python package for solving the Jeans equations for triaxial galaxies with separable potentials, following the methodology from ["General solution of the Jeans equations for triaxial galaxies with separable potentials"](https://arxiv.org/abs/astro-ph/0302172).

## Overview

This package provides tools for:
- Computing velocity moments and dispersions in triaxial potentials
- Solving the Jeans equations for triaxial galaxy models
- Implementing separable potential models
- Analyzing orbital dynamics in triaxial systems
- Fitting models to observational data

## Citation

If you use this package in your research, please cite:

> van de Ven, G., Hunter, C., Verolme, E.K., & de Zeeuw, P.T. "General solution of the Jeans equations for triaxial galaxies with separable potentials." *The Astrophysical Journal*, 343(1), 3-21.

and this repository:
```bibtex
@software{triaxial_jeans,
  author = {Nedělchev, Nikolay},
  title = {Triaxial Jeans: Solutions for Triaxial Galaxies with Separable Potentials},
  url = {https://github.com/bnedelchevned-oss/triaxial_jeans},
  year = {2026}
}
```

## Installation

To install the package, clone the repository and install the dependencies:

```bash
git clone https://github.com/bnedelchevned-oss/triaxial_jeans.git
cd triaxial_jeans
pip install -e .
```

## Quick Start

```python
import triaxial_jeans as tj
import numpy as np

# Define a triaxial potential with separable form
potential = tj.potentials.PowerLawStackelPotential(q1=0.9, q2=0.7, a=1.0)

# Initialize Jeans solver
jeans_solver = tj.jeans.JeansAnalyticalSolver(potential)

# Compute velocity moments at specific positions
x, y, z = np.array([0.5]), np.array([0.3]), np.array([0.2])
results = jeans_solver.solve_jeans_equations(x, y, z)

print(f"Velocity dispersions:")
print(f"  sigma_x = {results['sigma_x']}")
print(f"  sigma_y = {results['sigma_y']}")
print(f"  sigma_z = {results['sigma_z']}")
```

## Features

- **Separable Potentials**: Support for common separable potential models (de Zeeuw 1985)
- **Jeans Equations**: Analytical and numerical solutions for velocity moments and dispersions
- **Triaxial Geometries**: Handle full 3D triaxial systems with arbitrary axis ratios
- **Coordinate Systems**: Confocal ellipsoidal coordinate transformations
- **Observational Analysis**: Tools for comparing to observational data
- **Model Fitting**: Least-squares fitting with bootstrap error estimation

## Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## Examples

See the [examples](examples/) directory for Python scripts demonstrating:
- Basic potential models
- Jeans equation solutions
- Comparing different triaxial configurations
- Fitting to synthetic and real data

## Architecture

```
triaxial_jeans/
├── potentials/       # Separable potential implementations
│   ├── separable.py  # Power-law and homogeneous potentials
│   └── stackel.py    # Stäckel potential framework
├── jeans/            # Jeans equation solvers
│   ├── __init__.py   # Jeans solver base classes
│   └── solver.py     # Analytical and numerical solvers
├── kinematics/       # Velocity moment calculations
│   └── moments.py    # Velocity moment and dispersion tensors
├── fitting/          # Model fitting utilities
│   └── fitter.py     # Least-squares fitting and bootstrap analysis
└── utils/            # Helper functions
    └── coordinates.py # Coordinate transformations
```

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib (for visualization)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Implementation: Nedělchev Nikolay
- Theory: van de Ven, G., Hunter, C., Verolme, E.K., & de Zeeuw, P.T.

## References

- van de Ven, G., Hunter, C., Verolme, E.K., & de Zeeuw, P.T. (2003). "General solution of the Jeans equations for triaxial galaxies with separable potentials." *The Astrophysical Journal*, 343(1), 3-21. [arXiv:astro-ph/0302172](https://arxiv.org/abs/astro-ph/0302172)
- de Zeeuw, T. (1985). "Elliptical galaxies with power-law potentials." *Monthly Notices of the Royal Astronomical Society*, 216(2), 273-393.
- Schwarzschild, M. (1979). "A numerical model for a triaxial stellar system in dynamic equilibrium." *The Astrophysical Journal*, 232, 236-247.
