# Triaxial Jeans

A Python package for solving the Jeans equations for triaxial galaxies with separable potentials, following the methodology from ["General solution of the Jeans equations for triaxial galaxies with separable potentials"](https://arxiv.org/pdf/astro-ph/0302172).

## Overview

This package provides tools for:
- Computing velocity moments and dispersions in triaxial potentials
- Solving the Jeans equations for triaxial galaxy models
- Implementing separable potential models
- Analyzing orbital dynamics in triaxial systems
- Fitting models to observational data

## Citation

If you use this package in your research, please cite:

> de Zeeuw, T., Bureau, M., & Franx, M. (2002). "General solution of the Jeans equations for triaxial galaxies with separable potentials." *The Astrophysical Journal*, 343(1), 3-21.

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
# Example: Stäckel potential or other separable models

# Compute velocity moments
# ...

# Solve Jeans equations
# ...
```

## Features

- **Separable Potentials**: Support for common separable potential models
- **Jeans Equations**: Numerical solutions for velocity moments and dispersions
- **Triaxial Geometries**: Handle full 3D triaxial systems
- **Observational Analysis**: Tools for comparing to observational data
- **Visualization**: Plotting utilities for results and diagnostics

## Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## Examples

See the [examples](examples/) directory for Jupyter notebooks demonstrating:
- Basic potential models
- Jeans equation solutions
- Comparing different triaxial configurations
- Fitting to synthetic and real data

## Architecture

```
triaxial_jeans/
├── potentials/       # Separable potential implementations
├── jeans/            # Jeans equation solvers
├── kinematics/       # Velocity moment calculations
├── fitting/          # Model fitting utilities
└── utils/            # Helper functions and utilities
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

- Nedělchev Nikolay

## References

- de Zeeuw, T., Bureau, M., & Franx, M. (2002). General solution of the Jeans equations for triaxial galaxies with separable potentials. *The Astrophysical Journal*, 343(1), 3-21.
