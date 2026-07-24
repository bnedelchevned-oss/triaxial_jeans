# Documentation

## Overview

This directory contains documentation for the Triaxial Jeans package.

## Table of Contents

- `getting_started.md` - Installation and quick start guide
- `theory.md` - Mathematical background and theory
- `api_reference.md` - API reference documentation
- `examples.md` - Usage examples and tutorials

## Mathematical Background

The triaxial Jeans equations describe the dynamics of stars in triaxial gravitational potentials.

For separable potentials, the velocity moments can be computed analytically or semi-analytically, providing efficient models for galaxy kinematics.

### Key Concepts

- **Triaxial Systems**: Non-spherical mass distributions with three distinct axes
- **Separable Potentials**: Potentials that can be written as sums of one-dimensional functions in generalized coordinates (confocal ellipsoidal coordinates)
- **Jeans Equations**: Moment equations relating velocity dispersions to the potential and density
- **Confocal Coordinates**: Natural coordinate system for triaxial systems where the separability condition is satisfied

## Main Reference

The primary reference for this package is:

> van de Ven, G., Hunter, C., Verolme, E.K., & de Zeeuw, P.T. (2003). "General solution of the Jeans equations for triaxial galaxies with separable potentials." *The Astrophysical Journal*, 343(1), 3-21.
> ArXiv: [astro-ph/0302172](https://arxiv.org/abs/astro-ph/0302172)

## Additional References

- de Zeeuw, T. (1985). "Elliptical galaxies with power-law potentials." *Monthly Notices of the Royal Astronomical Society*, 216(2), 273-393.
- Schwarzschild, M. (1979). "A numerical model for a triaxial stellar system in dynamic equilibrium." *The Astrophysical Journal*, 232, 236-247.
- de Zeeuw, T., & Franx, M. (1991). "Structure and dynamics of elliptical galaxies." *Annual Review of Astronomy and Astrophysics*, 29, 239-274.

## Algorithm Overview

The package implements the general solution of the Jeans equations following van de Ven et al. (2003):

### Step 1: Define Separable Potential
Choose a separable potential in confocal ellipsoidal coordinates:
```
Phi(lambda, mu, nu) = Phi_lambda(lambda) + Phi_mu(mu) + Phi_nu(nu)
```

### Step 2: Specify Density Distribution
Select a density profile compatible with the potential.

### Step 3: Solve Jeans Equations
Compute velocity moments analytically or numerically from:
```
d/dr_i (rho * sigma_ij^2) + rho * dPhi/dr_i = 0
```

### Step 4: Extract Observables
Compute line-of-sight kinematics and other observables for comparison with data.

## Key Mathematical Concepts

### Confocal Ellipsoidal Coordinates

For a triaxial system with semi-axes a > b > c and axis ratios:
- q1 = b/a (intermediate to major)
- q2 = c/a (minor to major)

The confocal coordinates (λ, μ, ν) satisfy:
- x²/(λ + a²) + y²/(λ + b²) + z²/λ = 1

### Separability Condition

The Laplacian operator separates in confocal coordinates, allowing the Jeans equations to be solved analytically for each coordinate independently.

### Velocity Dispersion Tensor

For each point in the galaxy, the velocity dispersion tensor is:
```
<v_i v_j> = integral over velocity distribution
```

The eigenvalues give the principal velocity dispersions along the triaxial axes.
