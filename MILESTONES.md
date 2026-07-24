# Development Milestones

This document outlines the development roadmap for the Triaxial Jeans package, following the methodology from van de Ven et al. (2003).

## Overview

The package will be developed in stages, progressively building from basic geometric foundations to advanced solvers and extensions. Each milestone is designed to be testable and incrementally adds capabilities.

---

## Milestone 1: Geometry and Coordinate Transforms (~500 lines)
**Status**: 🟢 IN PROGRESS

### Objectives
- Implement confocal ellipsoidal coordinate system
- Cartesian ↔ ellipsoidal transformations
- Jacobian determinant calculations
- Metric tensor components

### Key Components
- `triaxial_jeans/utils/coordinates.py` - Core transformations
- `triaxial_jeans/utils/geometry.py` - Geometric utilities
- Unit tests for coordinate systems

### Deliverables
- [x] Confocal parameter calculations
- [x] Coordinate transformation functions
- [x] Jacobian determinant
- [ ] Complete unit test suite
- [ ] Numerical validation against analytical checks

### Lines of Code Target: ~500
- **Current**: ~350 lines
- **Remaining**: ~150 lines

---

## Milestone 2: Stäckel Potentials and Density (~800 lines)
**Status**: 🟡 IN PROGRESS

### Objectives
- Implement separable Stäckel potentials
- Power-law potential models (de Zeeuw 1985)
- Homogeneous triaxial ellipsoid potentials
- Density profile computations from potentials

### Key Components
- `triaxial_jeans/potentials/separable.py` - Separable potential classes
- `triaxial_jeans/potentials/stackel.py` - Stäckel potential framework
- Density computation via Poisson equation
- Acceleration calculations

### Deliverables
- [x] `SeparablePotential` base class
- [x] `PowerLawStackelPotential` class
- [x] `HomogeneousTriaxialPotential` class
- [ ] Density profile calculations
- [ ] Numerical validation of potentials
- [ ] Benchmark against analytical solutions

### Lines of Code Target: ~800
- **Current**: ~550 lines
- **Remaining**: ~250 lines

---

## Milestone 3: Differential Operators and Jeans Equations (~1,200 lines)
**Status**: 🟠 TODO

### Objectives
- Implement gradient, Laplacian, divergence in ellipsoidal coordinates
- Formulate Jeans equations in separable coordinates
- Virial theorem implementation
- Line-of-sight kinematics

### Key Components
- `triaxial_jeans/jeans/operators.py` - Differential operators
- `triaxial_jeans/jeans/solver.py` - Jeans equation solvers
- `triaxial_jeans/kinematics/moments.py` - Velocity moment computations
- Analytical solutions framework

### Deliverables
- [ ] Gradient operator in ellipsoidal coords
- [ ] Laplacian operator in ellipsoidal coords
- [ ] Divergence operator
- [ ] `JeansAnalyticalSolver` for separable potentials
- [ ] Virial relations implementation
- [ ] Line-of-sight velocity dispersion
- [ ] Numerical tests and validation

### Key Equations
```
Continuity:     ∇·(ρ v) = 0
Jeans equations: ∇·(ρ σ_ij) = -ρ ∇_i Φ
Virial relation: ∫ ρ σ²_ij dV ∝ ∫ ρ |∇Φ| dV
```

### Lines of Code Target: ~1,200
- **Current**: ~400 lines
- **Remaining**: ~800 lines

---

## Milestone 4: 2D Singular-Solution Solver (~2,000 lines)
**Status**: 🔴 TODO

### Objectives
- Implement 2D meridional plane solver
- Singular-solution method (van de Ven et al. 2003)
- Velocity moment recovery
- Regularization and numerical stability

### Key Components
- `triaxial_jeans/jeans/singular_solver.py` - Core 2D solver
- `triaxial_jeans/jeans/regularization.py` - Regularization techniques
- `triaxial_jeans/utils/meshgrid.py` - 2D grid utilities

### Deliverables
- [ ] 2D mesh generation in meridional plane
- [ ] Singular-solution ansatz implementation
- [ ] Boundary condition handling
- [ ] Velocity moment recovery
- [ ] Numerical stability analysis
- [ ] Convergence tests

### Algorithm
1. Discretize the meridional plane (x-z plane)
2. Set up singular solutions at specific points
3. Enforce Jeans equations and boundary conditions
4. Solve resulting linear system
5. Recover 3D velocity moments

### Lines of Code Target: ~2,000
- **Current**: 0 lines
- **Remaining**: ~2,000 lines

---

## Milestone 5: Full 3D Solver (~3,500-5,000 lines)
**Status**: 🔴 TODO

### Objectives
- Extend 2D solver to full 3D
- Handle all three coordinates (λ, μ, ν)
- Iterative solvers for complex potentials
- Performance optimization

### Key Components
- `triaxial_jeans/jeans/solver_3d.py` - 3D solver engine
- `triaxial_jeans/jeans/iterative.py` - Iterative methods
- `triaxial_jeans/utils/parallel.py` - Parallel utilities

### Deliverables
- [ ] 3D mesh generation in ellipsoidal coordinates
- [ ] Full 3D Jeans equation formulation
- [ ] Iterative solver (Gauss-Seidel, multigrid)
- [ ] Adaptive refinement
- [ ] Performance benchmarks
- [ ] GPU acceleration hooks

### Algorithm Features
- Separability exploitation for efficiency
- Adaptive mesh refinement near singularities
- Multi-level iteration schemes
- Parallel computation over grid points

### Lines of Code Target: ~3,500-5,000
- **Current**: 0 lines
- **Remaining**: ~3,500-5,000 lines

---

## Milestone 6: Paper Reproduction (~1,000 lines)
**Status**: 🔴 TODO

### Objectives
- Reproduce all numerical examples from van de Ven et al. (2003)
- Generate all figures from the paper
- Validate against published results
- Create comprehensive test suite

### Key Components
- `tests/test_paper_examples.py` - Unit tests
- `examples/paper_figures.py` - Figure reproduction scripts
- `examples/validation_suite.py` - Comprehensive validation

### Deliverables
- [ ] Example 1: Isotropic sphere (Section 3.1)
- [ ] Example 2: Flattened ellipsoid (Section 3.2)
- [ ] Example 3: Triaxial ellipsoid (Section 3.3)
- [ ] All 20+ figures from the paper
- [ ] Numerical agreement within specified tolerances
- [ ] Documentation of any deviations

### Validation Criteria
- Results agree with paper to within 1-5% (depending on quantity)
- Figures are visually identical to published versions
- All test cases pass automatically

### Lines of Code Target: ~1,000
- **Current**: 0 lines
- **Remaining**: ~1,000 lines

---

## Milestone 7: Modern Extensions (~2,000+ lines)
**Status**: 🔴 TODO (Post-release)

### Objectives
- GPU acceleration with CUDA/Metal/WebGPU
- Bayesian inference for parameter fitting
- JAX backend for automatic differentiation
- Advanced visualization and interactive tools

### Sub-projects

#### 7a: GPU Acceleration (~800 lines)
- CUDA kernels for hot-spot computations
- CuPy/PyTorch backend option
- Performance benchmarks
- Scaling analysis

#### 7b: Bayesian Fitting (~700 lines)
- PyMC3/Numpyro integration
- MCMC samplers for triaxial parameters
- Posterior visualization
- Model comparison tools

#### 7c: JAX Backend (~500 lines)
- Pure JAX implementation for autodiff
- vmap for batch computations
- jit compilation for speed
- Automatic gradient computation

### Additional Features
- Interactive Jupyter widgets
- 3D visualization with plotly/mayavi
- Web-based model explorer
- Publication-ready plotting

### Lines of Code Target: ~2,000+
- **Current**: 0 lines
- **Remaining**: ~2,000+ lines

---

## Development Statistics

### Code Organization
```
Total Target Lines: ~14,000-16,000
├── Core Library:      ~10,000 lines (Milestones 1-5)
├── Paper Validation:  ~1,000 lines (Milestone 6)
├── Tests & Docs:      ~2,000-3,000 lines
└── Extensions:        ~2,000+ lines (Milestone 7)
```

### Timeline Estimates
- **Milestone 1**: 1-2 weeks ✓
- **Milestone 2**: 1-2 weeks ✓
- **Milestone 3**: 2-3 weeks 🟡
- **Milestone 4**: 3-4 weeks 🔴
- **Milestone 5**: 4-6 weeks 🔴
- **Milestone 6**: 2-3 weeks 🔴
- **Milestone 7**: 4-6 weeks+ 🔴

### Cumulative Progress
- **Phase 1** (Milestones 1-2): ~23% complete ✓
- **Phase 2** (Milestones 3-4): Planning 🟡
- **Phase 3** (Milestones 5-6): Planning 🔴
- **Phase 4** (Milestone 7): Post-release 🔴

---

## Quality Assurance

### Testing Strategy
- Unit tests for each module (~100+ test cases)
- Integration tests for solver pipeline
- Numerical validation against known results
- Performance benchmarks
- Regression testing

### Documentation
- Docstrings for all public functions
- Mathematical derivations for key algorithms
- Usage examples and tutorials
- API reference documentation
- Developer guide

### Performance Targets
- Coordinate transformations: < 1 ms per 1000 points
- 2D solver convergence: < 100 iterations
- 3D solver: scales well to 10⁶ grid points
- GPU speedup: 10-100× over CPU

---

## Community Contributions

This is an open-source project welcoming contributions at all levels:

### Contribution Opportunities
- Algorithm optimization
- Performance profiling and optimization
- Additional potential models
- Observational applications
- Tutorial creation
- Documentation improvements

### How to Contribute
See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

---

## Version Roadmap

- **v0.1.0** (Milestones 1-2): Basic structure and potentials
- **v0.2.0** (Milestone 3): Jeans equations framework
- **v0.3.0** (Milestone 4): 2D solver
- **v0.4.0** (Milestone 5): 3D solver
- **v1.0.0** (Milestone 6): Paper reproduction and validation
- **v2.0.0** (Milestone 7): Advanced extensions

---

## References

- van de Ven, G., Hunter, C., Verolme, E.K., & de Zeeuw, P.T. (2003)
- de Zeeuw, T. (1985)
- Schwarzschild, M. (1979)

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/bnedelchevned-oss/triaxial_jeans).
