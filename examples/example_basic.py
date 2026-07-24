"""
Example script: Basic usage of triaxial Jeans models

This script demonstrates:
1. Creating a separable potential
2. Initializing a Jeans solver
3. Computing velocity moments at different positions
4. Analyzing the velocity dispersion tensor
"""

import numpy as np
from triaxial_jeans.potentials.separable import PowerLawStackelPotential
from triaxial_jeans.jeans.solver import JeansAnalyticalSolver
from triaxial_jeans.kinematics.moments import VelocityMomentsCalculator


def main():
    """Run basic example."""
    
    print("=" * 60)
    print("Triaxial Jeans: Basic Example")
    print("=" * 60)
    
    # Step 1: Create a separable potential
    print("\n1. Creating a Power-Law Stäckel Potential...")
    print("   Axis ratios: q1=0.9 (b/a), q2=0.7 (c/a)")
    print("   Power-law index: alpha=1.0 (Newtonian)")
    
    potential = PowerLawStackelPotential(
        q1=0.9,
        q2=0.7,
        a=1.0,
        alpha=1.0
    )
    print("   ✓ Potential created")
    
    # Step 2: Initialize Jeans solver
    print("\n2. Initializing Jeans Solver...")
    jeans_solver = JeansAnalyticalSolver(potential)
    print("   ✓ Solver initialized (Analytical method)")
    
    # Step 3: Create velocity moments calculator
    print("\n3. Setting up velocity moments calculator...")
    moments_calc = VelocityMomentsCalculator(potential, jeans_solver)
    print("   ✓ Calculator ready")
    
    # Step 4: Compute at various positions
    print("\n4. Computing velocity moments at different radii...")
    
    # Define positions along the major axis
    radii = np.array([0.1, 0.5, 1.0, 2.0])
    x_positions = radii  # Along x-axis (major axis a)
    y_positions = np.zeros_like(x_positions)
    z_positions = np.zeros_like(x_positions)
    
    for i, (x, y, z) in enumerate(zip(x_positions, y_positions, z_positions)):
        print(f"\n   Position {i+1}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Solve Jeans equations
        results = jeans_solver.solve_jeans_equations(
            np.array([x]), np.array([y]), np.array([z])
        )
        
        sigma_x = results['sigma_x'][0]
        sigma_y = results['sigma_y'][0]
        sigma_z = results['sigma_z'][0]
        
        print(f"   Velocity dispersions:")
        print(f"     σ_x = {sigma_x:.4f}")
        print(f"     σ_y = {sigma_y:.4f}")
        print(f"     σ_z = {sigma_z:.4f}")
        
        # Compute velocity ellipsoid shape
        shape = moments_calc.velocity_ellipsoid_shape(
            np.array([x]), np.array([y]), np.array([z])
        )
        
        print(f"   Principal dispersions:")
        print(f"     σ_1 = {shape['sigma_1'][0]:.4f} (major)")
        print(f"     σ_2 = {shape['sigma_2'][0]:.4f} (intermediate)")
        print(f"     σ_3 = {shape['sigma_3'][0]:.4f} (minor)")
        print(f"   Ratios: σ₂/σ₁ = {shape['ratio_21'][0]:.3f}, σ₃/σ₁ = {shape['ratio_31'][0]:.3f}")
        
        # Compute anisotropy parameter
        anis = moments_calc.anisotropy_parameter(
            np.array([x]), np.array([y]), np.array([z])
        )
        
        print(f"   Anisotropy: β = {anis['beta'][0]:.4f}")
        if anis['beta'][0] > 0:
            print(f"     (radial anisotropy)")
        elif anis['beta'][0] < 0:
            print(f"     (tangential anisotropy)")
        else:
            print(f"     (isotropic)")
    
    # Step 5: Compare geometry
    print("\n" + "=" * 60)
    print("5. Geometry Summary:")
    print("=" * 60)
    print(f"Triaxial ellipsoid geometry:")
    print(f"  Semi-major axis (a):      1.0 (units)")
    print(f"  Semi-intermediate axis:   {potential.q1:.2f} (units)")
    print(f"  Semi-minor axis (c):      {potential.q2:.2f} (units)")
    print(f"  Flattening q1 = b/a:      {potential.q1:.2f}")
    print(f"  Flattening q2 = c/a:      {potential.q2:.2f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
