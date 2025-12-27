"""
DIAGNOSTIC: Reproduce Saha et al. Fig 10(b) (Ar@C60 GASW).
Target: Squared Dipole Matrix Element |d_{3p->ed}|^2 vs Photoelectron Energy.

CRITICAL PHYSICS:
  1. Non-Relativistic (No SOC).
  2. Single Channel (3p -> ed only).
  3. No Prefactors (No E_phot, no alpha, no 4pi^2).
  4. Coulomb Phase Normalization (Crucial for correct Cooper Min location).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.getcwd())

from src import potential
from src import bound
from src import continuum
from src.potential import solve_gasw_parameters

# ==============================================================================
# 1. SETUP PARAMETERS (SAHA FIG 10)
# ==============================================================================
SPECIES = 'Ar'
L_INIT = 1       # 3p state
L_FINAL = 2      # d-wave continuum (Dominant channel)

# Energy Grid: Focused on Cooper Minimum (0.5 - 3.0 a.u.)
# Saha Fig 10 covers approx 0.0 to 3.5 a.u.
E_GRID = np.linspace(0.1, 3.5, 200)

# Saha Depths to reproduce
DEPTHS = [0.0, 0.30, 0.46, 0.56] # 0.0 is Free
COLORS = ['black', 'red', 'green', 'blue']
LABELS = ['Free', 'GASW 0.30', 'GASW 0.46', 'GASW 0.56']

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================
def calculate_dipole_sq(depth_au, energy_grid):
    """
    Calculates raw |<u_d | r | u_3p>|^2 without prefactors.
    """
    # 1. Configure Potential
    if depth_au == 0.0:
        A, U = 0.0, 0.0
    else:
        # Use Saha's Ratio Constraint
        A, U = solve_gasw_parameters(depth_au)
    
    # 2. Solve Bound State (3p) - Non-Relativistic
    # Note: target_n_idx=1 for Argon 3p (assuming 2p is n=0 in this pseudo-potential, 
    # or checking nodes. Usually for Ar Tong-Lin, 3p is the ground state returned).
    r_b, u_b, E_b, _, _ = bound.solve_ground_u(
        potential.VGASW_total_debye, 
        species=SPECIES, 
        A=A, U=U, mu=0.0,
        l_wave=L_INIT,       # l=1
        xi_soc=0.0,          # NR
        r_c=6.7, Delta=2.8   # Saha Constants
    )
    
    dipoles_sq = []
    
    for E_pe in energy_grid:
        # 3. Solve Continuum (epsilon-d) - Non-Relativistic
        # Using l_final = 2
        r_c, u_c, _ = continuum.compute_continuum_state(
            E_pe, 
            ell_cont=L_FINAL, 
            species=SPECIES, 
            A=A, U=U, mu=0.0,
            xi_soc=0.0,
            r_c=6.7, Delta=2.8
        )
        
        # 4. Calculate Dipole Integral (Raw)
        # Check grids match
        if len(r_c) != len(u_c): 
            dipoles_sq.append(0.0)
            continue
            
        # Interpolate bound to continuum grid
        u_b_interp = np.interp(r_c, r_b, u_b)
        
        # D = <u_c | r | u_b>
        integrand = u_c * r_c * u_b_interp
        D = np.trapz(integrand, r_c)
        
        # Store |D|^2
        dipoles_sq.append(D**2)
        
    return np.array(dipoles_sq)

# ==============================================================================
# 3. MAIN PLOT ROUTINE
# ==============================================================================
def main():
    print("="*60)
    print(" REPRODUCING SAHA FIG 10(b): |d_{3p->ed}|^2")
    print("="*60)
    
    plt.figure(figsize=(8, 6), dpi=150)
    
    for depth, col, lbl in zip(DEPTHS, COLORS, LABELS):
        print(f"Processing {lbl}...")
        
        d_sq = calculate_dipole_sq(depth, E_GRID)
        
        # Saha plots this on a log scale or linear? 
        # Fig 10(b) looks linear but values are small.
        # Let's plot linear first as per standard matrix element plots.
        plt.plot(E_GRID, d_sq, color=col, linewidth=2, label=lbl)

        # Print Cooper Min location for verification
        min_idx = np.argmin(d_sq)
        print(f"  -> Cooper Minimum at E = {E_GRID[min_idx]:.3f} a.u.")

    

    plt.xlabel("Photoelectron Energy (a.u.)", fontsize=12, fontweight='bold')
    plt.ylabel(r"$|D_{3p \to \epsilon d}|^2$ (a.u.)", fontsize=12, fontweight='bold')
    plt.title("Reproduction of Saha Fig 10(b) (Argon Dipole Squared)", fontsize=14)
    
    plt.xlim(0.0, 3.5)
    # y-limit: Saha's peak is around 0.05 - 0.10 roughly? 
    # Let matplotlib auto-scale first.
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "Saha_Fig10b_Reproduction.png"
    plt.savefig(out_file)
    print(f"\nPlot saved to {out_file}")
    print("Compare this explicitly with the paper figure.")
    plt.show()

if __name__ == "__main__":
    main()