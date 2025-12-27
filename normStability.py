"""
ACID TEST: Normalization vs. Analytic Hydrogen.
Target: Validate Phase/Amplitude extraction to precision standards.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mpmath

# Add src to path
sys.path.append(os.getcwd())

from src.continuum import solve_continuum
from src.normalization import energy_normalize_continuum

# ==============================================================================
# 1. ANALYTIC REFERENCE (Z=1 Hydrogen)
# ==============================================================================
def get_analytic_hydrogen_val(r_pt, E, ell):
    """
    Returns exact energy-normalized Coulomb wave at r_pt.
    Psi ~ sqrt(2/pi*k) * sin(kr - l*pi/2 + sigma + ...)
    """
    k = np.sqrt(2*E)
    eta = -1.0 / k # Z=1
    
    # mpmath for precision
    mpmath.mp.dps = 15
    
    # Regular Coulomb F (unnormalized)
    # F -> sin(...) asymptotically
    rho = k * r_pt
    F = float(mpmath.coulombf(ell, eta, rho))
    
    # Apply Energy Normalization Factor
    # F has amplitude 1. We want sqrt(2/pi*k)
    norm_factor = np.sqrt(2.0 / (np.pi * k))
    
    return F * norm_factor

# ==============================================================================
# 2. THE STRESS TEST
# ==============================================================================
def run_stress_test():
    print("="*60)
    print(" NORMALIZATION STRESS TEST (HYDROGEN Z=1)")
    print("="*60)
    
    # Energy Grid: Low k (danger zone) to High k
    # 0.01 a.u. -> k ~ 0.14 (Very long wavelength)
    energies = np.geomspace(0.01, 2.0, 30)
    
    amp_errors = []
    phase_metrics = [] # We'll use L2 norm deviation as proxy for phase/amp combined
    
    print(f"{'Energy':<10} | {'Max Rel Error':<15} | {'Status':<10}")
    print("-" * 45)
    
    for E in energies:
        # 1. Numerical Solution (Your Code)
        # Using a large box to ensure we settle into asymptotic region
        r_num, u_num, _ = solve_continuum(
            E_pe=E, ell=1, species='H', 
            A=0.0, U=0.0, mu=0.0, # Pure Coulomb
            R_max=300.0, N=6000   # High resolution
        )
        
        # 2. Compare to Analytic at the tail (Last 10%)
        # This checks if the normalization routine scaled it correctly
        mask_idx = np.where(r_num > 250.0)[0]
        r_tail = r_num[mask_idx]
        u_tail_num = u_num[mask_idx]
        
        u_tail_analytic = np.array([
            get_analytic_hydrogen_val(r, E, ell=1) for r in r_tail
        ])
        
        # 3. Calculate Errors
        # If phase is wrong, difference will oscillate
        # If amp is wrong, difference will be scaled
        diff = np.abs(u_tail_num - u_tail_analytic)
        max_amplitude = np.max(np.abs(u_tail_analytic))
        
        # Maximum Relative Deviation
        rel_error = np.max(diff) / max_amplitude
        
        amp_errors.append(rel_error)
        
        status = "PASS" if rel_error < 1e-3 else "WARN"
        if rel_error > 0.05: status = "FAIL"
        
        print(f"{E:<10.4f} | {rel_error:<15.2e} | {status}")

    # ==========================================================================
    # 3. VISUALIZATION
    # ==========================================================================
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.plot(energies, np.array(amp_errors)*100, 'o-', color='crimson', lw=2)
    ax1.set_xscale('log')
    ax1.set_xlabel("Photoelectron Energy (a.u.)")
    ax1.set_ylabel("Max Wavefunction Deviation (%)")
    ax1.set_title("Normalization Accuracy vs Analytic Hydrogen")
    ax1.grid(True, which="both", alpha=0.3)
    
    # Mark the Danger Zone (Low k)
    ax1.axvspan(0.01, 0.1, color='yellow', alpha=0.1, label='Low-k Danger Zone')
    
    
    
    # Threshold line
    ax1.axhline(1.0, color='k', linestyle='--', label='1% Tolerance')
    
    plt.legend()
    plt.savefig("normalization_stress_test.png")
    print(f"\nPlot saved to normalization_stress_test.png")
    plt.show()

if __name__ == "__main__":
    run_stress_test()