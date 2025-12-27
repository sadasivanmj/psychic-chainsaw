"""
relativistic_continuum.py

Continuum state solver using Numba-optimized Numerov method.
UPDATED: Compatible with the new normalization.py (Phase-Shift fixed).
"""
import numpy as np
from numba import njit

# --- 1. Import Physics Modules ---
from src.potential import VGASW_total_debye, SAE_PARAMS

# --- 2. Import Normalization (UPDATED) ---
# We now only need the master routine.
from src.normalization import energy_normalize_continuum

# ==============================================================================
# JIT KERNEL (Numerov Integration)
# ==============================================================================
@njit(fastmath=True, cache=True)
def numerov_loop(u, k_squared, h2_12, N):
    """
    Standard Numerov integration loop.
    Solves u''(r) + k^2(r) u(r) = 0
    """
    for n in range(1, N - 1):
        k2_nm1 = k_squared[n - 1]
        k2_n = k_squared[n]
        k2_np1 = k_squared[n + 1]
        
        numerator = (2.0 * (1.0 - 5.0 * h2_12 * k2_n) * u[n] 
                     - (1.0 + h2_12 * k2_nm1) * u[n - 1])
        denominator = 1.0 + h2_12 * k2_np1
        
        u[n + 1] = numerator / denominator

# ==============================================================================
# CORE SOLVER
# ==============================================================================
def solve_continuum(E_pe, ell, species, A, U, mu, 
                    j_total=None, xi_soc=0.0, 
                    R_max=200.0, N=5000, 
                    **kwargs):
    """
    Solves the radial Schrödinger equation for a continuum electron.
    """
    # 1. Setup Grid
    # Extend R_max slightly to ensure we have room for fitting
    R_calc = R_max + 20.0 
    N_calc = int(N * (R_calc / R_max))
    
    r = np.linspace(1e-4, R_calc, N_calc)
    dr = r[1] - r[0]
    h2_12 = (dr**2) / 12.0
    
    # 2. Calculate Potential V(r)
    # Using the L-S coupled potential from potential.py
    # We pass explicit r_c/Delta if they are in kwargs (for Confined cases)
    V = VGASW_total_debye(r, A=A, U=U, mu=mu, species=species, 
                          l_wave=ell, j_total=j_total, xi_soc=xi_soc, 
                          **kwargs)
    
    # 3. Effective Kinetic Energy Term: k^2(r) = 2(E - V_eff)
    # V_eff includes Centrifugal term
    centrifugal = ell * (ell + 1) / (2.0 * r**2)
    k_squared = 2.0 * (E_pe - (V + centrifugal))
    
    # 4. Integrate (Numerov)
    u = np.zeros(N_calc)
    # Boundary conditions for r -> 0
    # u(r) ~ r^(l+1)
    u[0] = r[0]**(ell + 1)
    u[1] = r[1]**(ell + 1)
    
    numerov_loop(u, k_squared, h2_12, N_calc)
    
    # 5. Normalize
    # The new routine handles dispatching between Coulomb (mu=0) and PlaneWave (mu>0)
    u_norm, phase_info = energy_normalize_continuum(r, u, E_pe, ell, A, U, mu)
    
    return r, u_norm, phase_info

# ==============================================================================
# DRIVER (Adaptive Grid)
# ==============================================================================
def compute_continuum_state(E_pe, ell_cont, species='H', 
                            A=0.0, U=0.0, mu=0.0, 
                            j_total=None, xi_soc=0.0, 
                            **kwargs):
    """
    Wrapper that selects grid size based on energy and calls solver.
    """
    # Grid Selection Strategy
    # Low energy needs larger box to capture long wavelengths
    if E_pe < 1e-4:   R_max, N = 10000.0, 100000
    elif E_pe < 1e-2: R_max, N = 2000.0, 20000
    elif E_pe < 0.1:  R_max, N = 500.0, 10000
    else:             R_max, N = 250.0, 6000
    
    # Solve
    r, u, details = solve_continuum(
        E_pe, ell_cont, species, A, U, mu, 
        j_total=j_total, xi_soc=xi_soc, 
        R_max=R_max, N=N, 
        **kwargs  # Passes r_c, Delta, etc.
    )
    
    return r, u, details