# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 19:54:35 2025

@author: harry
"""

"""
Verification of Relativistic SAE Model against Zhang et al. (2025).
Reproduces Table I: Low-lying bound states of Xenon with Spin-Orbit Coupling.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from src.potential import (
    VGASW_total_debye, 
    SAE_PARAMS, 
    TongLinParams, 
    get_LS_factor
)

# ==============================================================================
# 1. SETUP XENON PARAMETERS
# ==============================================================================
Ha_to_eV = 27.211386
XI_XE = 2.1602e-5

SAE_PARAMS['Xe'] = {
    'default': TongLinParams(
        Z_c=1.0,
        a1=54.5084, a2=4.4941,
        a3=-92.7753, a4=7.0344,
        a5=8.2042, a6=0.8308,
        ground_l=1,      
        ground_n_idx=3   # 5p is the 4th state (idx 3) for l=1
    )
}

# ==============================================================================
# 2. SOLVER ENGINE
# ==============================================================================
def get_valence_states(species, l_wave, j_total, xi, max_n=12):
    """
    Solves for eigenvalues and maps them to principal quantum number n.
    Returns a dict: {n: energy_eV}
    """
    # Grid Setup (High resolution for deep core)
    R_max = 80.0
    N_grid = 15000 
    r = np.linspace(1e-5, R_max, N_grid)
    dr = r[1] - r[0]
    
    V = VGASW_total_debye(
        r, species=species, 
        l_wave=l_wave, j_total=j_total, xi_soc=xi, 
        A=0, U=0
    )
    
    r_safe = np.clip(r, 1e-12, None)
    V_eff = V + l_wave * (l_wave + 1) / (2.0 * r_safe**2)
    
    k = 1.0 / (2.0 * dr**2)
    d = 2.0 * k + V_eff[1:-1]
    e = -k * np.ones(len(d) - 1)
    
    # Solve for enough states to cover the valence shell
    # We need indices up to ~10 to find 9s, 8d etc.
    try:
        n_solve = 15 
        vals, _ = eigh_tridiagonal(d, e, select='i', select_range=(0, n_solve-1))
        
        # Map Index k -> Principal Quantum Number n
        # n = l + 1 + k
        results = {}
        for k, energy_au in enumerate(vals):
            n = l_wave + 1 + k
            results[n] = energy_au * Ha_to_eV
            
        return results
    except Exception as err:
        return {}

# ==============================================================================
# 3. REPRODUCTION LOOP
# ==============================================================================
def reproduce_table():
    print("="*90)
    print("REPRODUCING TABLE I: Xenon Low-Lying Bound States (eV)")
    print("Corrected for Core State Indices")
    print("="*90)
    
    # Define Targets from Table I
    # (Orbital Label, L, n, Reference Energy for J=3/2 or J=5/2 if L>=2)
    # Using J=3/2 as primary check, J=1/2 as secondary
    
    # We will simulate and look up these specific (n, l) states
    targets = [
        # Label, n, l
        ('5p', 5, 1), ('6s', 6, 0), ('6p', 6, 1), ('5d', 5, 2), 
        ('7s', 7, 0), ('7p', 7, 1), ('6d', 6, 2), ('4f', 4, 3),
        ('8s', 8, 0), ('8p', 8, 1), ('7d', 7, 2), ('5f', 5, 3)
    ]
    
    # Paper Data for Comparison (J=3/2 mostly, J=1/2 for s)
    # Dictionary Key: "{Label}_{J_numerator}"
    paper_vals = {
        '5p_3': -12.1592, '5p_1': -13.4424,
        '6s_1': -3.7533,  # s is always 1/2
        '6p_3': -2.3413,  '6p_1': -2.4361,
        '5d_3': -2.0001,  '5d_5': -2.0001, # Paper lists 5d under J=3/2 and J=1/2 cols?
                                           # Wait, d splits into 3/2 and 5/2.
                                           # Zhang Table I lists J=3/2 and J=1/2 columns.
                                           # But d-states are 5/2 and 3/2. 
                                           # Let's check their notation. 
                                           # Usually "J=1/2" col implies the lower J for that L?
                                           # No, let's just calculate our J's and see what matches.
        '4f_5': -0.8579,  '4f_7': -0.8579,
    }

    print(f"{'State':<6} {'n':<3} {'l':<3} {'J':<5} {'Calc (eV)':<12} {'Paper Ref':<12} {'Diff':<8}")
    print("-" * 75)

    for label, n, l in targets:
        # Determine J values to calc
        if l == 0:
            js = [0.5]
        else:
            js = [l - 0.5, l + 0.5]
            
        for j in js:
            j_str = f"{int(2*j)}/2"
            
            # Solve
            states = get_valence_states('Xe', l, j, XI_XE)
            calc_E = states.get(n, np.nan)
            
            # Try to match with paper dict key (e.g. 5p_3 for 5p_3/2)
            # Paper J=3/2 column -> key suffix _3
            # Paper J=1/2 column -> key suffix _1
            # Note: For d-waves (5/2, 3/2), paper columns are likely just two split components.
            # We will print all and compare visually if auto-match fails.
            
            ref_key = f"{label}_{int(2*j)}"
            ref = paper_vals.get(ref_key, None)
            
            # Visual helper for table
            ref_txt = f"{ref:.4f}" if ref else "---"
            diff_txt = f"{abs(calc_E - ref):.4f}" if ref else "---"
            
            # Special logic for paper's specific table layout
            # If we calculated -12.15, that's clearly the -12.1592 entry.
            
            print(f"{label:<6} {n:<3} {l:<3} {j_str:<5} {calc_E:<12.4f} {ref_txt:<12} {diff_txt:<8}")

if __name__ == "__main__":
    reproduce_table()