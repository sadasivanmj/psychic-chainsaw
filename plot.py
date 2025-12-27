"""
Reproduce Saha et al. Fig 10 using RELATIVISTIC (L-S Coupled) Solver.
Target: Ar@C60 Cross Section Shape (Cooper Minimum Shift).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from matplotlib.ticker import AutoMinorLocator

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORTS ---
try:
    from potential import VGASW_total_debye
    from bound import solve_ground_u
    from cross_section import compute_relativistic_spectrum
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Scale factor to match Saha's arbitrary y-axis units (approx proportional to Mb)
# Adjusted for Relativistic Cross Section (Mb) -> Saha Unit
PAPER_SCALE_FACTOR = 0.0022 

# Physics Constants for Argon
SPECIES = 'Ar'
L_INIT = 1       # p-orbital
J_INIT = 1.5     # 3p_3/2 (Ground State)
XI_SOC = 2.61e-5 # Spin-Orbit Strength

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_gasw_params_analytic(D_target, sigma=1.70, R_const=-24.5):
    """Analytic solution for GASW parameters (A, U) given depth D."""
    K = np.sqrt(2.0 * np.pi) * sigma
    denominator = 1.0 - (R_const / K)
    if abs(denominator) < 1e-10: return -3.59, 0.7
    U = D_target / denominator
    A = U - D_target
    return A, U

def calculate_spectrum_for_plotting(E_pe_array, A, U, mu):
    """
    Computes the total cross section spectrum using the L-S coupled solver.
    Returns: Array of values scaled to match Saha Fig 10.
    """
    # 1. Solve Bound State (3p_3/2)
    # We pass r_c=6.7 and Delta=2.8 explicitly to match Saha parameters
    r_b, u_b, E_b, _, _ = solve_ground_u(
        VGASW_total_debye, 
        species=SPECIES, 
        R_max=60.0, N=6000,
        A=A, U=U, mu=mu,
        r_c=6.7, Delta=2.8,
        l_wave=L_INIT, j_total=J_INIT, xi_soc=XI_SOC
    )
    
    # 2. Compute Relativistic Spectrum
    # This returns Total Sigma in Atomic Units (a.u.)
    sigma_total_au, _ = compute_relativistic_spectrum(
        E_pe_array, r_b, u_b, E_b,
        l_initial=L_INIT, j_initial=J_INIT,
        species=SPECIES, xi_soc=XI_SOC,
        A=A, U=U, mu=mu,
        r_c=6.7, Delta=2.8, # Pass confinement params
        n_workers=None
    )
    
    # 3. Convert to Saha Units
    # Saha plots proportional to E * |D|^2. 
    # Sigma_total_au is proportional to E * |D|^2 already.
    # We just apply the scaling factor.
    return sigma_total_au * PAPER_SCALE_FACTOR

# ==============================================================================
# 3. MAIN SIMULATION
# ==============================================================================

def main():
    print("="*80)
    print(f"REPRODUCING SAHA FIG 10 (RELATIVISTIC L-S COUPLING)")
    print("="*80)

    # Configuration
    DEPTHS = [0.30, 0.46, 0.56]
    # Energy Grid: 0.3 to 3.0 a.u. (Photoelectron Energy)
    E_pe = np.linspace(0.3, 3.0, 100) 
    
    results_asw = {}
    results_gasw = {}

    # ---------------------------------------------------------
    # 1. CALCULATIONS
    # ---------------------------------------------------------
    print("Calculating Free Argon...")
    res_free = calculate_spectrum_for_plotting(E_pe, A=0.0, U=0.0, mu=0.0)

    print("Calculating ASW Cases...")
    for U_val in DEPTHS:
        print(f"  ASW Depth U={U_val}")
        # ASW: A=0, U=Depth
        results_asw[U_val] = calculate_spectrum_for_plotting(E_pe, A=0.0, U=U_val, mu=0.0)

    print("Calculating GASW Cases...")
    for D_val in DEPTHS:
        print(f"  GASW Depth D={D_val}")
        # GASW: A, U from analytic solution
        A_calc, U_calc = get_gasw_params_analytic(D_val)
        results_gasw[D_val] = calculate_spectrum_for_plotting(E_pe, A=A_calc, U=U_calc, mu=0.0)

    # ---------------------------------------------------------
    # 2. PLOTTING
    # ---------------------------------------------------------
    print("\nGenerating Final Plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True)
    plt.subplots_adjust(hspace=0.20)
    
    colors = ['red', 'lime', 'blue'] 
    
    def style_axis(ax, label_text):
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12, width=1.5)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # SAHA LIMITS
        ax.set_ylim(0.01, 0.08)
        ax.set_xlim(0.3, 3.0)
        
        ax.set_ylabel(r"$|d_{if}|^2$ (arb. units)", fontsize=14, fontweight='bold', labelpad=10)
        
        ax.text(0.08, 0.85, label_text, transform=ax.transAxes, fontsize=11, 
                fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', pad=4.0))

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # --- Panel (a): ASW ---
    ax1.plot(E_pe, res_free, 'k-', lw=2.5, label='Free')
    for i, D in enumerate(DEPTHS):
        ax1.plot(E_pe, results_asw[D], color=colors[i], lw=2.5, label=f'U={D:.2f} a.u.')
    
    style_axis(ax1, r"(a)   Ar@C$_{60}$-ASW")
    ax1.legend(loc='upper right', frameon=True, fontsize=10, 
               edgecolor='black', framealpha=1.0, borderpad=0.6)

    # --- Panel (b): GASW ---
    ax2.plot(E_pe, res_free, 'k-', lw=2.5, label='Free')
    for i, D in enumerate(DEPTHS):
        label_txt = f'V$_{{GASW}}$(r$_c$)={D:.2f} a.u.'
        ax2.plot(E_pe, results_gasw[D], color=colors[i], lw=2.5, label=label_txt)
        
    style_axis(ax2, r"(b)   Ar@C$_{60}$-GASW")
    ax2.set_xlabel("Photoelectron energy (a.u.)", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=False, fontsize=10)

    plt.savefig('saha_fig10_relativistic.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: saha_fig10_relativistic.png")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()