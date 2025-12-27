"""
VERIFICATION SCRIPT: H@C60 Bound States (Saha et al. Figure 2).
Checks if the Hydrogen 1s wavefunction responds correctly to confinement.

Physics:
  - Species: Hydrogen (Z=1, l=0, j=0.5).
  - Potential: Coulomb + Hybrid Confinement (Gaussian + ASW).
  - r_c = 6.7 a.u.
  - Confinement Depths: 0.30, 0.46, 0.56, 1.03 a.u.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from bound import solve_ground_u
    from potential import VGASW_total_debye
except ImportError as e:
    print(f"Error importing physics modules: {e}")
    sys.exit(1)

# ==============================================================================
# 2. PARAMETER SOLVER (Saha Ratio Constraint)
# ==============================================================================
# Constants from Saha et al.
RC = 6.7      # Cage Center
SIGMA = 1.70  # Gaussian Width
DELTA = 2.8   # ASW Width
R_CONST = -24.5 # Shape Constraint Ratio

def solve_saha_params(D_target):
    """
    Solves for (A, U) such that:
      1. Total depth at r_c equals D_target.
      2. The shape matches Saha's ratio constraint.
    """
    def equations(x):
        A, U = x
        if U <= 0 or A >= 0: return [1e5, 1e5] # Penalty for wrong signs
        
        # Eq 1: A - U = -D (Total potential at center is -D)
        eq1 = (A - U) + D_target
        
        # Eq 2: Ratio Constraint (A * sqrt(2pi) * sigma) / U = -24.5
        ratio = (A * np.sqrt(2*np.pi) * SIGMA) / U
        eq2 = ratio - R_CONST
        return [eq1, eq2]

    # Initial Guess
    factor = 1 - R_CONST / (np.sqrt(2*np.pi) * SIGMA)
    U_guess = D_target / factor if abs(factor) > 1e-6 else 0.7
    A_guess = U_guess - D_target
    
    sol = fsolve(equations, [A_guess, U_guess])
    return sol[0], sol[1]

# ==============================================================================
# 3. VERIFICATION LOGIC
# ==============================================================================
def verify_hydrogen():
    print(f"{'='*60}\nVERIFYING HYDROGEN (H@C60) BOUND STATES\n{'='*60}")
    
    # Target Depths from Saha et al. Figure 2
    DEPTHS = [0.30, 0.46, 0.56, 1.03]
    COLORS = ['black', 'red', 'blue', 'magenta']
    
    # Physics Params for Hydrogen 1s
    SPECIES = 'H'
    L_WAVE = 0
    J_TOTAL = 0.5
    XI_SOC = 0.0 # Non-relativistic for H
    
    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=120)
    ax2 = ax1.twinx() # Axis for Potential
    
    # Plot Grid
    r_plot = np.linspace(0.01, 16.0, 500)
    
    print(f"{'Depth (a.u.)':<15} | {'Energy (1s) a.u.':<20} | {'A':<8} | {'U':<8}")
    print("-" * 65)

    for D, col in zip(DEPTHS, COLORS):
        # 1. Get Confinement Params
        A_val, U_val = solve_saha_params(D)
        
        # 2. Solve Bound State
        try:
            r, u, E, _, _ = solve_ground_u(
                VGASW_total_debye, 
                species=SPECIES, 
                R_max=60.0, N=6000,
                A=A_val, U=U_val, mu=0.0, # Free C60
                r_c=RC, Delta=DELTA,      # Correct Constants
                l_wave=L_WAVE, j_total=J_TOTAL, xi_soc=XI_SOC
            )
            
            print(f"{D:<15.2f} | {E:<20.5f} | {A_val:<8.3f} | {U_val:<8.3f}")
            
            # 3. Plot Wavefunction (Left Axis)
            u_interp = np.interp(r_plot, r, u)
            ax1.plot(r_plot, u_interp, color=col, lw=2.5, label=f'D={D} (E={E:.3f})')
            
            # 4. Plot Potential (Right Axis)
            # Use the imported function to ensure potential.py is correct
            V = VGASW_total_debye(r_plot, A=A_val, U=U_val, mu=0.0, 
                                  species=SPECIES, r_c=RC, Delta=DELTA)
            ax2.plot(r_plot, V, color=col, ls='--', lw=1, alpha=0.5)
            
        except Exception as e:
            print(f"{D:<15.2f} | {'FAILED':<20} | {e}")

    # --- Formatting ---
    
    ax1.set_xlabel("Radius (a.u.)", fontsize=12, fontweight='bold')
    ax1.set_ylabel(r"Radial Wavefunction $u_{1s}(r)$", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Potential V(r) (a.u.)", fontsize=12, rotation=270, labelpad=15)
    
    ax1.set_xlim(0, 16)
    ax1.axhline(0, color='gray', lw=0.5)
    
    # Mark Cage Position
    ax1.axvline(RC, color='green', ls=':', alpha=0.8)
    ax1.text(RC, 0.9, 'C60 Cage\n(6.7 a.u.)', color='green', ha='center', transform=ax1.get_xaxis_transform())

    plt.title("H@C60 Wavefunctions (Saha et al. Reproduction)\nUsing corrected r_c=6.7", fontsize=14)
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines.append(plt.Line2D([0],[0], color='gray', ls='--', label='Potential'))
    labels.append('Potential')
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig("Hydrogen_Saha_Fig2_Verification.png")
    print(f"\n>>> Plot saved to 'Hydrogen_Saha_Fig2_Verification.png'")
    plt.show()

if __name__ == "__main__":
    verify_hydrogen()