"""
Compare Free vs. Confined Argon (Depth=0.30 a.u.)
Focus: Zoomed view of the Cooper Minimum shift.
Physics: Relativistic (Spin-Orbit included), Exact Tong & Lin Parameters.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

# --- ADD PROJECT ROOT TO PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Physics Modules
from src.potential import VGASW_total_debye, solve_gasw_parameters, SAE_PARAMS, TongLinParams
from src.bound import solve_ground_u
from src.cross_section import compute_relativistic_spectrum

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SPECIES = 'Ar'
OCCUPATION = 6.0         
XI_SOC = 2.6121e-05  # Relativistic Spin-Orbit Strength

# Zoomed Grid for Cooper Minimum (0.5 - 1.5 a.u.)
E_KIN_GRID = np.linspace(0.5, 1.5, 200)

OUTPUT_IMG = "Argon_Cooper_Shift_Zoomed.png"

def ensure_exact_tong_lin():
    """Enforce Exact Tong & Lin (2005) Parameters"""
    SAE_PARAMS['Ar']['default'] = TongLinParams(
        Z_c=1.0, 
        a1=16.039, a2=2.007, 
        a3=-25.543, a4=4.525, 
        a5=0.961, a6=0.443,
        ground_l=1, ground_n_idx=1
    )

def calculate_profile(depth_val, label):
    print(f"\n>>> Simulating {label}...")
    
    # 1. Get Potential Parameters
    if depth_val == 0.0:
        A_val, U_val = 0.0, 0.0
        depth_str = "Free"
    else:
        # Use the lookup table in potential.py
        A_val, U_val = solve_gasw_parameters(depth_val)
        depth_str = f"Depth={depth_val}"
    
    print(f"    Params: A={A_val:.4f}, U={U_val:.4f}")

    # 2. Solve Bound State
    # target_n_idx=1 for Ar 3p (Standard Tong-Lin)
    r_b, u_b, E_b, _, _ = solve_ground_u(
        VGASW_total_debye, 
        species=SPECIES, 
        R_max=60.0, N=12000, 
        A=A_val, U=U_val,
        l_wave=1, j_total=1.5, xi_soc=XI_SOC,
        target_n_idx=1 
    )
    print(f"    Bound Energy: {E_b:.4f} a.u.")

    # 3. Compute Cross Section
    # Using n_workers for speed
    n_cpu = max(1, cpu_count() - 1)
    
    sigma_total, _ = compute_relativistic_spectrum(
        E_KIN_GRID, r_b, u_b, E_b, 
        l_initial=1, j_initial=1.5, 
        species=SPECIES, xi_soc=XI_SOC,
        A=A_val, U=U_val, n_workers=n_cpu
    )
    
    # Scale by occupation
    sigma_total *= OCCUPATION
    
    # Find Minimum
    min_idx = np.argmin(sigma_total)
    min_E = E_KIN_GRID[min_idx]
    
    return sigma_total, min_E, E_b

def main():
    ensure_exact_tong_lin()
    
    # --- Run Simulations ---
    sig_free, min_free, E_free = calculate_profile(0.0, "Free Argon")
    sig_conf, min_conf, E_conf = calculate_profile(0.30, "Confined Argon (0.30 a.u.)")
    
    # --- Plotting ---
    plt.figure(figsize=(10, 7), dpi=120)
    
    # Plot Curves
    plt.plot(E_KIN_GRID, sig_free, 'k-', lw=2.5, label=f'Free Ar (Min @ {min_free:.3f} a.u.)')
    plt.plot(E_KIN_GRID, sig_conf, 'r-', lw=2.5, label=f'Confined 0.30 (Min @ {min_conf:.3f} a.u.)')
    
    # Add Arrows indicating the shift
    plt.annotate('', xy=(min_conf, 0.02), xytext=(min_free, 0.02),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    plt.text((min_free + min_conf)/2, 0.025, "Shift", color='blue', ha='center', fontweight='bold')
    
    # Vertical lines for minima
    plt.axvline(min_free, color='k', linestyle=':', alpha=0.5)
    plt.axvline(min_conf, color='r', linestyle=':', alpha=0.5)

    # Formatting
    plt.xlabel("Photoelectron Energy (a.u.)", fontsize=12, fontweight='bold')
    plt.ylabel("Cross Section (a.u.)", fontsize=12, fontweight='bold')
    plt.title(f"Cooper Minimum Shift: Free vs Confined Argon\n(Exact Tong & Lin Params, Relativistic)", fontsize=14)
    
    plt.xlim(0.5, 1.5)
    plt.ylim(0.0, 0.20) # Zoomed Y-axis for clarity
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\n>>> Plot saved to '{OUTPUT_IMG}'")
    print(f">>> Observed Shift: {min_conf - min_free:.3f} a.u.")
    
    # Conclusion text
    if min_conf > min_free:
        print(">>> RESULT: The minimum shifted RIGHT (Higher Energy).")
    else:
        print(">>> RESULT: The minimum shifted LEFT (Lower Energy).")

    plt.show()

if __name__ == "__main__":
    main()