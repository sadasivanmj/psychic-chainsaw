"""
Calculate RELATIVISTIC Photoionization Cross Section for Free Argon.
Physics: L-S Coupling (Spin-Orbit) in Intermediate Coupling Scheme.
Output: CSV file and Plot of Total + Partial Cross Sections.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import cpu_count

# --- START OF CRITICAL FIX ---
# Add the project root (where the 'src' folder is located) to the path.
# This assumes the script is inside the root folder, which contains 'src'.
# If 'argon_cross_section.py' is in the same directory as 'src', this should work.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# --- END OF CRITICAL FIX ---


# Import RELATIVISTIC Physics Modules using the 'src.' prefix
# These must match your file names inside the 'src' folder.
from src.potential import VGASW_total_debye
from src.bound import solve_ground_u
from src.cross_section import compute_relativistic_spectrum

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SPECIES = 'Ar'
OCCUPATION = 6.0         # 3p^6 shell
XI_SOC = 2.6121e-05      # Tuned Spin-Orbit Strength for Ar
OUTPUT_CSV = "Ar_Relativistic_CrossSection.csv"
OUTPUT_IMG = "Ar_Relativistic_CrossSection.png"

# Physics Constants
Ha_to_eV = 27.211386
Mb_conv = 28.0028

def run_simulation():
    print(f"[{SPECIES}] Starting Relativistic Simulation (L-S Coupling)...")
    print(f"    Spin-Orbit Xi:   {XI_SOC:.4e}")
    print(f"    Cores Available: {cpu_count()}")
    
    # 1. Define Energy Grid (Kinetic Energy in a.u.)
    e_kin_au = np.concatenate([
        np.linspace(0.01, 1.2, 250),
        np.linspace(1.25, 4.5, 100)
    ])
    print(f"    Grid Size: {len(e_kin_au)} points")

    t0 = time.time()

    # --------------------------------------------------------------------------
    # STEP 1: Bound State (Argon 3p_3/2 Ground State)
    # --------------------------------------------------------------------------
    print("\n>>> [1/2] Solving Bound State (3p_3/2)...")
    r_b, u_b, E_b, l_init, _ = solve_ground_u(
        VGASW_total_debye, 
        species=SPECIES, 
        R_max=60.0, 
        N=12000, 
        A=0.0, U=0.0,
        l_wave=1, 
        j_total=1.5, 
        xi_soc=XI_SOC
    )
    
    Ip_eV = abs(E_b) * Ha_to_eV
    print(f"    Bound Energy (3p_3/2): {E_b:.4f} a.u. ({Ip_eV:.2f} eV)")

    # --------------------------------------------------------------------------
    # STEP 2: Continuum Spectrum (Relativistic Channels)
    # --------------------------------------------------------------------------
    print("\n>>> [2/2] Computing Relativistic Spectrum...")
    print("    Channels: 3p_3/2 -> [ed_5/2, ed_3/2, es_1/2]")
    
    sigma_total_au, details_list = compute_relativistic_spectrum(
        e_kin_au, r_b, u_b, E_b, 
        l_initial=1, j_initial=1.5, 
        species=SPECIES, xi_soc=XI_SOC,
        A=0.0, U=0.0, mu=0.0, 
        n_workers=None
    )
    
    elapsed = time.time() - t0
    print(f"    Simulation Complete in {elapsed:.1f} seconds.")

    # --------------------------------------------------------------------------
    # STEP 3 & 4: Post-Processing, Save, and Plot
    # (Same as before, ensuring unit conversion and data saving)
    # --------------------------------------------------------------------------
    sig_d5 = []
    sig_d3 = []
    sig_s1 = []
    
    for det in details_list:
        sig_d5.append(det.get('sigma_L2_J5/2', 0.0))
        sig_d3.append(det.get('sigma_L2_J3/2', 0.0))
        sig_s1.append(det.get('sigma_L0_J1/2', 0.0))
        
    photon_ev = (e_kin_au * Ha_to_eV) + Ip_eV
    factor = Mb_conv * OCCUPATION
    
    sigma_total_mb = sigma_total_au * factor
    sigma_d5_mb = np.array(sig_d5) * factor
    sigma_d3_mb = np.array(sig_d3) * factor
    sigma_s1_mb = np.array(sig_s1) * factor
    
    # Save to CSV
    df = pd.DataFrame({
        'Kinetic_Energy_au': e_kin_au,
        'Photon_Energy_eV': photon_ev,
        'Sigma_Total_Mb': sigma_total_mb,
        'Sigma_d5_2_Mb': sigma_d5_mb,
        'Sigma_d3_2_Mb': sigma_d3_mb,
        'Sigma_s1_2_Mb': sigma_s1_mb
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n>>> Data saved to '{OUTPUT_CSV}'")

    # Plotting
    min_idx = np.argmin(sigma_total_mb)
    min_E = photon_ev[min_idx]
    
    plt.figure(figsize=(9, 7), dpi=120)
    plt.plot(photon_ev, sigma_total_mb, 'k-', linewidth=2.5, label='Total Relativistic')
    plt.plot(photon_ev, sigma_d5_mb, 'r--', linewidth=1.5, label=r'$3p \to \epsilon d_{5/2}$ (Dom)')
    plt.plot(photon_ev, sigma_d3_mb, 'g--', linewidth=1.5, label=r'$3p \to \epsilon d_{3/2}$')
    plt.plot(photon_ev, sigma_s1_mb, 'b:', linewidth=1.5, label=r'$3p \to \epsilon s_{1/2}$')
    
    plt.xlabel("Photon Energy (eV)", fontsize=12, fontweight='bold')
    plt.ylabel("Cross Section (Mb)", fontsize=12, fontweight='bold')
    plt.title("Argon Relativistic Photoionization (Spin-Orbit)", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(15, 80)
    plt.ylim(0, 45)
    
    plt.annotate(f'Cooper Min\n{min_E:.1f} eV', 
                 xy=(min_E, sigma_total_mb[min_idx]), 
                 xytext=(min_E, sigma_total_mb[min_idx] + 8),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f">>> Plot saved to '{OUTPUT_IMG}'")
    plt.show()

if __name__ == "__main__":
    run_simulation()