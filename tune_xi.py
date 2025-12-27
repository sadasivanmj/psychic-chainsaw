"""
Argon Spin-Orbit Calibrator.
Finds the exact 'xi_soc' parameter to match experimental fine-structure splitting.
"""
import numpy as np
from src.potential import VGASW_total_debye
from src.bound import solve_ground_u

# Constants
Ha_to_eV = 27.211386
TARGET_SPLITTING_eV = 0.177  # Experimental val for Ar (3p_3/2 vs 3p_1/2)

def calibrate_xi():
    print("="*60)
    print("CALIBRATING SPIN-ORBIT COUPLING FOR ARGON")
    print("="*60)
    
    # 1. First Pass: Use a small guess
    xi_guess = 0.0001
    print(f"\n[1] Running Test with xi_guess = {xi_guess}...")
    
    # Solve j=3/2 (Ground state)
    _, _, E_32, _, _ = solve_ground_u(
        VGASW_total_debye, species='Ar', R_max=50.0, N=10000,
        l_wave=1, j_total=1.5, xi_soc=xi_guess, A=0, U=0
    )
    
    # Solve j=1/2 (Excited state)
    _, _, E_12, _, _ = solve_ground_u(
        VGASW_total_debye, species='Ar', R_max=50.0, N=10000,
        l_wave=1, j_total=0.5, xi_soc=xi_guess, A=0, U=0
    )
    
    # Calculate Splitting
    delta_E_au = abs(E_12 - E_32)
    delta_E_eV = delta_E_au * Ha_to_eV
    
    print(f"    E(3/2): {E_32:.6f} a.u.")
    print(f"    E(1/2): {E_12:.6f} a.u.")
    print(f"    Splitting: {delta_E_eV:.6f} eV")
    
    # 2. Linear Scaling to find Exact Xi
    # Formula: xi_target = xi_guess * (Target_Split / Current_Split)
    scaling_factor = TARGET_SPLITTING_eV / delta_E_eV
    xi_final = xi_guess * scaling_factor
    
    print(f"\n[2] Calibration Result")
    print(f"    Target Splitting: {TARGET_SPLITTING_eV} eV")
    print(f"    Required xi_soc:  {xi_final:.8e}")
    
    # 3. Verification Run
    print(f"\n[3] Verifying with xi_soc = {xi_final:.8e}...")
    _, _, E_32_v, _, _ = solve_ground_u(
        VGASW_total_debye, species='Ar', R_max=50.0, N=10000,
        l_wave=1, j_total=1.5, xi_soc=xi_final, A=0, U=0
    )
    _, _, E_12_v, _, _ = solve_ground_u(
        VGASW_total_debye, species='Ar', R_max=50.0, N=10000,
        l_wave=1, j_total=0.5, xi_soc=xi_final, A=0, U=0
    )
    
    final_split = abs(E_12_v - E_32_v) * Ha_to_eV
    print(f"    Final Splitting:  {final_split:.6f} eV")
    
    if abs(final_split - TARGET_SPLITTING_eV) < 0.001:
        print("\n✅ SUCCESS: Parameter tuned successfully.")
        print(f"   Use xi_soc = {xi_final:.8e} in your simulations.")
    else:
        print("\n⚠️ WARNING: Non-linearity detected. You may need one more iteration.")

if __name__ == "__main__":
    calibrate_xi()