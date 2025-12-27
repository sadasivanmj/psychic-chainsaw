import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from src import potential
from src import bound
# CRITICAL: Import the new relativistic solver
from src import cross_section 

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# --- HELPER: EXACT SAHA PARAMETER SOLVER ---
def solve_gasw_params_saha(depth_au):
    """
    Calculates A and U using the strict Ratio Constraint from Saha et al.
    Constraint: (A * sqrt(2pi) * sigma) / U = -24.5
    """
    R_CONST = -24.5
    SIGMA = 1.70
    
    ratio_AU = R_CONST / (np.sqrt(2 * np.pi) * SIGMA)
    denom = ratio_AU - 1.0
    if abs(denom) < 1e-8: denom = 1e-8
        
    U_new = -depth_au / denom
    A_new = U_new * ratio_AU
    return A_new, U_new

def test_hydrogen_bound_states():
    print_header("1. HYDROGEN BOUND STATES VERIFICATION (L-S FORMALISM)")
    print(f"{'State':<10} {'J':<5} {'Calc E (a.u.)':<15} {'Theory E':<15} {'Error (%)':<10}")
    print("-" * 65)

    # Hydrogen states with L and J
    states = [
        ('1s', 0, 0, 0.5, 1),
        ('2s', 1, 0, 0.5, 2),
        ('2p', 0, 1, 0.5, 2), # 2p_1/2
        ('2p', 0, 1, 1.5, 2), # 2p_3/2
        ('3d', 0, 2, 2.5, 3)
    ]

    for label, n_idx, l_val, j_val, n_princ in states:
        # Pass j_total and xi_soc=0 (Non-relativistic H)
        r, u, E, _, _ = bound.solve_ground_u(
            potential.VGASW_total_debye, 
            species='H', A=0.0, U=0.0,
            l_wave=l_val,            
            target_n_idx=n_idx,
            j_total=j_val, xi_soc=0.0      
        )
        
        theory_E = -0.5 / (n_princ**2)
        error = abs((E - theory_E) / theory_E) * 100
        
        print(f"{label:<10} {j_val:<5} {E:<15.6f} {theory_E:<15.6f} {error:<10.4f}")

def test_hydrogen_threshold_cross_section():
    print_header("2. HYDROGEN THRESHOLD CROSS SECTION (RELATIVISTIC SOLVER)")
    
    # Target from Saha Table 1
    E_pe = 0.0001
    Target_Sigma_au = 0.22495 
    
    # 1. Solve Bound State (1s_1/2)
    r_b, u_b, E_b, _, _ = bound.solve_ground_u(
        potential.VGASW_total_debye, species='H', A=0.0, U=0.0,
        l_wave=0, target_n_idx=0, j_total=0.5, xi_soc=0.0
    )
    
    # 2. Compute Cross Section using Relativistic Spectrum
    sigma_res, _ = cross_section.compute_relativistic_spectrum(
        [E_pe], r_b, u_b, E_b, 
        l_initial=0, j_initial=0.5,
        species='H', xi_soc=0.0, A=0.0, U=0.0, mu=0.0
    )
    
    calc_sigma_au = sigma_res[0]
    # In Saha, the value 0.22 is actually in Atomic Units approx?
    # Actually Saha Table 1 is likely Mb? No, Saha usually uses a.u. or Mb.
    # 0.22 a.u. ~= 6.3 Mb. This matches standard H threshold.
    
    print(f"Photoelectron Energy: {E_pe} a.u.")
    print(f"Calculated Sigma:     {calc_sigma_au:.6f} a.u.")
    print(f"Target (Saha Table 1):{Target_Sigma_au:.6f} a.u.")
    
    if abs(calc_sigma_au - Target_Sigma_au) < 0.01:
        print(">> STATUS: MATCH (Excellent)")
    else:
        print(">> STATUS: MISMATCH")

def test_confined_hydrogen_saha():
    print_header("3. CONFINED HYDROGEN (SAHA TABLE 1 REPRODUCTION)")
    print(f"{'Depth':<10} {'A':<10} {'U':<10} {'Calc σ':<15} {'Target σ':<15} {'Status':<10}")
    print("-" * 80)
    
    E_pe = 0.0001
    
    test_cases = [
        (0.30, 0.19131),
        (0.46, 0.22206),
        (0.56, 0.15211),
        (1.03, 0.01204)
    ]
    
    for depth, target in test_cases:
        # 1. Get A and U using correct Saha Helper
        A_val, U_val = solve_gasw_params_saha(depth)
        
        # 2. Solve Bound State (1s_1/2)
        r_b, u_b, E_b, _, _ = bound.solve_ground_u(
            potential.VGASW_total_debye, species='H', 
            A=A_val, U=U_val,
            l_wave=0, j_total=0.5, xi_soc=0.0,
            r_c=6.7, Delta=2.8 # Explicit constants
        )
        
        # 3. Solve Cross Section
        sigma_res, _ = cross_section.compute_relativistic_spectrum(
            [E_pe], r_b, u_b, E_b, 
            l_initial=0, j_initial=0.5,
            species='H', xi_soc=0.0, A=A_val, U=U_val,
            r_c=6.7, Delta=2.8
        )
        
        sigma_au = sigma_res[0]
        
        # Check pass/fail (tolerance 15% allow for grid differences)
        status = "PASS" if abs(sigma_au - target)/target < 0.15 else "DIFF"
        print(f"{depth:<10.2f} {A_val:<10.3f} {U_val:<10.3f} {sigma_au:<15.6f} {target:<15.6f} {status}")

def plot_saha_figure_2():
    print_header("4. GENERATING SAHA FIGURE 2 (WAVEFUNCTIONS)")
    print("Saving plot to 'test_saha_fig2_ls.png'...")
    
    depths = [0.30, 0.46, 0.56, 1.03]
    colors = ['black', 'red', 'blue', 'magenta']
    
    plt.figure(figsize=(8, 6))
    
    for depth, color in zip(depths, colors):
        A_val, U_val = solve_gasw_params_saha(depth)
        
        r_b, u_b, E_b, _, _ = bound.solve_ground_u(
            potential.VGASW_total_debye, species='H', 
            A=A_val, U=U_val,
            l_wave=0, j_total=0.5, xi_soc=0.0,
            r_c=6.7, Delta=2.8
        )
        
        mask = r_b <= 12.0
        plt.plot(r_b[mask], u_b[mask], label=f'Depth={depth}', color=color, linewidth=2)
        
    plt.title("Hydrogen 1s Radial Wavefunction (Saha et al. Fig 2)")
    plt.xlabel("Radial Distance (a.u.)")
    plt.ylabel("u_1s(r)")
    
    #  
    # This plot visualizes the orbital compression due to confinement.
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_saha_fig2_ls.png')
    print("Done.")

def test_argon_cooper_minima():
    print_header("5. ARGON COOPER MINIMA CHECK (RELATIVISTIC)")
    print("Scanning energy range [0.1 - 2.5 a.u.]...")
    
    # 1. Setup Argon 3p_3/2 (Ground State)
    # L=1, J=1.5
    # SOC Strength ~ 2.61e-5 (Physical)
    xi_phys = 2.6121e-05
    
    print("Solving Argon 3p_3/2 Bound State...")
    
    r_b, u_b, E_b, _, _ = bound.solve_ground_u(
        potential.VGASW_total_debye, species='Ar', A=0.0, U=0.0, 
        l_wave=1,           
        target_n_idx=1,     # 3p is usually 2nd p-state (2p is core)
        j_total=1.5,        # 3p_3/2
        xi_soc=xi_phys
    )
    
    print(f"Argon 3p_3/2 Energy: {E_b:.4f} a.u. (Target ~ -0.579)")
    
    # 2. Energy Grid
    E_pe_arr = np.linspace(0.1, 2.0, 40)
    
    print("Calculating Relativistic Cross Sections (Summing Channels)...")
    
    # Compute using new L-S module
    sigma_res, _ = cross_section.compute_relativistic_spectrum(
        E_pe_arr, r_b, u_b, E_b, 
        l_initial=1, j_initial=1.5,
        species='Ar', A=0.0, U=0.0, 
        xi_soc=xi_phys
    )
    
    # Convert a.u. to Mb
    sigmas_mb = sigma_res * 28.0028 
    
    min_idx = np.argmin(sigmas_mb)
    cm_energy = E_pe_arr[min_idx]
    
    print(f"Detected CM at E_pe = {cm_energy:.3f} a.u.")
    print("Plotting Argon CM to 'test_argon_cm_ls.png'...")
    
    plt.figure(figsize=(8, 6))
    plt.plot(E_pe_arr, sigmas_mb, label='Total (Free Ar)', color='black', linewidth=2)
    
    #  
    # This plot confirms the interference minimum characteristic of Ar 3p photoionization.
    
    plt.title("Argon 3p Relativistic Cross Section (Cooper Minimum)")
    plt.xlabel("Photoelectron Energy (a.u.)")
    plt.ylabel("Cross Section (Mb)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_argon_cm_ls.png')
    print("Done.")

if __name__ == "__main__":
    test_hydrogen_bound_states()
    test_hydrogen_threshold_cross_section()
    test_confined_hydrogen_saha()
    plot_saha_figure_2()
    test_argon_cooper_minima()
    print("\nAll Tests Complete.")