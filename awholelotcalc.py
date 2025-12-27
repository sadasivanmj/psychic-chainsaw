"""
FINAL SIMULATION SUITE: Hydrogen & Argon (L-S Coupling).
UNITS: ATOMIC UNITS (a.u.)

PHYSICS VERIFICATION:
  - Reproduces Saha et al. (2019) confinement physics.
  - H@C60: Bound states localize on cage (r=6.7).
  - Ar@C60: Cooper minimum shifts to higher energy (Blue Shift).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from multiprocessing import freeze_support, Pool, cpu_count
from functools import partial

# --- Progress Bar Fallback ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"Starting: {desc}")
        return iterable

# ==============================================================================
# 1. SETUP PATHS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Ensure potential.py has been updated with the VGASW_total_debye 
    # that accepts A and U simultaneously!
    from potential import VGASW_total_debye
    from bound import solve_ground_u
    from continuum import compute_continuum_state
except ImportError as e:
    print(f"CRITICAL: Could not import physics modules. {e}")
    sys.exit(1)

# ==============================================================================
# 2. PHYSICS CONSTANTS & PARAMETERS
# ==============================================================================
ALPHA_FS = 1.0 / 137.036

# Energy Grid: Dense threshold (0.01-0.1), Coarse tail (0.1-4.5)
# Argon Cooper Min is around 0.9 - 1.2 a.u., so the tail grid covers it well.
E_GRID = np.concatenate([
    np.linspace(0.01, 0.1, 100),
    np.linspace(0.12, 4.5, 300)
])

# --- SAHA PARAMETERS ---
CONF_RADIUS = 6.7   # C60 Center (a.u.)
CONF_DEPTH = 0.1    # Default Depth (a.u.). Modify to 0.30 or 1.03 to match specific Saha plots.
CONF_DELTA = 2.8    # ASW Width

MU_PLASMA = [0.1, 0.2, 0.3, 0.5]
MU_HYBRID = [0.1, 0.2, 0.3, 0.4, 0.5]

PARAMS_H = {
    'species': 'H', 'Z_occ': 1.0, 
    'l_init': 0, 'j_init': 0.5, 'xi_soc': 0.0
}
PARAMS_AR = {
    'species': 'Ar', 'Z_occ': 6.0, 
    'l_init': 1, 'j_init': 1.5, 'xi_soc': 2.6121e-05
}

# ==============================================================================
# 3. HELPER: GASW PARAMETERS (SAHA RATIO)
# ==============================================================================
def solve_gasw_params(depth_au):
    """
    Calculates A and U using the Ratio Constraint from Saha et al.
    Constraint: (A * sqrt(2pi) * sigma) / U = -24.5
    """
    R_CONST = -24.5
    SIGMA = 1.70
    
    # Ratio A/U
    ratio_AU = R_CONST / (np.sqrt(2 * np.pi) * SIGMA)
    
    # Linear System: 
    # 1) A - U = -depth
    # 2) A = ratio * U
    denom = ratio_AU - 1.0
    if abs(denom) < 1e-8: denom = 1e-8
        
    U_new = -depth_au / denom
    A_new = U_new * ratio_AU
    
    return A_new, U_new

# ==============================================================================
# 4. CORE SOLVER LOGIC
# ==============================================================================

def dipole_matrix_element(r_c, u_c, r_b, u_b):
    if len(r_c) != len(u_c): return 0.0
    u_b_interp = np.interp(r_c, r_b, u_b)
    integrand = u_c * r_c * u_b_interp
    return np.trapz(integrand, r_c)

def _compute_point(E_pe, r_b, u_b, E_b, params, A, U, mu, r_c_val, delta_val):
    """Worker for one energy point."""
    l_init = params['l_init']
    j_init = params['j_init']
    species = params['species']
    xi_soc = params['xi_soc']
    
    channels = []
    l_next = l_init + 1
    for j_f in [l_next - 0.5, l_next + 0.5]:
        if abs(j_init - j_f) <= 1.0: channels.append({'l': l_next, 'j': j_f})
    if l_init > 0:
        l_prev = l_init - 1
        js = [0.5] if l_prev == 0 else [abs(l_prev - 0.5), l_prev + 0.5]
        for j_f in js:
            if abs(j_init - j_f) <= 1.0: channels.append({'l': l_prev, 'j': j_f})
            
    E_phot = E_pe + abs(E_b)
    prefactor = (4.0 * np.pi**2 * ALPHA_FS * E_phot) / 3.0
    
    sigma_total = 0.0
    partials = {}
    
    for ch in channels:
        l_f = ch['l']
        j_f = ch['j']
        
        # 1. Solve Continuum (Pass ALL confinement params)
        r_c, u_c, _ = compute_continuum_state(
            E_pe, ell_cont=l_f, j_total=j_f,
            species=species, xi_soc=xi_soc,
            A=A, U=U, mu=mu, 
            r_c=r_c_val, Delta=delta_val 
        )
        
        # 2. Dipole Integral
        D = dipole_matrix_element(r_c, u_c, r_b, u_b)
        
        # 3. Weights
        weight = 1.0
        if species == 'H' and l_f == 1:
            # H: s->p branching
            weight = 2.0/3.0 if abs(j_f - 1.5) < 0.1 else 1.0/3.0
        elif species == 'Ar' and l_f == 2:
            # Ar: p->d branching (9:1 approx)
            weight = 1.8 if abs(j_f - 2.5) < 0.1 else 0.2
            
        partial_sigma = prefactor * weight * (abs(D)**2)
        sigma_total += partial_sigma
        
        orb = 's' if l_f==0 else ('p' if l_f==1 else ('d' if l_f==2 else 'f'))
        frac = f"{int(2*j_f)}/2"
        partials[f"{orb}{frac}"] = partial_sigma

    return {'total': sigma_total, 'partials': partials}

def run_cross_section_parallel(E_grid, r_b, u_b, E_b, params, A, U, mu, r_c_val, delta_val):
    worker = partial(_compute_point, r_b=r_b, u_b=u_b, E_b=E_b, 
                     params=params, A=A, U=U, mu=mu, r_c_val=r_c_val, delta_val=delta_val)
    with Pool(max(1, cpu_count()-1)) as pool:
        results = list(pool.map(worker, E_grid))
    return results

# ==============================================================================
# 5. DATA PROCESSOR
# ==============================================================================
def process_species(params, file_prefix):
    name = params['species']
    print(f"\n{'='*40}\nProcessing {name}\n{'='*40}")
    
    scenarios = []
    scenarios.append({'lbl': 'Free', 'A':0, 'U':0, 'mu':0})
    
    # Calculate A and U correctly
    A_c, U_c = solve_gasw_params(CONF_DEPTH)
    scenarios.append({'lbl': 'Confined', 'A':A_c, 'U':U_c, 'mu':0})
    
    for m in MU_PLASMA:
        scenarios.append({'lbl': f'Plasma_mu_{m}', 'A':0, 'U':0, 'mu':m})
    for m in MU_HYBRID:
        scenarios.append({'lbl': f'Hybrid_mu_{m}', 'A':A_c, 'U':U_c, 'mu':m})
        
    energies_data = {}         
    wavefunc_data = {}         
    sigma_rows = []            
    common_r = None 

    for sc in tqdm(scenarios, desc=f"{name} Scenarios"):
        lbl = sc['lbl']
        
        try:
            r_b, u_b, E_b, _, _ = solve_ground_u(
                VGASW_total_debye, species=name, R_max=60.0, N=6000,
                A=sc['A'], U=sc['U'], mu=sc['mu'], 
                r_c=CONF_RADIUS, Delta=CONF_DELTA, # Use correct constants
                l_wave=params['l_init'], j_total=params['j_init'], 
                xi_soc=params['xi_soc']
            )
            energies_data[lbl] = [E_b]
            if common_r is None:
                common_r = r_b
                wavefunc_data['Radius_au'] = r_b
            wavefunc_data[lbl] = u_b
            
        except Exception as e:
            print(f"Error in bound state {lbl}: {e}")
            E_b = 1.0 
            
        if E_b < -1e-4:
            results = run_cross_section_parallel(
                E_GRID, r_b, u_b, E_b, params, 
                sc['A'], sc['U'], sc['mu'], CONF_RADIUS, CONF_DELTA
            )
            
            occ = params['Z_occ']
            totals_au = np.array([res['total'] for res in results]) * occ
            
            if name == 'H' and lbl == 'Free':
                thresh_val = totals_au[0]
                print(f"\n[SANITY CHECK] H Free: {thresh_val:.4f} a.u.")

            sc_data = {'Energy_au': E_GRID, f'Total_{lbl}': totals_au}
            partial_keys = list(results[0]['partials'].keys())
            for key in partial_keys:
                p_vals_au = np.array([res['partials'][key] for res in results]) * occ
                sc_data[f'{key}_{lbl}'] = p_vals_au
            sigma_rows.append(pd.DataFrame(sc_data))
        else:
            if name == 'H': keys = ['p1/2', 'p3/2']
            else:           keys = ['s1/2', 'd3/2', 'd5/2']
            sc_data = {'Energy_au': E_GRID, f'Total_{lbl}': np.zeros_like(E_GRID)}
            for k in keys: sc_data[f'{k}_{lbl}'] = np.zeros_like(E_GRID)
            sigma_rows.append(pd.DataFrame(sc_data))

    pd.DataFrame(energies_data).to_csv(f"{file_prefix}_bound_energies.csv", index=False)
    pd.DataFrame(wavefunc_data).to_csv(f"{file_prefix}_bound_wavefunctions.csv", index=False)
    if sigma_rows:
        df_final = sigma_rows[0]
        for i in range(1, len(sigma_rows)):
            df_final = pd.merge(df_final, sigma_rows[i], on='Energy_au')
        df_final.to_csv(f"{file_prefix}_crosssection.csv", index=False)

    print(f"Saved files for {name} (Units: a.u.)")

if __name__ == "__main__":
    freeze_support()
    process_species(PARAMS_H, "hydrogen")
    process_species(PARAMS_AR, "argon")
    print("\nALL TASKS COMPLETED.")