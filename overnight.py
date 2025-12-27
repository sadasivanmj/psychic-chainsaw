"""
OVERNIGHT PRODUCTION: DASHBOARD EDITION
---------------------------------------
A verbose, monitoring-focused production engine.
Features:
  - Real-time Bound State Physics reporting (Energy + Nodes)
  - Live Cooper Minimum detection and Shift calculation
  - High-precision Progress Bars with ETA
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import freeze_support, Pool, cpu_count
from functools import partial

# Force Matplotlib backend for headless running
plt.switch_backend('Agg')

# --- 1. PRETTY PRINTING & PROGRESS BARS ---
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("NOTE: 'tqdm' library not found. Falling back to simple logging.")
    # Fallback dummy class
    class tqdm:
        def __init__(self, iterable, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.desc = desc
            print(f"Started: {desc}")
        def __iter__(self):
            for i, item in enumerate(self.iterable):
                if i % 500 == 0: print(f"  ... processed {i} points")
                yield item

def log_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def log_case_header(depth, mu, index, total):
    print(f"\n" + "-"*80)
    print(f" >> CASE {index}/{total}: Depth = {depth:.2f} a.u. | Mu = {mu:.2f} a.u.")
    print("-"*80)

def log_success(msg):
    print(f"    [SUCCESS] {msg}")

def log_info(msg):
    print(f"    [INFO]    {msg}")

def log_fail(msg):
    print(f"    [FAILURE] {msg}")

# ==============================================================================
# 2. PATHS & IMPORTS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from bound import solve_ground_u
    from cross_section import compute_relativistic_spectrum
    from potential import VGASW_total_debye
except ImportError as e:
    log_fail(f"Could not import physics modules: {e}")
    sys.exit(1)

# ==============================================================================
# 3. SIMULATION MATRIX
# ==============================================================================
# Adaptive Grid: Dense in Cooper Zone (0.5 - 1.8 a.u.)
E_GRID = np.concatenate([
    np.linspace(0.01, 0.5, 200),     # Threshold / Resonances
    np.linspace(0.501, 1.8, 1500),   # COOPER ZONE (High Res)
    np.linspace(1.81, 4.0, 300)      # High Energy Tail
])

DEPTH_LIST = [0.0, 0.30, 0.46, 0.56, 1.03]
MU_VALUES = [0.0, 0.1, 0.25, 0.5]

PARAMS_AR = {
    'species': 'Ar', 'Z_occ': 6.0, 
    'l_init': 1, 'j_init': 1.5, 'xi_soc': 2.6121e-05,
    'r_c': 6.7, 'Delta': 2.8
}

# ==============================================================================
# 4. PHYSICS HELPERS
# ==============================================================================
def solve_gasw_params(depth_au):
    """Saha Ratio Constraint Solver"""
    if depth_au == 0.0: return 0.0, 0.0
    R_CONST = -24.5
    SIGMA = 1.70
    ratio_AU = R_CONST / (np.sqrt(2 * np.pi) * SIGMA)
    denom = ratio_AU - 1.0
    if abs(denom) < 1e-8: denom = 1e-8
    U_new = -depth_au / denom
    A_new = U_new * ratio_AU
    return A_new, U_new

# ==============================================================================
# 5. PARALLEL WORKER & SCANNER
# ==============================================================================
def _compute_point(E_pe, r_b, u_b, E_b, params, A, U, mu):
    # Compute single energy point
    sigma_au_list, details = compute_relativistic_spectrum(
        [E_pe], r_b, u_b, E_b,
        l_initial=params['l_init'], j_initial=params['j_init'],
        species=params['species'], xi_soc=params['xi_soc'],
        A=A, U=U, mu=mu,
        r_c=params['r_c'], Delta=params['Delta']
    )
    
    sigma_total = sigma_au_list[0]
    
    # Extract Dominant Dipole (for diagnostics)
    d_sq_sum = 0.0
    if details and len(details) > 0:
        det = details[0]
        for key, val in det.items():
            if key.startswith('D_'):
                d_sq_sum += abs(val)**2

    return {'sigma': sigma_total, 'dipole_sq': d_sq_sum}

def run_parallel_scan(E_grid, r_b, u_b, E_b, params, A, U, mu):
    worker = partial(_compute_point, r_b=r_b, u_b=u_b, E_b=E_b, 
                     params=params, A=A, U=U, mu=mu)
    
    n_cores = max(1, int(cpu_count() * 0.9)) # Use 90% CPU
    
    # Nice progress bar description
    desc_str = f"    -> Scanning Spectrum ({len(E_grid)} pts)"
    
    with Pool(n_cores) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(pool.imap(worker, E_grid), total=len(E_grid), 
                                desc=desc_str, unit="pt", ncols=100, colour='green'))
        else:
            print(desc_str)
            results = list(pool.imap(worker, E_grid))
        
    return results

# ==============================================================================
# 6. MAIN PRODUCTION CONTROLLER
# ==============================================================================
def run_argon_production():
    params = PARAMS_AR
    name = params['species']
    
    # Setup Output
    base_folder = os.path.join("Results_Production_v3", name)
    if not os.path.exists(base_folder): os.makedirs(base_folder)
    
    report_path = os.path.join(base_folder, "MINIMA_REPORT.txt")
    
    # Initialize Report File
    with open(report_path, "w") as f:
        f.write(f"COOPER MINIMUM DASHBOARD: {name}\n")
        f.write("="*75 + "\n")
        f.write(f"{'Depth':<8} | {'Mu':<6} | {'Bound E':<10} | {'Min Loc':<10} | {'Shift':<10} | {'Status':<10}\n")
        f.write("-" * 75 + "\n")
    
    log_section(f"STARTING PRODUCTION RUN: {name}")
    print(f"Total Cases: {len(DEPTH_LIST) * len(MU_VALUES)}")
    print(f"Total Energy Points per Case: {len(E_GRID)}")
    print(f"Output Directory: {base_folder}")

    # Track Baseline for Shift Calculation
    baseline_min = None
    
    case_counter = 0
    total_cases = len(DEPTH_LIST) * len(MU_VALUES)

    for depth in DEPTH_LIST:
        A_c, U_c = solve_gasw_params(depth)
        
        for mu in MU_VALUES:
            case_counter += 1
            log_case_header(depth, mu, case_counter, total_cases)
            
            # --- PHASE 1: BOUND STATE ---
            log_info("Solving Bound State (3p)...")
            try:
                # WIDE SEARCH WINDOW to ensure we find deep confined states
                r_b, u_b, E_b, n_nodes, _ = solve_ground_u(
                    VGASW_total_debye, species=name, 
                    A=A_c, U=U_c, mu=mu,
                    r_c=params['r_c'], Delta=params['Delta'],
                    l_wave=params['l_init'], j_total=params['j_init'], 
                    xi_soc=params['xi_soc'],
                    R_max=60.0, N=6000,
                    E_min=-3.0, E_max=-0.1 
                )
                log_success(f"Locked 3p State: E = {E_b:.5f} a.u. | Nodes: {n_nodes}")
                
            except Exception as e:
                log_fail(f"Bound State Solver Crashed: {e}")
                continue
            
            # Sanity Check
            if E_b > -0.1:
                log_fail(f"State too shallow (E={E_b:.4f}). Likely finding excited state. Skipping.")
                continue

            # --- PHASE 2: CONTINUUM SCAN ---
            results = run_parallel_scan(E_GRID, r_b, u_b, E_b, params, A_c, U_c, mu)
            
            # --- PHASE 3: PROCESS & SAVE ---
            sigmas = np.array([r['sigma'] for r in results]) * params['Z_occ']
            dipoles = np.array([r['dipole_sq'] for r in results])
            
            df = pd.DataFrame({
                'Energy_au': E_GRID,
                'Sigma_Total_au': sigmas,
                'Sigma_Mb': sigmas * 28.0028,
                'Dipole_Sq_au': dipoles
            })
            
            save_name = f"{name}_Depth_{depth:.2f}_Mu_{mu:.2f}.csv"
            df.to_csv(os.path.join(base_folder, save_name), index=False)
            
            # --- PHASE 4: LIVE DIAGNOSTICS ---
            # Find Minima
            min_idx = np.argmin(sigmas)
            min_loc = E_GRID[min_idx]
            min_val = sigmas[min_idx] * 28.0028
            
            # Calculate Shift
            shift_str = "-"
            if depth == 0.0 and mu == 0.0:
                baseline_min = min_loc
                shift_str = "Baseline"
            elif baseline_min is not None:
                shift = min_loc - baseline_min
                # Formatting: + for Blue Shift, - for Red Shift
                shift_str = f"{shift:+.4f}"
            
            print(f"    -> DIAGNOSTIC: Min found at {min_loc:.4f} a.u. ({min_val:.2f} Mb)")
            print(f"    -> SHIFT STATUS: {shift_str}")

            # Append to Report
            with open(report_path, "a") as f:
                f.write(f"{depth:<8.2f} | {mu:<6.2f} | {E_b:<10.4f} | {min_loc:<10.4f} | {shift_str:<10} | {'DONE':<10}\n")

if __name__ == "__main__":
    freeze_support()
    t0 = time.time()
    
    run_argon_production()
    
    elapsed = (time.time() - t0) / 3600.0
    log_section(f"ALL TASKS COMPLETED. Total Time: {elapsed:.2f} hours.")