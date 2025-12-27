"""
Bound state solver using finite difference method.
UPDATED: Accepts 'target_n_idx' to override defaults from the running program.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid
from src.potential import SAE_PARAMS

try:
    from src.potential import solve_gasw_parameters
except ImportError:
    pass

def solve_ground_u(V_func, species='H', R_max=60.0, N=12000, 
                   l_wave=None, j_total=None, xi_soc=0.0, 
                   target_n_idx=None,  # <--- NEW ARGUMENT
                   **Vkwargs):
    """
    Solves for the bound state wavefunction u(r).
    
    Parameters:
    -----------
    target_n_idx : int (optional)
        Manually select the nth eigenstate (0 = ground, 1 = first excited, etc.)
        Overrides the default found in SAE_PARAMS.
    """
    
    # 1. Load Defaults from Database
    default_ell = 0
    state_idx = 0
    
    if species in SAE_PARAMS:
        elem_data = SAE_PARAMS[species]
        params = elem_data.get('default', list(elem_data.values())[0])
        default_ell = params.ground_l
        state_idx = params.ground_n_idx
    
    # 2. Apply Overrides from Running Program
    if l_wave is not None:
        ell = l_wave
    else:
        ell = default_ell
        
    if target_n_idx is not None:
        state_idx = target_n_idx  # <--- OVERRIDE HERE
        
    # 3. Setup Grid
    # Use the fine grid (1e-8) to match your successful snippet
    r = np.linspace(1e-8, R_max, N)
    dr = r[1] - r[0]
    
    # 4. Potential Calculation
    V_total = V_func(r, species=species, 
                     l_wave=ell, j_total=j_total, xi_soc=xi_soc, 
                     **Vkwargs)
    
    # 5. Hamiltonian Construction
    r_safe = np.clip(r, 1e-12, None)
    V_eff = V_total + ell * (ell + 1) / (2.0 * r_safe**2)
    
    # Finite Difference Matrix
    N_inner = len(r) - 2
    k = 1.0 / (2.0 * dr**2)
    
    d = 2.0 * k + V_eff[1:-1]
    e = -k * np.ones(N_inner - 1)
    
    # 6. Diagonalization
    # We solve for enough eigenvalues to reach the desired index
    try:
        E_all, U_all = eigh_tridiagonal(d, e, select='i', select_range=(0, state_idx + 1))
    except Exception as e:
        print(f"Solver Error: {e}")
        return r, np.zeros_like(r), 0.0, V_eff, 0.0
    
    # Check if we found enough states
    if len(E_all) <= state_idx:
        print(f"Warning: Requested state index {state_idx}, but only found {len(E_all)} bound states.")
        idx = len(E_all) - 1
    else:
        idx = state_idx
        
    E0 = E_all[idx]
    u_int = U_all[:, idx]
    
    # 7. Post-Processing
    max_idx = np.argmax(np.abs(u_int))
    if u_int[max_idx] < 0:
        u_int = -u_int
        
    nrm = np.sqrt(trapezoid(u_int**2, r[1:-1]))
    if nrm < 1e-15: nrm = 1.0
    u_int /= nrm
    
    u_full = np.zeros_like(r)
    u_full[1:-1] = u_int
    
    norm_check = trapezoid(u_full**2, r)
    return r, u_full, E0, V_eff, norm_check