"""
Relativistic Photoionization Cross Section Calculator.
FIXED: 
1. Typo 'diag_lists' -> 'diag_list'.
2. Branching Ratios: p->d (2/3) vs p->s (1/3). This shifts the Cooper Minimum Left.
"""
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from numba import njit
from src.continuum import compute_continuum_state

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, *args, **kwargs): return iterable

ALPHA_FS = 1.0 / 137.036

@njit(fastmath=True, cache=True)
def dipole_matrix_element(r_c, u_c, r_b, u_b):
    """Calculates <u_c | r | u_b> (Length Form)"""
    u_b_interp = np.interp(r_c, r_b, u_b)
    integrand = u_c * r_c * u_b_interp
    integral = 0.0
    for i in range(len(r_c) - 1):
        dr = r_c[i+1] - r_c[i]
        avg = 0.5 * (integrand[i] + integrand[i+1])
        integral += avg * dr
    return integral

def _compute_single_relativistic(E_pe, r_bound, u_bound, E_bound, 
                                 l_initial, j_initial, species, 
                                 xi_soc, A, U, mu, **kwargs):
    try:
        channels = []
        
        # ---------------------------------------------------------
        # 1. Determine Allowed Channels & CORRECT Weights
        # ---------------------------------------------------------
        
        # CASE A: s-shell (l=0) -> p (l=1)
        if l_initial == 0:
            # s_1/2 -> p_1/2 (1/3 of total)
            channels.append({'l': 1, 'j': 0.5, 'w': 1.0/3.0})
            # s_1/2 -> p_3/2 (2/3 of total)
            channels.append({'l': 1, 'j': 1.5, 'w': 2.0/3.0})
            
        # CASE B: p-shell (l=1) -> d (l=2) and s (l=0)
        elif l_initial == 1:
            # BRANCHING RATIO FIX:
            # l=1 -> l=2 (d-wave): Weight = 2/3 (Dominant)
            # l=1 -> l=0 (s-wave): Weight = 1/3 (Weaker)
            
            w_d = 2.0/3.0
            w_s = 1.0/3.0
            
            if abs(j_initial - 1.5) < 0.1: # Initial p_3/2
                # p_3/2 -> d (Split 1:9 into d3/2, d5/2)
                channels.append({'l': 2, 'j': 1.5, 'w': 0.1 * w_d}) 
                channels.append({'l': 2, 'j': 2.5, 'w': 0.9 * w_d})
                
                # p_3/2 -> s (Only s1/2)
                channels.append({'l': 0, 'j': 0.5, 'w': 1.0 * w_s})

            else: # Initial p_1/2
                # p_1/2 -> d (Only d3/2)
                channels.append({'l': 2, 'j': 1.5, 'w': 1.0 * w_d})
                # p_1/2 -> s (Only s1/2)
                channels.append({'l': 0, 'j': 0.5, 'w': 1.0 * w_s})

        else:
            # Fallback for d, f shells (approximate)
            l_next = l_initial + 1
            l_prev = max(0, l_initial - 1)
            channels.append({'l': l_next, 'j': l_next-0.5, 'w': 0.5})
            channels.append({'l': l_prev, 'j': l_prev+0.5, 'w': 0.5})

        # Override for specific analysis
        if 'override_channels' in kwargs and kwargs['override_channels']:
            channels = kwargs['override_channels']

        # ---------------------------------------------------------
        # 2. Compute
        # ---------------------------------------------------------
        sigma_sum = 0.0
        details = {}
        E_photon = E_pe + abs(E_bound)
        prefactor = (4.0 * np.pi**2 * ALPHA_FS * E_photon) / 3.0
        
        for ch in channels:
            l_f = ch['l']
            j_f = ch['j']
            weight = ch['w']
            
            # Solve Continuum
            r_cont, u_cont, diag = compute_continuum_state(
                E_pe, ell_cont=l_f, j_total=j_f, 
                species=species, xi_soc=xi_soc, 
                A=A, U=U, mu=mu, **kwargs
            )
            
            # Dipole Integral
            D = dipole_matrix_element(r_cont, u_cont, r_bound, u_bound)
            
            # Partial Sigma
            partial_sigma = prefactor * weight * (abs(D)**2)
            sigma_sum += partial_sigma
            
            label_j = f"{int(2*j_f)}/2"
            details[f"sigma_L{l_f}_J{label_j}"] = partial_sigma
            details[f"D_L{l_f}_J{label_j}"] = D 
            
            # Store by L for checking
            l_key = f"sigma_l{l_f}"
            details[l_key] = details.get(l_key, 0.0) + partial_sigma

        return {'success': True, 'sigma': sigma_sum, 'details': details}

    except Exception as e:
        return {'success': False, 'error': str(e), 'sigma': 0.0}

def compute_relativistic_spectrum(E_pe_array, r_bound, u_bound, E_bound, 
                                  l_initial, j_initial, species, xi_soc,
                                  A=0.0, U=0.0, mu=0.0, n_workers=None, **kwargs):
    if n_workers is None: n_workers = max(1, cpu_count() - 1)
    worker = partial(_compute_single_relativistic,
                     r_bound=r_bound, u_bound=u_bound, E_bound=E_bound,
                     l_initial=l_initial, j_initial=j_initial,
                     species=species, xi_soc=xi_soc,
                     A=A, U=U, mu=mu, **kwargs)
    
    # Serial execution for small jobs
    if n_workers <= 1 or len(E_pe_array) < 5:
        results = [worker(E) for E in tqdm(E_pe_array, desc=f"   Calc Rel-Sigma")]
    else:
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(pool.imap(worker, E_pe_array), total=len(E_pe_array), desc="   Calc Rel-Sigma"))
            
    sigma_list = [res['sigma'] if res['success'] else 0.0 for res in results]
    diag_list  = [res.get('details', {}) if res['success'] else {'error': res.get('error')} for res in results]
    
    # FIXED RETURN STATEMENT (removed 's' from diag_list)
    return np.array(sigma_list), diag_list