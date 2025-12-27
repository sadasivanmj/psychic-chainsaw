"""
normalization.py
SPARSE EXACT VERSION: 
Uses mpmath for exact accuracy, but downsamples the fitting grid 
to drastically increase speed (from ~50s -> ~0.5s per point).
"""

import numpy as np
import mpmath

# ==============================================================================
# 1. COULOMB FUNCTIONS (Sparse Wrapper)
# ==============================================================================
def get_coulomb_fg_sparse(ell, eta, rho_array, max_points=40):
    """
    Computes Exact Regular (F) and Irregular (G) Coulomb functions using mpmath.
    OPTIMIZATION: Only computes for a subset of points to save time.
    
    Returns:
        F_sub, G_sub: Values at the sampled points.
        indices: The indices of rho_array that were used.
    """
    n_total = len(rho_array)
    
    # Determine indices to sample (evenly spaced)
    if n_total > max_points:
        indices = np.linspace(0, n_total - 1, max_points, dtype=int)
    else:
        indices = np.arange(n_total, dtype=int)
        
    rho_sub = rho_array[indices]
    
    F_list = []
    G_list = []
    
    # mpmath configuration (Double Precision)
    mpmath.mp.dps = 15
    
    for rho in rho_sub:
        # mpmath calculation (The expensive part)
        f_val = mpmath.coulombf(ell, eta, rho)
        g_val = mpmath.coulombg(ell, eta, rho)
        F_list.append(float(f_val))
        G_list.append(float(g_val))
        
    return np.array(F_list), np.array(G_list), indices

# ==============================================================================
# 2. MATCHING SOLVERS (Least Squares on Sparse Grid)
# ==============================================================================

def fit_to_coulomb(r_fit, u_fit, k, ell, Z=1.0):
    """
    Fits u(r) ~ A * F(kr) + B * G(kr) using Sparse Exact Sampling.
    """
    eta = -Z / k
    rho = k * r_fit
    
    # 1. Get Exact F, G on a sparse subset
    F_sub, G_sub, indices = get_coulomb_fg_sparse(ell, eta, rho, max_points=40)
    
    # 2. Extract corresponding u values
    u_sub = u_fit[indices]
    
    # 3. Least Squares Fit on the subset
    # Design Matrix M = [F, G]
    M = np.column_stack((F_sub, G_sub))
    
    # Solve M * [a, b]^T = u_sub
    coeffs, _, _, _ = np.linalg.lstsq(M, u_sub, rcond=None)
    a, b = coeffs
    
    # 4. Amplitude & Normalization
    calc_amp = np.sqrt(a**2 + b**2)
    if calc_amp < 1e-20: calc_amp = 1.0
    
    target_amp = np.sqrt(2.0 / (np.pi * k))
    norm_factor = target_amp / calc_amp
    
    return norm_factor, {'a': a, 'b': b, 'method': 'coulomb_sparse_exact'}

def fit_to_planewave(r_fit, u_fit, k, ell):
    """
    Fits u(r) ~ A * sin(...) + B * cos(...)
    Plane waves are cheap, so we can use all points or sparse. 
    Using sparse for consistency.
    """
    n_total = len(r_fit)
    max_points = 50
    if n_total > max_points:
        indices = np.linspace(0, n_total - 1, max_points, dtype=int)
    else:
        indices = np.arange(n_total, dtype=int)
        
    r_sub = r_fit[indices]
    u_sub = u_fit[indices]
    
    phase_l = ell * np.pi / 2.0
    arg = k * r_sub - phase_l
    
    S = np.sin(arg)
    C = np.cos(arg)
    
    M = np.column_stack((S, C))
    coeffs, _, _, _ = np.linalg.lstsq(M, u_sub, rcond=None)
    a_s, b_c = coeffs
    
    calc_amp = np.sqrt(a_s**2 + b_c**2)
    if calc_amp < 1e-20: calc_amp = 1.0
    
    target_amp = np.sqrt(2.0 / (np.pi * k))
    norm_factor = target_amp / calc_amp
    
    return norm_factor, {'a': a_s, 'b': b_c, 'method': 'plane_wave'}

# ==============================================================================
# 3. MASTER NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_continuum_coulomb_free(r, u_raw, E, ell, Z=1.0):
    return energy_normalize_continuum(r, u_raw, E, ell, A=0, U=0, mu=0)

def energy_normalize_continuum(r, u_raw, E, ell, A, U, mu):
    k = np.sqrt(2.0 * E)
    
    # Define Fitting Window (Last 15% of grid)
    N = len(r)
    start = int(0.85 * N)
    end = int(0.98 * N)
    
    r_fit = r[start:end]
    u_fit = u_raw[start:end]
    
    # --- DISPATCH LOGIC ---
    if abs(mu) < 1e-9:
        # Neutral Atom (Coulomb Tail) - Uses Sparse mpmath
        norm_factor, info = fit_to_coulomb(r_fit, u_fit, k, ell, Z=1.0)
    else:
        # Screened/Short Range (Plane Wave Tail)
        norm_factor, info = fit_to_planewave(r_fit, u_fit, k, ell)
        
    u_norm = u_raw * norm_factor
    info['k'] = k
    info['norm_factor'] = norm_factor
    
    return u_norm, info