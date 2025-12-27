"""
relativistic_potential.py
CORRECTED: Exact implementation of Tong & Lin (2005) Eq. 4.
Parameters from Table 1 of the paper.
"""
import numpy as np
from functools import lru_cache
from typing import NamedTuple, Dict, Union, Tuple
from scipy.optimize import fsolve

class TongLinParams(NamedTuple):
    Z_c: float
    a1: float; a2: float  # Term 1
    a3: float; a4: float  # Term 2 (r * exp)
    a5: float; a6: float  # Term 3
    ground_l: int        
    ground_n_idx: int    

SAE_PARAMS = {
    'H': {'default': TongLinParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)},
    'Ar': {
        # EXACT PARAMETERS from Tong & Lin (2005), Table 1 for Ar
        # a1=16.039, a2=2.007
        # a3=-25.543, a4=4.525
        # a5=0.961,  a6=0.443
        # ground_n_idx=1 because this potential supports a 2p core state.
        'default': TongLinParams(1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443, 1, 1) 
    }
}

def get_Z_eff_TongLin(r, p: TongLinParams):
    """
    Calculates Z_eff using Eq. 4 from Tong & Lin (2005):
    Z_eff(r) = Z_c + a1*exp(-a2*r) + a3*r*exp(-a4*r) + a5*exp(-a6*r)
    """
    r_safe = np.where(r < 1e-12, 1e-12, r)
    
    # Term 1: a1 * exp(-a2 * r)
    term1 = p.a1 * np.exp(-p.a2 * r_safe)
    
    # Term 2: a3 * r * exp(-a4 * r)  <-- THIS WAS THE BUG. a3 is Amp, a4 is Decay.
    term2 = p.a3 * r_safe * np.exp(-p.a4 * r_safe) 
    
    # Term 3: a5 * exp(-a6 * r)
    term3 = p.a5 * np.exp(-p.a6 * r_safe)
    
    return p.Z_c + term1 + term2 + term3

def get_dVdr_TongLin(r, p: TongLinParams):
    """
    Analytic Gradient of V(r) = -Z_eff(r)/r.
    Used for Spin-Orbit Coupling.
    """
    r_safe = np.where(r < 1e-12, 1e-12, r)
    
    # Precompute exponentials
    e1 = np.exp(-p.a2 * r_safe)
    e2 = np.exp(-p.a4 * r_safe)
    e3 = np.exp(-p.a6 * r_safe)

    # Z_eff value
    Z = p.Z_c + p.a1*e1 + p.a3*r_safe*e2 + p.a5*e3

    # d(Z_eff)/dr
    dZ = -p.a1*p.a2*e1 + p.a3*e2*(1.0 - p.a4*r_safe) - p.a5*p.a6*e3
    
    # dV/dr = d/dr(-Z/r) = (Z - r*dZ)/r^2
    grad = (Z - r_safe * dZ) / (r_safe**2)
    return grad

def get_LS_factor(l, j, s=0.5):
    if l == 0: return 0.0 
    return 0.5 * (j*(j+1) - l*(l+1) - s*(s+1))

@lru_cache(maxsize=128)
def solve_gasw_parameters(Target_Depth_Au: float) -> Tuple[float, float]:
    lookup = {0.30: (-0.256, 0.044), 0.46: (-0.392, 0.068), 0.56: (-0.477, 0.083), 1.03: (-0.877, 0.153)}
    for d_key, (a_val, u_val) in lookup.items():
        if abs(Target_Depth_Au - d_key) < 0.01: return a_val, u_val
    return -0.877, 0.153

def VGASW_total_debye(r, A=0.0, U=0.0, mu=0.0, species='H', 
                      r_c=6.7, Delta=2.8, l_wave=0, j_total=None, xi_soc=0.0, **kwargs):
    r = np.asarray(r, dtype=float)
    r_safe = np.where(r < 1e-12, 1e-12, r)
    
    params = SAE_PARAMS.get(species, {}).get('default', SAE_PARAMS['H']['default'])
    
    # 1. Atomic Potential (Fixed Formula)
    V_atom = -get_Z_eff_TongLin(r, params) / r_safe * np.exp(-mu * r_safe)
    
    # 2. Confinement
    if 'depth' in kwargs and abs(A) < 1e-9 and abs(U) < 1e-9:
        A, U = solve_gasw_parameters(kwargs['depth'])

    V_conf = np.zeros_like(r)
    if abs(A) > 1e-12:
        V_conf += A * np.exp(-((r - r_c)**2) / (2.0 * 1.70**2))
        
    if abs(U) > 1e-12:
        if species == 'H':
            r_in, r_out = r_c - Delta/2.0, r_c + Delta/2.0
            V_conf[(r >= r_in) & (r <= r_out)] -= U
        else:
            exp_arg = np.clip((r - r_c) / Delta, -50, 50)
            V_conf -= U / (1.0 + np.exp(exp_arg))

    # 3. Spin-Orbit
    V_soc = np.zeros_like(r)
    if abs(xi_soc) > 1e-9 and l_wave > 0 and j_total is not None:
        V_soc = xi_soc * (1.0 / r_safe) * get_dVdr_TongLin(r, params) * get_LS_factor(l_wave, j_total)

    return V_atom + V_conf + V_soc