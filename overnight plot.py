import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# ==============================================================================
# 0. PUBLICATION STYLE CONFIGURATION
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',      # LaTeX-like math font
    'axes.linewidth': 1.0,           # Thicker borders
    'axes.labelsize': 16,            # Readable labels
    'xtick.direction': 'in',         # Physics standard
    'ytick.direction': 'in',
    'xtick.top': True,               # Ticks on all sides
    'ytick.right': True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': False,         # Clean legends
    'figure.dpi': 300,               # Print resolution
    'lines.linewidth': 2.0           # Bold lines
})

# ==============================================================================
# 1. SETUP
# ==============================================================================
DATA_DIR = "Results_Production_v3/Ar"
OUTPUT_DIR = "Ar_Analysis_Plots_Pub"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Parameters used in simulation
DEPTHS = [0.0, 0.30, 0.46, 0.56, 1.03]
MUS = [0.0, 0.1, 0.25, 0.5]

def load_data(depth, mu):
    """Safe loader for Argon CSVs."""
    filename = f"Ar_Depth_{depth:.2f}_Mu_{mu:.2f}.csv"
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        # Fallback for filenames that might not have double decimals if generated differently
        # But overnight_v3 uses .2f so it should be fine.
        print(f"Warning: Missing file {filename}")
        return None
    return pd.read_csv(path)

# ==============================================================================
# PLOT 1: BASELINE VERIFICATION (Free Ar Cooper Minimum)
# ==============================================================================
def plot_baseline_check():
    print("Generating Figure 1: Free Argon Baseline (Cooper Minimum)...")
    df = load_data(0.0, 0.0)
    if df is None: return

    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot Cross Section in Mb
    ax.plot(df['Energy_au'], df['Sigma_Mb'], 'k-', label=r'Calculated $\sigma_{3p}$')
    
    # Find Minimum
    min_idx = df['Sigma_Mb'].idxmin()
    min_energy = df['Energy_au'].iloc[min_idx]
    min_val = df['Sigma_Mb'].iloc[min_idx]
    
    # Annotation
    textstr = '\n'.join((
        r'$\bf{Cooper\ Minimum}$',
        r'$E_{min} \approx %.3f$ a.u.' % (min_energy, ),
        r'$\sigma_{min} \approx %.3f$ Mb' % (min_val, )
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.55, 0.65, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Mark the spot
    ax.plot(min_energy, min_val, 'ro', markersize=6)

    ax.set_xlabel(r"Photoelectron Energy $\epsilon$ (a.u.)")
    ax.set_ylabel(r"Cross Section $\sigma$ (Mb)")
    ax.set_title(r"Fig 1. Free Argon Photoionization (Cooper Minimum)")
    
    # Zoom for clarity around the minimum
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(0.0,0.35*28)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Argon_Baseline.png")
    plt.close()

# ==============================================================================
# PLOT 2: PURE PLASMA EFFECTS (Debye Only)
# ==============================================================================
def plot_debye_only():
    print("Generating Figure 2: Plasma Screening on Cooper Minimum...")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = plt.cm.magma(np.linspace(0.2, 0.85, len(MUS)))
    
    for mu, col in zip(MUS, colors):
        df = load_data(0.0, mu)
        if df is None: continue
        
        label_str = r'Free' if mu == 0.0 else r'$\mu = %.2f$' % mu
        ax.plot(df['Energy_au'], df['Sigma_Mb'], color=col, label=label_str)

    ax.set_xlabel(r"Photoelectron Energy $\epsilon$ (a.u.)")
    ax.set_ylabel(r"Cross Section $\sigma$ (Mb)")
    ax.set_title(r"Fig 2. Debye Plasma Effects on Argon CM ($D=0$)")
    
    # Zoom to show if Min disappears or shifts
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, 35.0) 
    
    ax.legend(title=r"Screening Parameter")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Argon_Debye.png")
    plt.close()

# ==============================================================================
# PLOT 3: PURE CONFINEMENT (GASW Only - The Shift)
# ==============================================================================
def plot_confinement_only():
    print("Generating Figure 3: Confinement Induced Shift...")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = ['black', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3'] # Set 1
    
    for depth, col in zip(DEPTHS, colors):
        df = load_data(depth, 0.0)
        if df is None: continue
        
        lbl = r'Free' if depth == 0.0 else r'$D = %.2f$' % depth
        ax.plot(df['Energy_au'], df['Sigma_Mb'], color=col, label=lbl, alpha=0.9)

    ax.set_xlabel(r"Photoelectron Energy $\epsilon$ (a.u.)")
    ax.set_ylabel(r"Cross Section $\sigma$ (Mb)")
    ax.set_title(r"Fig 3. Confinement Induced Cooper Minimum Shift")
    
    # Zoom specifically on the Cooper Minimum region
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, 30.0)
    
    ax.legend(title=r"Well Depth (a.u.)", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Argon_Confinement.png")
    plt.close()

# ==============================================================================
# PLOT 4: COMBINED MATRIX
# ==============================================================================
def plot_combined_matrix():
    print("Generating Figure 4: Combined Matrix (Ar)...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    colors = plt.cm.magma(np.linspace(0.2, 0.85, len(MUS)))
    
    for i, depth in enumerate(DEPTHS):
        ax = axes[i]
        
        # Title Box
        ax.text(0.95, 0.95, r'$D = %.2f$ a.u.' % depth, 
                transform=ax.transAxes, ha='right', va='top',
                fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        for mu, col in zip(MUS, colors):
            df = load_data(depth, mu)
            if df is None: continue
            
            lbl = r'$\mu=%.2f$' % mu if i == 0 else "_nolegend_"
            ax.plot(df['Energy_au'], df['Sigma_Mb'], color=col, linewidth=1.5, label=lbl)
            
        # Optional: Add arrows pointing to CM location if feasible
    
    # Global labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r"Photoelectron Energy $\epsilon$ (a.u.)", fontsize=18, labelpad=15)
    plt.ylabel(r"Cross Section $\sigma$ (Mb)", fontsize=18, labelpad=20)
    plt.title(r"Fig 4. Argon Cooper Minimum Evolution under Plasma & Confinement", fontsize=16, y=1.02)
    
    axes[5].axis('off')
    
    axes[0].legend(loc='upper right', title=r"Screening $\mu$")
    
    # Axis limits for clear CM visibility
    plt.xlim(0.0, 3.0)
    plt.ylim(0.0, 40.0) # Adjust based on data if needed
    
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Argon_Matrix.png")
    plt.close()

if __name__ == "__main__":
    plot_baseline_check()
    plot_debye_only()
    plot_confinement_only()
    plot_combined_matrix()
    print(f"\nAll Publication-Ready Argon figures saved to: {OUTPUT_DIR}")