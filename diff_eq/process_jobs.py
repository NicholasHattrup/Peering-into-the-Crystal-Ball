import numpy as np
import os, sys
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
plt.style.use(['science', 'no-latex'])

def main():
    print('Processing jobs...')
    timestep = 0.002
    rho_vals = sys.argv[1].split(',')
    T_vals = sys.argv[2].split(',')
    num_jobs = int(sys.argv[3])
    NUM_ROWS = 1 # THIS IS HARD CODED ASSUMING len(rho_vals) == len(T_vals) == 1
    NUM_COLS = 1

    directories = [f'rho_{rho}_T_{T}' for rho, T in zip(rho_vals, T_vals)]

    Ds_all = []
    msds_all = []


    for directory in directories:
        msds = [] 
        Ds = []
        for i in range(1, num_jobs + 1):
            msd_df = pd.read_csv(directory + f'/job_{i}/msd_{i}.dat', comment='#', delim_whitespace=True, names=['t', 'msd'])
            t = msd_df['t'].values * timestep
            t -= t[0]
            msd = msd_df['msd'].values
            msd[0] = 0
            D = (msd @ t / (t @ t)) / 6 
            msds.append(msd)
            Ds.append(D)
        D_avg = np.array(Ds).mean()
        Ds_all.append(D_avg)
        msds_all.append(msds)
    
    # Again assuming only one plot 
    Ds = np.array(Ds)
    indices = np.arange(len(Ds))
    sample_size = np.min((128, len(Ds)))
    repeats = 16384
    choices = np.random.choice(indices, size=(repeats, sample_size), replace=True)
    D_avgs = Ds[choices].mean(axis=1) 
    D_mean = D_avgs.mean()
    D_std = D_avgs.std()
    
    fig, axs = plt.subplots(NUM_ROWS, 2*NUM_COLS, figsize=(14, 6))
    axs = axs.flatten()
    for msd in msds_all[0]:
        axs[0].plot(t, msd, color='black', alpha=0.1)
    axs[0].plot(t, np.mean(msds_all[0], axis=0), color='red', linewidth=2, linestyle='-.')

    axs[1].hist(D_avgs, bins=30, density=True, alpha=0.25, color='steelblue', edgecolor='black')
    axs[1].text(0.05, 0.95, rf'D = {D_mean:.4f} $\pm$ {D_std:.4f} ($\sigma ^2 / \tau$)', transform=axs[1].transAxes, verticalalignment='top', fontsize=24)

    axs[1].set_ylim(None, axs[1].get_ylim()[1] * 1.25)

    for ax in axs:
        ax.tick_params(direction='in', which='both', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', labelsize=24)

    axs[0].set_xlabel(rf'Time ($\tau$)', fontsize=32)
    axs[0].set_ylabel(rf'MSD ($\sigma^2$)', fontsize=32)

    axs[1].set_xlabel(rf'Diffusion ($\sigma^2 / \tau$)', fontsize=32)
    axs[1].set_ylabel('Probability Density', fontsize=32)

    plt.tight_layout()
    plt.show()
    plt.savefig('diff_eq.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()

