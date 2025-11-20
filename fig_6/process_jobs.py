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
    D_ref = 0.0512
    rho_vals = sys.argv[1].split(',')
    T_vals = sys.argv[2].split(',')
    num_jobs = int(sys.argv[3])
    NUM_ROWS = 1 # THIS IS HARD CODED ASSUMING len(rho_vals) == len(T_vals) == 2
    NUM_COLS = 2

    directories = [f'rho_{rho}_T_{T}' for rho, T in zip(rho_vals, T_vals)] # Should just be one 

    msd_equils_all = [] 
    msd_crystals_all = []
    for directory in directories:
        msd_equils = [] 
        msd_crystals = []
        for i in range(1, num_jobs+1):
            msd_equil = pd.read_csv(directory+f'/job_{i}/msd_equil_{i}.dat', comment='#', delim_whitespace=True, names=['time', 'msd'])
            msd_crystal = pd.read_csv(directory+f'/job_{i}/msd_crystal_{i}.dat', comment='#', delim_whitespace=True, names=['time', 'msd'])
            times_equil = msd_equil['time'].values * timestep
            times_crystal = msd_crystal['time'].values * timestep
            msd_equil = msd_equil['msd'].values
            msd_crystal = msd_crystal['msd'].values
            msd_equils.append(msd_equil)
            msd_crystals.append(msd_crystal)
        msd_equil_avg = np.mean(msd_equils, axis=0)
        msd_crystal_avg = np.mean(msd_crystals, axis=0)
        msd_equil_avg[0] = 0
        msd_crystal_avg[0] = 0
        msd_equils_all.append(msd_equil_avg)
        msd_crystals_all.append(msd_crystal_avg)

    times_equil -= times_equil[0]
    times_crystal -= times_crystal[0]

    # MSD = At^2 
    ballistic_crystal = 3*times_crystal**2
    ballistic_equil = 3*times_equil**2

    linear_crystal = 6*times_crystal*D_ref
    linear_equil = 6*times_equil*D_ref

    # Hard coded from a previous 100 job run ... you can recompute if needed and will likely be slightly different but should be close enough  
    t_b = 0.0014591586242133944 # These are already normalized by t_D (i.e. t_b = t_B / t_D = t_B / (1/D_ref))
    t_l = 0.009613026631644206
 
    log_times = np.log(times_equil)
    log_min = np.min(log_times[1:])
    log_max = np.max(log_times)
    points = np.linspace(log_min, log_max, 16)
    # Get log_times closest to points
    closest_points = []
    for point in points:
        closest_points.append(np.argmin(np.abs(log_times - point)))

    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(12, 4))

    # msd_equil[0] = msd_equil[1]/10
    # times_equil[0] = times_equil[1]/10
    axs[0].plot(times_equil[closest_points]*D_ref,
                msd_equil[closest_points],
                marker='o',
                markersize=8,
                markerfacecolor='white',
                markeredgecolor='black',
                markeredgewidth=1.25,
                color='black',   # line color
                linewidth=.75,
                zorder=2,
                alpha=.9,
                label='MD')
    axs[0].plot(times_equil*D_ref, ballistic_equil, linewidth=1.5, alpha=0.9, color='gold', label=r'$\frac{3T^{*}}{m} t^2$', linestyle='-.')
    axs[0].plot(times_equil*D_ref, linear_equil, label=r'$6Dt$', linewidth=1.5, alpha=0.9, color='#C6AB80', linestyle='-.')


    axs[0].axvline(x=t_b, color='black', linestyle=':', alpha=0.25)
    axs[0].axvline(x=t_l, color='black', linestyle=':', alpha=0.25)


    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'$t/t_D$', fontsize=24)
    axs[0].set_ylabel('MSD', fontsize=24)

    x_min, x_max = axs[0].get_xlim()
    axs[0].axvspan(1e-5, t_b,color='gold', alpha=0.12)
    axs[0].axvspan(t_l, x_max, color='#38220F', alpha=0.12)
    axs[0].set_xlim(1e-5, x_max)
    axs[0].set_ylim(1e-5, None)




    axs[1].plot(times_crystal[closest_points]*D_ref,
                msd_crystal[closest_points],
                marker='o',
                markersize=8,
                markerfacecolor='white',
                markeredgecolor='black',
                markeredgewidth=1.25,
                color='black',   # line color
                linewidth=.75,
                zorder=2,
                alpha=.9,
                label='MD')


    axs[1].plot(times_crystal*D_ref, ballistic_crystal, label='Ballistic', linewidth=1.5, alpha=0.9, color='gold', linestyle='-.')
    axs[1].plot(times_crystal*D_ref, linear_crystal, label='Linear', linewidth=1.5, alpha=0.9, color='#C6AB80', linestyle='-.')
    axs[1].axvline(x=t_b, color='black', linestyle=':', alpha=0.25)
    axs[1].axvline(x=t_l, color='black', linestyle=':', alpha=0.25)
    # Color the region in between the two lines 



    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$t/t_D$', fontsize=24)

    _, x_max = axs[1].get_xlim()
    axs[1].axvspan(1e-5, t_b,color='gold', alpha=0.12)
    axs[1].axvspan(t_l, x_max, color='#38220F', alpha=0.12)
    axs[1].set_xlim(1e-5, x_max)
    axs[1].set_ylim(1e-5, None)



    # Make axis font bigger
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)


    axs[0].set_title('(a) Equilibrium', fontsize=24, pad=10, loc='left')
    axs[1].set_title('(b) Out of Equilibrium', fontsize=24, pad=10, loc='left')

    # Place legend markers outside plot in middle 
    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=20, frameon=False, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()

    axs[0].text(0.00004, 25, r'$t<t_B$', fontsize=20, ha='center', va='center', color='black')
    axs[0].text(0.02, 25, r'$t>t_L$', fontsize=20, ha='center', va='center', color='black')
    axs[1].text(0.00004, 25, r'$t<t_B$', fontsize=20, ha='center', va='center', color='black')
    axs[1].text(0.02, 25, r'$t>t_L$', fontsize=20, ha='center', va='center', color='black')

    # Save fig as svg if editing (needed for paper)
    # plt.savefig('fig_6.svg', dpi=300, bbox_inches='tight')

    plt.savefig('fig_6.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()

