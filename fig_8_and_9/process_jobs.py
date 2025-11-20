import numpy as np
import os, sys
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
plt.style.use(['science', 'no-latex'])

def parseRDF(file):
    with open(file) as f:
        f.readline()
        f.readline()
        f.readline()
        _, bins = f.readline().split()
        bins = int(bins)
        length = len(f.readlines()) + 1
    samples = length // (bins+1)
    times = np.empty(samples)
    distances = np.empty(bins)
    rdf = np.empty((samples, bins))
    with open(file) as f:
        data = f.readlines()
        offSet = 3
        idx = 0
        for i in range(offSet, length, bins+1):
            times[idx] = float(data[i].split()[0])
            for j in range(i+1, i+bins+1):
                rdf[idx, j-i-1] = float(data[j].split()[-1]) # Normalized against density
            idx += 1
        for i in range(offSet+1, offSet+bins+1):
            distances[i-offSet-1] = float(data[i].split()[1]) # Actual values
    return times, distances, rdf



def main():
    print('Processing jobs...')
    timestep = 0.002
    rho_vals = sys.argv[1].split(',')
    T_vals = sys.argv[2].split(',')
    num_jobs = int(sys.argv[3])
    NUM_ROWS = 3 # THIS IS HARD CODED ASSUMING len(rho_vals) == len(T_vals) == 9
    NUM_COLS = 3

    directories = [f'rho_{rho}_T_{T}' for rho, T in zip(rho_vals, T_vals)]

    diffusions = []
    msds_all = []
    rdfs_all = []
    rdf_avg_all = []

    for directory in directories:
        msds = [] 
        for i in range(1, num_jobs + 1):
            msd_df = pd.read_csv(directory + f'/job_{i}/msd_{i}.dat', comment='#', delim_whitespace=True, names=['t', 'msd'])
            t = msd_df['t'].values * timestep
            msd = msd_df['msd'].values
            msds.append(msd)

        msd_avg = np.array(msds).mean(axis=0)
        t -= t[0]
        msd_avg[0] = 0 

        D = (msd_avg @ t / (t @ t)) / 6
        diffusions.append(D)
        msds_all.append(msd_avg)


    msds_all = np.array(msds_all)

    log_times = np.log(t)
    log_min = np.min(log_times[1:])
    log_max = np.max(log_times)
    points = np.linspace(log_min, log_max, 16)
    # Get log_times closest to points
    closest_points = []
    for point in points:
        closest_points.append(np.argmin(np.abs(log_times - point)))

    for directory in directories:
        rdfs = []
        for i in range(1, num_jobs + 1):
            _, dists, rdf = parseRDF(directory + f'/job_{i}/rdf_{i}.dat')
            rdfs.append(rdf)
        rdf_avg = np.mean(rdfs, axis=(0, 1))
        rdfs_all.append(rdfs)
        rdf_avg_all.append(rdf_avg)

    rdfs_all =  np.array(rdfs_all)
    # RDF figure 
    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(18, 6))


    y_min_all = np.inf
    y_max_all = -np.inf

    sigma_peaks = []
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            idx = 3*i+j
            for rdfs in rdfs_all[idx]:
                for rdf in rdfs:
                    axs[i, j].plot(dists, rdf, color='gainsboro', alpha=0.05)
            axs[i, j].plot(dists, rdf_avg_all[idx], color='cornflowerblue', alpha=1, linewidth=2.5, linestyle='-.')
            cs = CubicSpline(dists, rdf_avg_all[idx], bc_type='natural')
            dists_interp = np.linspace(dists[0], dists[-1], int(len(dists)*200))
            rdf_interp = cs(dists_interp)
            peak = np.argmax(rdf_interp)
            sigma_peaks.append(dists_interp[peak])
            # Draw vertical line at peak
            axs[i, j].axvline(x=dists_interp[peak], color='black', linestyle=':', alpha=0.75, linewidth=1.5)


            y_min, y_max = axs[i, j].get_ylim()
            y_min_all = min(y_min, y_min_all)
            y_max_all = max(y_max, y_max_all)
            axs[i, j].set_xlim(0, 2)
            axs[i, j].set_yticks([0, 1, 2, 3, 4])
            axs[i, j].set_yticklabels([0, 1, 2, 3, 4], fontsize=16)
            axs[i, j].set_xticks([0, 0.5, 1, 1.5, 2])
            axs[i, j].set_xticklabels([0, 0.5, 1, 1.5, 2], fontsize=16)
            axs[-1, j].set_xlabel(r'$r/\sigma$', fontsize=24)
            axs[i, 0].set_ylabel(r'$g(r)$', fontsize=24)
            rho, temp = float(rho_vals[idx]), float(T_vals[idx])
            axs[i, j].set_title(r'$T^{*} = %.2f,  \,  \rho^{*} = %.2f$' % (temp, rho), fontsize=18, pad=10)
            
    y_min = 0 
    y_max = np.ceil(y_max_all)
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            idx = 3*i+j
            axs[i, j].set_ylim(y_min, y_max)
            axs[i, j].text(1.5, y_max-1, r'$r_{\mathrm{peak}}=%.3f \, \sigma$' % sigma_peaks[idx], fontsize=16, ha='center', va='center', color='black')


    plt.tight_layout()
    plt.savefig('rdfs.png', dpi=300, bbox_inches='tight')

    plt.clf()

    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(18, 6))


    y_min_all = np.inf 
    y_max_all = -np.inf


    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            idx = 3*i+j
            axs[i, j].plot(t[closest_points]*diffusions[idx],
                        msds_all[idx, closest_points],
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
            y_min, y_max = axs[i, j].get_ylim()
            y_min_all = min(y_min, y_min_all)
            y_max_all = max(y_max, y_max_all)

            axs[i, j].set_xscale('log')
            axs[i, j].set_yscale('log')
            axs[-1, j].set_xlabel(r'$t/t_{ \mathrm{D} }$', fontsize=24)
            axs[i, 0].set_ylabel('MSD', fontsize=24)

            x_min, x_max = axs[i, j].get_xlim()
            axs[i, j].set_xlim(1e-5, x_max)
            rho, temp = float(rho_vals[idx]), float(T_vals[idx])
            axs[i, j].set_title(r'$T^{*} = %.2f,  \,  \rho^{*} = %.2f$' % (temp, rho), fontsize=18, pad=10)

            t_B = (2**(1/6)-sigma_peaks[idx])/np.sqrt(3*temp)
            t_L = (1/(np.pi * sigma_peaks[idx]**2 * rho))/np.sqrt(3*temp)

            t_B *= diffusions[idx]
            t_L *= diffusions[idx]

            axs[i, j].axvspan(1e-5, t_B,color='mediumseagreen', alpha=0.12)
            axs[i, j].axvspan(t_L, x_max, color='mediumpurple', alpha=0.12)

    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            axs[i, j].set_ylim(1e-5, y_max_all*2)
            axs[i, j].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()

    # Save fig as svg
    plt.savefig('msd_regions.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()

