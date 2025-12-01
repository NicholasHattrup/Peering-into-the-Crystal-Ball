import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import scienceplots
import matplotlib as mpl
import copy
from matplotlib.ticker import StrMethodFormatter  # or FuncFormatter/FormatStrFormatter
plt.style.use(['science', 'no-latex'])



mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman','Palatino','Georgia'],
    'font.size':         12,     # base font size
    'axes.titlesize':    16,
    'axes.labelsize':    16,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'axes.linewidth':    3,    # thicker axis lines
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'axes.prop_cycle':   mpl.cycler('color', mpl.cm.tab10.colors),
})



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


def load_dump(file_name, columns, ids=True):
    with open(file_name, 'r') as f:
        # Read the entire file into memory
        lines = f.readlines()

    num_atoms = None
    header_size = 0

    # Parse the header to find number of atoms and header size
    for i, line in enumerate(lines):
        line = line.strip()
        if 'ITEM: NUMBER OF ATOMS' in line:
            num_atoms = int(lines[i + 1].strip())
        elif 'ITEM: ATOMS' in line:
            header_size = i + 1
            break

    if num_atoms is None:
        raise ValueError("Number of atoms not found in the file header.")

    # Calculate lines per configuration
    lines_per_config = header_size + num_atoms

    # Calculate the number of configurations
    total_lines = len(lines)
    num_configs = total_lines // lines_per_config

    # Preallocate the array
    data = np.zeros((num_configs, num_atoms, len(columns)))
    start_idx = header_size
    end_idx = start_idx + num_atoms

    if ids:
        num_features = len(columns) + 1
    else:
        num_features = len(columns)
    

    # Extract configurations
    if ids: 
        for config_index in range(num_configs):
            config_data = np.fromstring("\n".join(lines[start_idx:end_idx]), sep=' ').reshape(-1, num_features)
            data[config_index, config_data[:, 0].astype(int) - 1] = config_data[:, 1:]
            start_idx += lines_per_config
            end_idx += lines_per_config
    else:
        for config_index in range(num_configs):
            config_data = np.fromstring("\n".join(lines[start_idx:end_idx]), sep=' ').reshape(-1, num_features)
            data[config_index] = config_data
            start_idx += lines_per_config
            end_idx += lines_per_config
    return data

def s2_integrand(g_r, r, rho=1):
    s2 = (g_r*np.log(g_r) - g_r + 1)
    # Set NaN to 1
    s2[np.isnan(s2)] = 1
    return -2*np.pi*rho*s2*r**2





def main():
    print('Processing jobs...')
    timestep = 0.002
    D_ref = 0.0512  # Reference diffusion coefficient for scaling time axis
    # We use a = 0.728 and b = 0.449 -> b * exp(a * s2)
    c1 = 0.449 
    c2 = 0.728
    rho_vals = sys.argv[1].split(',')
    T_vals = sys.argv[2].split(',')
    num_jobs = int(sys.argv[3])
    NUM_ROWS = 3 # THIS IS HARD CODED ASSUMING len(rho_vals) == len(T_vals) == 1
    NUM_COLS = 2

    directories = [f'rho_{rho}_T_{T}' for rho, T in zip(rho_vals, T_vals)] # Should just be one 


    for i, directory in enumerate(directories):
        rho, T = float(rho_vals[i]), float(T_vals[i])
        scaling_factor = rho**(-1/3)*T**(1/2)
        msds = [pd.read_csv(directory + f'/job_{i}/' + f"msd_{i}.dat", comment="#", delim_whitespace=True, names=['time_step', 'msd'], header=None, skiprows=2) for i in range(1, num_jobs+1)]
        vacfs = [pd.read_csv(directory + f'/job_{i}/' + f"vacf_{i}.dat", comment="#", delim_whitespace=True, names=['time_step', 'vacf'], header=None, skiprows=2) for i in range(1, num_jobs+1)]

        msd_array = np.array([msd['msd'].values for msd in msds])
        vacf_array = np.array([vacf['vacf'].values for vacf in vacfs])
        times = msds[0]['time_step'].values.astype(float)
        times -= times[0]
        times *= timestep

        ave_msd = np.mean(msd_array, axis=0)
        ave_vacf = np.mean(vacf_array, axis=0)

        std_msd = np.std(msd_array, axis=0)
        std_vacf = np.std(vacf_array, axis=0)

        D_EH = np.zeros((len(msds), len(times)))
        for idx, msd in enumerate(msds):
            D = [0]
            msd_data = msd['msd'].values
            for t in range(2, len(times)+1):
                est = (times[:t] * msd_data[:t]).sum() / (times[:t]**2).sum() / 6
                D.append(est)
            D_EH[idx] = D

        D_EH_ave = D_EH.mean(axis=0)
        D_EH_std = D_EH.std(axis=0)

        D_GK = np.zeros((len(vacfs), len(times)))
        for idx, vacf in enumerate(vacfs):
            vacf_data = vacf['vacf'].values
            D = cumulative_trapezoid(vacf_data, times, initial=0) / 3
            D_GK[idx] = D

        D_GK_ave = D_GK.mean(axis=0)
        D_GK_std = D_GK.std(axis=0)



        s2_vals = []
        for i in range(1, num_jobs+1):
            s2_times, s2_distances, s2_rdf = parseRDF(directory + f'/job_{i}/' + f'rdf_{i}.dat')
            # cumulative average rdf 
            rdf_cum_avg = np.cumsum(s2_rdf, axis=0) / (np.arange(s2_rdf.shape[0])[:, None] + 1)

            s2_cum_avg = np.log(rdf_cum_avg) * rdf_cum_avg - rdf_cum_avg + 1
            # Switch nan to 1 
            s2_cum_avg[np.isnan(s2_cum_avg)] = 1
            s2_cum_avg *= s2_distances ** 2
            s2_cum_avg *= -2 * np.pi * rho
            s2_cum_avg = np.trapz(s2_cum_avg, s2_distances, axis=1)
            s2_vals.append(s2_cum_avg)


        s2_times -= s2_times[0]
        s2_times *= timestep

        s2_vals_mean = np.mean(s2_vals, axis=0)
        s2_vals_std = np.std(s2_vals, axis=0)


        D_EES = [] 
        for s2_val in s2_vals:
            D = c1 * np.exp(c2 * s2_val)
            D *= scaling_factor
            D_EES.append(D)
        D_EES_ave = c1 * np.exp(c2 * s2_vals_mean)
        D_EES_ave *= scaling_factor
        D_EES_std = np.std(D_EES, axis=0)


    # Make 3 plots 
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    msd_idx = np.argmin(np.abs(times - .1/D_ref))

    ax[0,0].set_xlim(0, .1)
    ax[0,0].set_ylim(0, ave_msd[msd_idx])
    ax[1,0].set_xlim(0, 0.1)
    ax[1,1].set_xlim(0, 0.1)
    ax[0, 1].set_xlim(0, 1)
    ax[2,0].set_xlim(0, .1)
    ax[2, 1].set_xlim(0, .1)


    ax[0,0].plot(times*D_ref, ave_msd, color='darkblue', label='Average', linewidth=2)
    ax[0, 0].fill_between(times*D_ref, ave_msd - 2*std_msd, ave_msd + 2*std_msd, color='blue', alpha=0.15, label=r'$\pm 2\sigma$')


    ax[1,0].plot(times*D_ref, ave_vacf, color='purple', label='Average', linewidth=2)
    ax[1, 0].fill_between(times*D_ref, ave_vacf - 2*std_vacf, ave_vacf + 2*std_vacf, color='purple', alpha=0.15, label=r'$\pm 2\sigma$')


    ax[2,0].plot(s2_times*D_ref, -s2_vals_mean, color='darkgreen', label='Average', linewidth=2)
    ax[2,0].fill_between(s2_times*D_ref, -s2_vals_mean - 2*s2_vals_std, -s2_vals_mean + 2*s2_vals_std, color='green', alpha=0.15, label=r'$\pm 2\sigma$')

    ax[0,1].plot(times*D_ref, D_EH_ave, color='darkblue', linewidth=2)
    ax[0,1].fill_between(times*D_ref, D_EH_ave - 2*D_EH_std, D_EH_ave + 2*D_EH_std, color='blue', alpha=0.15)

    ax[1,1].plot(times*D_ref, D_GK_ave, color='purple', linewidth=2)
    ax[1,1].fill_between(times*D_ref, D_GK_ave - 2*D_GK_std, D_GK_ave + 2*D_GK_std, color='purple', alpha=0.15)


    ax[2,1].plot(s2_times*D_ref, D_EES_ave, color='darkgreen', linewidth=2)
    ax[2,1].fill_between(s2_times*D_ref, D_EES_ave - 2*D_EES_std, D_EES_ave + 2*D_EES_std, color='green', alpha=0.15)

    ax[0, 1].plot(times*D_ref, D_ref*np.ones_like(times), color='black', linestyle='-.', label=r'$D_{\mathrm{eq}}$')
    ax[1, 1].plot(times*D_ref, D_ref*np.ones_like(times), color='black', linestyle='-.', label=r'$D^{*}$')
    ax[2, 1].plot(s2_times*D_ref, D_ref*np.ones_like(s2_times), color='black', linestyle='-.', label=r'$D^{*}$')


    ax[0, 0].set_ylabel(r'MSD', fontsize=24)
    ax[1, 0].set_ylabel(r'VACF', fontsize=24)
    ax[2, 0].set_ylabel(r'$-s_2$', fontsize=24)

    ax[2, 0].set_xlabel(r'$t/t_{D}$', fontsize=32)
    ax[0, 1].set_ylabel(r'$D_{\mathrm{EH}}$', fontsize=32)
    ax[1, 1].set_ylabel(r'$D_{\mathrm{GK}}$', fontsize=32)
    ax[2, 1].set_xlabel(r'$t/t_{D}$', fontsize=32)
    ax[2, 1].set_ylabel(r'$D_{\mathrm{EES}}$', fontsize=32)


    for a in ax.flatten():

        a.patch.set_facecolor('white')
        # 5) tighten up tick placement
        a.tick_params(which='major', width=1.0)
        a.tick_params(which='minor', length=3)

    ax[2, 1].set_ylim(s2_vals_mean[-1]-.25, s2_vals_mean[-1]+.25)
    ax[0, 1].set_ylim(0, 0.075)
    ax[2, 1].set_ylim(0, 0.075)
    ax[0, 1].set_xlim(0, 0.1)
    ax[1, 1].set_ylim(0, 0.075)
    ax[2, 0].set_ylim(0, 4)



    for a in ax.flatten():
        a.tick_params(axis='both', which='major', labelsize=22)


    # Set y ticks for ax 1 0
    ax[0, 1].set_yticks([0, 0.025, 0.05, 0.075])
    ax[1, 1].set_yticks([0, 0.025, 0.05, 0.075])
    ax[2, 1].set_yticks([0, 0.025, 0.05, 0.075])

    ax[0, 0].set_yticks([0, 0.2, 0.4, 0.6])
    ax[1, 0].set_yticks([0, 1, 2, 3])

    # Set x ticks for ax 1 0
    for a in ax.flatten():
        a.set_xticks([0, 0.05, 0.1])

    ax[0, 0].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[0, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[1, 0].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[1, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[2, 0].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075,          
    ax[0, 1].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[0, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0    .1      
    ax[1, 1].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[1, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[2, 1].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1
    ax[2, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))  # 0.025, 0.05, 0.075, 0.1


    plt.tight_layout()

    handles, labels = ax[0, 0].get_legend_handles_labels()
    handles_2, labels_2 = ax[0, 1].get_legend_handles_labels()

    handles.extend(handles_2)
    labels.extend(labels_2)

    handles = [copy.copy(h) for h in handles]   
    handles[0].set_color('black')
    handles[1].set_color('black')


    fig.legend(
        handles, labels,
        ncol=3,
        fontsize=18,
        frameon=False,            # ← your “row 1” label
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08)         # tweak the y to sit just under the axes
    )


    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for i, a in enumerate(ax.flatten()):
        a.text(0.05, 0.9, labels[i], transform=a.transAxes, fontsize=28,
                verticalalignment='top') #, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))



    # # make resolution better for viewing in notebook 
    plt.tight_layout()

    # plt.savefig("fig_1.svg", dpi=450, bbox_inches='tight', format='svg') # svg if edits needed for exact figure from paper


    plt.savefig("fig_1.png", dpi=450, bbox_inches='tight', format='png')





if __name__ == "__main__":
    main()

