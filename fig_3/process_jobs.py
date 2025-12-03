import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter  # or FuncFormatter/FormatStrFormatter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from textwrap import wrap
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager as fm
import seaborn as sns
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


def compute_msd(positions, t_delay_indices, length, chunk_size=16):
    """
    Memory-efficient + fast MSD computation.
    positions: (T, N, 3)
    t_delay_indices: list/array of starting frames
    length: number of frames per origin
    """
    t_delay_indices = np.asarray(t_delay_indices)
    O = len(t_delay_indices)
    N = positions.shape[1]

    # Pre-allocate output (small)
    msds = np.empty((O, length), dtype=positions.dtype)

    # Preallocate reusable buffer for slicing (max chunk, length, N, 3)
    buffer = np.empty((chunk_size, length, N, 3), dtype=positions.dtype)

    for start in tqdm(range(0, O, chunk_size)):
        stop = min(start + chunk_size, O)
        size = stop - start
        idx_chunk = t_delay_indices[start:stop]

        # Fill buffer without advanced indexing (cheap!)
        for i, idx0 in enumerate(idx_chunk):
            buffer[i] = positions[idx0:idx0+length]

        # Subtract origin
        buffer[:size] -= buffer[:size, 0:1]

        # Square displacement
        sd = np.sum(buffer[:size] ** 2, axis=-1)

        # Average over atoms
        msds[start:stop] = np.mean(sd, axis=-1)

    return msds

def compute_vacf(velocities, t_delay_indices, length, chunk_size=16):
    """
    Memory-efficient + fast VACF computation.
    velocities: (T, N, 3)
    """
    t_delay_indices = np.asarray(t_delay_indices)
    O = len(t_delay_indices)
    N = velocities.shape[1]

    vacfs = np.empty((O, length), dtype=velocities.dtype)
    buffer = np.empty((chunk_size, length, N, 3), dtype=velocities.dtype)

    for start in tqdm(range(0, O, chunk_size)):
        stop = min(start + chunk_size, O)
        size = stop - start
        idx_chunk = t_delay_indices[start:stop]

        # Load chunk into buffer
        for i, idx0 in enumerate(idx_chunk):
            buffer[i] = velocities[idx0:idx0+length]

        # Dot with initial velocity (shape broadcast: (size, L, N))
        dot = np.sum(buffer[:size] * buffer[:size, 0:1], axis=-1)
        vacfs[start:stop] = np.mean(dot, axis=-1) # Unnormalized VACF
    return vacfs


def compute_s2(g_r, r, rho=0.85):
    s2 = (g_r*np.log(g_r) - g_r + 1)
    # Set NaN to 1
    s2[np.isnan(s2)] = 1
    s2 = -2*np.pi*rho*s2*r**2
    return np.trapz(y=s2, x=r, axis=1)

def D_EH_(msds, t_array):
    # msd shape = (num_delays, length)
    return np.cumsum(msds * t_array, axis=-1) / np.cumsum(t_array*t_array) / 6

def D_GK_(vacfs, t_array, T=1, m=1):
    # vacf shape = (num_delays, length)
    factor = m/(3*T) # from partition theorem 1/2 m <v^2> = 3/2 kT
    return cumulative_trapezoid(vacfs, t_array, axis=-1, initial=0) * factor

def D_EES_(s2s, c1, c2, scaling_factor):
    # s2s shape = (num_delays, length)
    exp_term = c1 * np.exp(c2 * s2s)
    return scaling_factor * exp_term

def get_percent_error(vals, val_true):
    return 100*(vals - val_true)/val_true



def compute_all(rho, T, path, t_delay_indices, t_sample_idx, t_samples, c1, c2, scaling_factor, idx):
    positions = load_dump(os.path.join(path, f'job_{idx+1}', f'pos_{idx+1}.dat'), columns=['xu', 'yu', 'zu'], ids=False)
    velocities = load_dump(os.path.join(path, f'job_{idx+1}', f'vel_{idx+1}.dat'), columns=['vx', 'vy', 'vz'], ids=False)
    times, r, gr = parseRDF(os.path.join(path, f'job_{idx+1}', f'rdf_{idx+1}.dat'))
    msds = compute_msd(positions, t_delay_indices, length=t_sample_idx)
    vacfs = compute_vacf(velocities, t_delay_indices, length=t_sample_idx)
    s2s = compute_s2(gr, r, rho=rho)
    D_EH = D_EH_(msds, t_samples)
    D_GK = D_GK_(vacfs, t_samples, T=T, m=1)
    D_EES = D_EES_(s2s, c1, c2, scaling_factor)

    return msds, vacfs, s2s, D_EH, D_GK, D_EES, times, r

# We wrap compute_all so executor.map passes args cleanly
def compute_all_wrapper(args):
    return compute_all(*args)


def remove_trailing_zeros(x, pos):
    # Format the number with a fixed number of decimals, then strip trailing zeros
    s = f"{x:.3f}"  # or use a different precision if you prefer
    s = s.rstrip("0").rstrip(".")
    return s


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

    # t_delay times to use for witholding MSD and VACF calculations IN UNITS of tD ~ 19.53 tau
    t_delays = list(np.arange(0, 101) * 0.0001) + list(0.01 + np.arange(1, 181) * 0.0005) + list(0.1 + np.arange(1, 181) * 0.005)
    # HARD coded from generate_jobs.py which dumps every 2 timesteps with dt = 0.002 for 19532 total steps
    t_array = np.arange(0, 19532, 2) * timestep * D_ref
    t_samples = np.arange(0, 9766, 2) * timestep   # for s2 calculations

    t_delay_indices = [] # start of time-series data for measurments
    t_sample_idx = np.ceil(1/(2*timestep*D_ref)).astype(int) # 4883  # number of steps corresponding to 1 tD

    for t in t_delays:
        idx = (np.abs(t_array - t)).argmin()
        t_delay_indices.append(idx) if idx not in t_delay_indices else None

    msds_all = []
    vacfs_all = []
    s2s_all = []    
    D_EH_all = []
    D_GK_all = []   
    D_EES_all = []

    N_CORES = np.min([10, os.cpu_count()])  # or set to something smaller

    for i, directory in enumerate(directories):
        rho, T = float(rho_vals[i]), float(T_vals[i])
        scaling_factor = rho**(-1/3)*T**(1/2)
        job_args = [(rho, T, directory, t_delay_indices, t_sample_idx, t_samples, c1, c2, scaling_factor, j) for j in range(num_jobs)]

        print(f"Running {num_jobs} jobs in parallel with {N_CORES} cores...")
        results = []

  

        with ProcessPoolExecutor(max_workers=N_CORES) as ex:
            for out in tqdm(ex.map(compute_all_wrapper, job_args), total=num_jobs):
                results.append(out)

            # Unpack results
            msds_all  = np.array([res[0] for res in results])
            vacfs_all = np.array([res[1] for res in results])
            s2s_all   = np.array([res[2] for res in results])
            D_EH_all  = np.array([res[3] for res in results])
            D_GK_all  = np.array([res[4] for res in results])
            D_EES_all = np.array([res[5] for res in results])
            times         = np.array(results[0][6])   # same for all
            r            = np.array(results[0][7])   # same for all


        # For figure 3, EH and GK use MSD and VACF averaged prior to D calc, whereas EES uses s2 from single trajectory 
        msd_avg = np.mean(msds_all, axis=0)
        vacf_avg = np.mean(vacfs_all, axis=0)
        D_EH_avg = D_EH_(msd_avg, t_samples) # Trajectory-averaged MSD
        D_GK_avg = D_GK_(vacf_avg, t_samples, T=T, m=1) # Trajectory-averaged VACF
        D_EES_avg = np.mean(D_EES_all, axis=0) # Single trajectory s2s averaged D

        # Figure 3 
        D_EH_err = get_percent_error(D_EH_avg, D_ref)
        D_GK_err = get_percent_error(D_GK_avg, D_ref)
        D_EES_err = get_percent_error(D_EES_avg, D_ref)

        # For figure 4 
        D_EH_avg_SNR = np.mean(D_EH_all, axis=0) 
        D_GK_avg_SNR = np.mean(D_GK_all, axis=0) 
        D_EES_avg_SNR = np.mean(D_EES_all, axis=0) 
        D_EH_std = np.std(D_EH_all, axis=0)
        D_GK_std = np.std(D_GK_all, axis=0)
        D_EES_std = np.std(D_EES_all, axis=0)
        D_EH_SNR = D_EH_avg_SNR / D_EH_std
        D_GK_SNR = D_GK_avg_SNR / D_GK_std
        D_EES_SNR = D_EES_avg_SNR / D_EES_std

        t_samples_reduced = t_samples*D_ref 
        t_delays_reduced = np.array(t_delay_indices) * timestep * D_ref


        times_new_roman = fm.FontProperties(family="Times New Roman", size=32)


        ### FIGURE 3 EH ERROR PLOT ###
        max_idx = np.ceil(np.argmin(np.abs(t_samples_reduced - .1))).astype(int)
        max_delay_idx = np.ceil(np.argmin(np.abs(t_delays_reduced - .1))).astype(int)
        time_series = t_samples_reduced[:max_idx+1]
        D_EH_err = 100 * (D_EH_avg - D_ref) / D_ref
        vals = D_EH_err[:max_delay_idx+1, :max_idx+1]

        # Create a meshgrid for the contour plot
        X, Y = np.meshgrid(time_series, t_delays_reduced[:max_delay_idx+1])


        max_val = 110
        min_val = -100

        intercept = 1.5*0.0512
        slope = -1.5
        masked_vals = np.ma.masked_where(Y <= slope * X + intercept, vals)
        masked_vals = np.ma.masked_where(X <= .0512*.55, masked_vals)




        # Plot the contour plot
        plt.figure(figsize=(12, 8))
        color_norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)  # Center at 0

        # Filled contours
        contour_filled = plt.contourf(X, Y, vals, levels=np.linspace(min_val, max_val, 1000), norm=color_norm, cmap='RdBu_r')

        # Add contour lines
        # custom_levels = [3] + [con for con in np.linspace(3.5, 20, 200)] + [21, 22, 23, 24, 25, 30]
        custom_levels = [3, 10, 30]
        contour_lines = plt.contour(X, Y, masked_vals, levels=custom_levels, colors='black', linewidths=2, linestyles='solid', alpha=0.75)

        c_lines = plt.clabel(contour_lines, inline=True, fontsize=32, fmt="%.1f", colors='black') #, manual=label_positions)

        for text in c_lines:
            text.set_fontproperties(times_new_roman)

        # Add color bar
        cbar = plt.colorbar(contour_filled)
        cbar.set_label(r"$\mathrm{\% \ Error}$", rotation=270, labelpad=25, fontsize=48)
        # Place colorbar label at top of colorbar given specific coords 
        cbar.set_ticks(np.linspace(-100, 110, 7))  # Use dynamic ticks
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(times_new_roman)

        # Axis labels and title
        plt.xlabel(r"$t_{\mathrm{sample}}/t_D$", fontsize=48, labelpad=10)
        plt.ylabel(r"$t_{\mathrm{delay}}/t_D$", fontsize=48, labelpad=10)
        title = r'% Error in Estimation of Self-Diffusivity as a Function of Start and Sample Time via EH'
        wrapped_title = '\n'.join(wrap(title, width=50))

        # plt.title(wrapped_title, fontsize=30, pad=10, fontweight='bold')

        # Adjust ticks
        plt.yticks(np.arange(0, .125, .025), fontsize=32)
        plt.xticks(np.arange(0, .125, .025), fontsize=32)
        # Add 10 padding to both ticks 
        plt.tick_params(axis='x', direction='in', pad=10, length=10)
        plt.tick_params(axis='y', direction='in', pad=10, length=10)

        # Make the tick fonts the same font as the labels 
        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')



        ax = plt.gca()

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(remove_trailing_zeros))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(remove_trailing_zeros))


        plt.xlim(0, .1)
        plt.ylim(0, .1)

        # Remove top and right ticks
        ax.yaxis.set_ticks_position('left')

        ax.xaxis.set_ticks_position('bottom')



        for spine in ax.spines.values():
            spine.set_linewidth(3)

        plt.tight_layout()
        plt.savefig('Error_EH.png', dpi=450, format='png') # switch to svg for vector graphics
        # delete plt object
        plt.clf()




        ### FIGURE 3 GK ERROR PLOT ###
        max_idx = np.ceil(np.argmin(np.abs(t_samples_reduced - .1))).astype(int)
        max_delay_idx = np.ceil(np.argmin(np.abs(t_delays_reduced - .1))).astype(int)
        time_series = t_samples_reduced[:max_idx+1]
        D_GK_err = 100 * (D_GK_avg - D_ref) / D_ref
        vals = D_GK_err[:max_delay_idx+1, :max_idx+1]

        # Create a meshgrid for the contour plot
        X, Y = np.meshgrid(time_series, t_delays_reduced[:max_delay_idx+1])




        max_val = 110
        min_val = -100

        bools = np.logical_or(X <= 0.14*0.0512, Y <= (0.2 - 0.2*X)*0.0512)
        bools = np.logical_or(bools, Y <= 0.1*0.0512)
        bools = np.logical_or(bools, Y >= -0.75*X+0.13)
        masked_vals = np.ma.masked_where(bools, vals)




        # Plot the contour plot
        plt.figure(figsize=(12, 8))
        color_norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)  # Center at 0

        # Filled contours
        contour_filled = plt.contourf(X, Y, vals, levels=np.linspace(min_val, max_val, 1000), norm=color_norm, cmap='RdBu_r')

        # Add contour lines
        # custom_levels = [3] + [con for con in np.linspace(3.5, 20, 200)] + [21, 22, 23, 24, 25, 30]
        custom_levels = [3, 10, 30]
        contour_lines = plt.contour(X, Y, masked_vals, levels=custom_levels, colors='black', linewidths=2, linestyles='solid', alpha=0.75)

        c_lines = plt.clabel(contour_lines, inline=True, fontsize=32, fmt="%.1f", colors='black') #, manual=label_positions)

        for text in c_lines:
            text.set_fontproperties(times_new_roman)

        # Add color bar
        cbar = plt.colorbar(contour_filled)
        cbar.set_label(r"$\mathrm{\% \ Error}$", rotation=270, labelpad=25, fontsize=48)
        # Place colorbar label at top of colorbar given specific coords 
        cbar.set_ticks(np.linspace(-100, 110, 7))  # Use dynamic ticks
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(times_new_roman)

        # Axis labels and title
        plt.xlabel(r"$t_{\mathrm{sample}}/t_D$", fontsize=48, labelpad=10)
        plt.ylabel(r"$t_{\mathrm{delay}}/t_D$", fontsize=48, labelpad=10)
        title = r'% Error in Estimation of Self-Diffusivity as a Function of Start and Sample Time via EH'
        wrapped_title = '\n'.join(wrap(title, width=50))

        # plt.title(wrapped_title, fontsize=30, pad=10, fontweight='bold')

        # Adjust ticks
        plt.yticks(np.arange(0, .125, .025), fontsize=32)
        plt.xticks(np.arange(0, .125, .025), fontsize=32)
        # Add 10 padding to both ticks 
        plt.tick_params(axis='x', direction='in', pad=10, length=10)
        plt.tick_params(axis='y', direction='in', pad=10, length=10)

        # Make the tick fonts the same font as the labels 
        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')



        ax = plt.gca()

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(remove_trailing_zeros))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(remove_trailing_zeros))


        plt.xlim(0, .1)
        plt.ylim(0, .1)

        # Remove top and right ticks
        ax.yaxis.set_ticks_position('left')

        ax.xaxis.set_ticks_position('bottom')



        for spine in ax.spines.values():
            spine.set_linewidth(3)

        plt.tight_layout()
        plt.savefig('Error_GK.png', dpi=450, format='png') # switch to svg for vector graphics
        # delete plt object
        plt.clf()


        ### FIGURE 3 EES ERROR PLOT ###
        ees_samples = np.floor(D_EES_avg.shape[0]/2).astype(int)
        vals = np.empty((ees_samples, ees_samples))
        for i in range(ees_samples):
            vals[i] = D_EES_avg[i:ees_samples+i]

        vals = 100 * (vals - D_ref) / D_ref
        vals = vals[t_delay_indices[:-1], :]
        vals = vals[:, t_delay_indices[:-1]]

        # Create a meshgrid for the contour plot
        X, Y = np.meshgrid(t_delays_reduced[:-1], t_delays_reduced[:-1])

        slope = -1
        intercept = 0.015

        mask = Y > slope*X + intercept
        masked_vals = np.ma.masked_where(Y <= slope * X + intercept, vals)

        max_val = 110
        min_val = -100

        # Plot the contour plot
        plt.figure(figsize=(12, 8))
        color_norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)  # Center at 0

        # Filled contours
        contour_filled = plt.contourf(X, Y, vals, levels=np.linspace(min_val, max_val, 1000), norm=color_norm, cmap='RdBu_r')

        # Add contour lines
        # custom_levels = [3] + [con for con in np.linspace(3.5, 20, 200)] + [21, 22, 23, 24, 25, 30]
        custom_levels = [3, 10, 30]
        contour_lines = plt.contour(X, Y, masked_vals, levels=custom_levels, colors='black', linewidths=2, linestyles='solid', alpha=0.75)

        c_lines = plt.clabel(contour_lines, inline=True, fontsize=32, fmt="%.1f", colors='black') #, manual=label_positions)

        for text in c_lines:
            text.set_fontproperties(times_new_roman)

        # Add color bar
        cbar = plt.colorbar(contour_filled)
        cbar.set_label(r"$\mathrm{\% \ Error}$", rotation=270, labelpad=25, fontsize=48)
        # Place colorbar label at top of colorbar given specific coords 
        cbar.set_ticks(np.linspace(-100, 110, 7))  # Use dynamic ticks
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(times_new_roman)

        # Axis labels and title
        plt.xlabel(r"$t_{\mathrm{sample}}/t_D$", fontsize=48, labelpad=10)
        plt.ylabel(r"$t_{\mathrm{delay}}/t_D$", fontsize=48, labelpad=10)
        title = r'% Error in Estimation of Self-Diffusivity as a Function of Start and Sample Time via EH'
        wrapped_title = '\n'.join(wrap(title, width=50))

        # plt.title(wrapped_title, fontsize=30, pad=10, fontweight='bold')

        # Adjust ticks
        plt.yticks(np.arange(0, .125, .025), fontsize=32)
        plt.xticks(np.arange(0, .125, .025), fontsize=32)
        # Add 10 padding to both ticks 
        plt.tick_params(axis='x', direction='in', pad=10, length=10)
        plt.tick_params(axis='y', direction='in', pad=10, length=10)

        # Make the tick fonts the same font as the labels 
        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')



        ax = plt.gca()

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(remove_trailing_zeros))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(remove_trailing_zeros))


        plt.xlim(0, .1)
        plt.ylim(0, .1)

        # Remove top and right ticks
        ax.yaxis.set_ticks_position('left')

        ax.xaxis.set_ticks_position('bottom')



        for spine in ax.spines.values():
            spine.set_linewidth(3)

        plt.tight_layout()
        plt.savefig('Error_EES.png', dpi=450, format='png') # switch to svg for vector graphics
        # delete plt object
        plt.clf()



        # All in tD units
        np.savez_compressed('msds_all.npz', msds_all=msds_all, t_array=t_samples*D_ref, t_delay_indices=t_delay_indices)
        np.savez_compressed('vacfs_all.npz', vacfs_all=vacfs_all, t_array=t_samples*D_ref, t_delay_indices=t_delay_indices)
        np.savez_compressed('s2s_all.npz', s2s_all=s2s_all, r=r, t_array=times*D_ref)
        np.savez_compressed('D_EH_all.npz', D_EH_all=D_EH_all, t_array=t_samples*D_ref, t_delay_indices=t_delay_indices)
        np.savez_compressed('D_EH_avg.npz', D_EH_avg=D_EH_avg, t_array=t_samples*D_ref)
        np.savez_compressed('D_EH_avg_SNR.npz', D_EH_avg_SNR=D_EH_avg_SNR, t_array=t_samples*D_ref)
        np.savez_compressed('D_EH_std.npz', D_EH_std=D_EH_std, t_array=t_samples*D_ref)
        np.savez_compressed('D_EH_SNR.npz', D_EH_SNR=D_EH_SNR, t_array=t_samples*D_ref)
        np.savez_compressed('D_GK_all.npz', D_GK_all=D_GK_all, t_array=t_samples*D_ref, t_delay_indices=t_delay_indices)
        np.savez_compressed('D_GK_avg.npz', D_GK_avg=D_GK_avg, t_array=t_samples*D_ref)
        np.savez_compressed('D_GK_avg_SNR.npz', D_GK_avg_SNR=D_GK_avg_SNR, t_array=t_samples*D_ref)
        np.savez_compressed('D_GK_std.npz', D_GK_std=D_GK_std, t_array=t_samples*D_ref)
        np.savez_compressed('D_GK_SNR.npz', D_GK_SNR=D_GK_SNR, t_array=t_samples*D_ref)
        np.savez_compressed('D_EES_all.npz', D_EES_all=D_EES_all, t_array=t_samples*D_ref, t_delay_indices=t_delay_indices)
        np.savez_compressed('D_EES_avg.npz', D_EES_avg=D_EES_avg, t_array=t_samples*D_ref)
        np.savez_compressed('D_EES_avg_SNR.npz', D_EES_avg_SNR=D_EES_avg_SNR, t_array=t_samples*D_ref)
        np.savez_compressed('D_EES_std.npz', D_EES_std=D_EES_std, t_array=t_samples*D_ref)
        np.savez_compressed('D_EES_SNR.npz', D_EES_SNR=D_EES_SNR, t_array=t_samples*D_ref)
        np.savez_compressed('D_errors.npz', D_EH_err=D_EH_err, D_GK_err=D_GK_err, D_EES_err=D_EES_err, t_array=t_samples*D_ref)

  

if __name__ == "__main__":
    main()

