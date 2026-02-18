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
    t_array = np.arange(0, 19532, 1) * timestep * D_ref
    t_samples = np.arange(0, 9766, 1) * timestep   # for s2 calculations

    t_delay_indices = [] # start of time-series data for measurments
    t_sample_idx = np.ceil(1/(1*timestep*D_ref)).astype(int) # 4883  # number of steps corresponding to 1 tD

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

        print(D_EH_SNR.shape)

        t_samples_reduced = t_samples*D_ref 
        t_delays_reduced = np.array(t_delay_indices) * timestep * D_ref


        times_new_roman = fm.FontProperties(family="Times New Roman", size=32)

        fig, ax = plt.subplots(5, 2, figsize=(9, 15)) #, sharex='col') #, sharey='row')

        GK_idxs  = [0, 98, 127, 178, 278]
        EH_idxs  = [0, 98, 127, 178, 278]
        EES_idxs = [0, 98, 127, 178, 278]
        d_lags   = [0, 0.01, 0.025, 0.05, 0.1]

        # EES_plot_indices = [    0,     1,     2,     3,     4,     7,    11,    19,    32, 54,    91,   152,   256,   431,   725,  1220,  2052,  3452, 5806,  9765,    
        #         98, 99, 100, 101, 102, 105, 109, 117, 130, 152, 189, 250, 354, 529, 823, 1318, 2150, 3550, 5904, 9863,   
        #         244, 245, 246, 247, 248, 251, 255, 263, 276, 298, 335, 396, 500, 675, 969, 1464, 2296, 3696, 6050, 10009,   
        #         488, 489, 490, 491, 492, 495, 499, 507, 520, 542, 579, 640, 744, 919, 1213, 1708, 2540, 3940, 6294, 10253, 
        #         977, 978, 979, 980, 981, 984, 988, 996, 1009, 1031, 1068, 1129, 1233, 1408, 1702, 2197, 3029, 4429, 6783, 10742]

        GK_indices_plot = [0, 1, 2, 3 , 4, 7, 11, 19, 32, 54, 91, 152, 256, 431, 725, 1220, 2052, 3452, 5806, 9765]
        EH_indices_plot = GK_indices_plot
        EES_plot_indices = EH_indices_plot

        min_y = np.min([np.min(D_GK_avg_SNR[idx] - 2 * D_GK_std[idx]) for idx in GK_idxs]) / 0.0512
        max_y = np.max([np.max(D_GK_avg_SNR[idx] + 2 * D_GK_std[idx]) for idx in GK_idxs]) / 0.0512
        print(times.shape)
        vacf_times_high_res_mod = times.copy()
        vacf_times_high_res_mod[0] = vacf_times_high_res_mod[1] * 0.1

        vacf_times_high_res_mod_diff = vacf_times_high_res_mod * 0.0512
        msd_times_diff = times * 0.0512
        msd_times_diff[0] = msd_times_diff[1] * 0.1

        # EES log-sampled indices
        logs = np.log10(vacf_times_high_res_mod_diff)
        log_points = np.linspace(np.min(logs), np.max(logs), 20)
        indices = [np.argmin((logs - lp) ** 2) for lp in log_points]

        for r in range(5):
            # -------- Left col: D/Deq --------
            aL = ax[r, 0]
            aL.plot(msd_times_diff[EH_indices_plot],
                    D_EH_avg_SNR[EH_idxs[r], EH_indices_plot] / 0.0512,
                    color='steelblue', label='EH', marker='o', markersize=4)
            aL.plot(vacf_times_high_res_mod_diff[GK_indices_plot],
                    D_GK_avg_SNR[GK_idxs[r], GK_indices_plot] / 0.0512,
                    color='indigo', label='GK', marker='o', markersize=4)
            aL.plot(vacf_times_high_res_mod_diff[EES_plot_indices],
                    D_EES_avg_SNR[EES_idxs[r], EES_plot_indices] / 0.0512,
                    color='seagreen', label='EES', marker='o', markersize=4)

            # Fill between (only for D/Deq)
            aL.fill_between(vacf_times_high_res_mod_diff,
                            (D_GK_avg_SNR[GK_idxs[r], :] - 2 * D_GK_std[GK_idxs[r], :]) / 0.0512,
                            (D_GK_avg_SNR[GK_idxs[r], :] + 2 * D_GK_std[GK_idxs[r], :]) / 0.0512,
                            alpha=0.25, color='blueviolet')
            aL.fill_between(msd_times_diff,
                            (D_EH_avg_SNR[EH_idxs[r], :] - 2 * D_EH_std[EH_idxs[r], :]) / 0.0512,
                            (D_EH_avg_SNR[EH_idxs[r], :] + 2 * D_EH_std[EH_idxs[r], :]) / 0.0512,
                            alpha=0.25, color='cornflowerblue')
            aL.fill_between(vacf_times_high_res_mod_diff[EES_plot_indices],
                            (D_EES_avg_SNR[EES_idxs[r], EES_plot_indices] - 2 * D_EES_std[EES_idxs[r], EES_plot_indices]) / 0.0512,
                            (D_EES_avg_SNR[EES_idxs[r], EES_plot_indices] + 2 * D_EES_std[EES_idxs[r], EES_plot_indices]) / 0.0512,
                            alpha=0.25, color='green')

            aL.axhline(1.0, linestyle='--', color='black', linewidth=2)
            aL.set_xscale('log', base=10)
            aL.set_xlim(vacf_times_high_res_mod_diff[1] / 10, 1)
            aL.set_ylim(min_y, max_y)
            aL.tick_params(axis='x', labelsize=18)
            aL.tick_params(axis='y', labelsize=18)

        #     if r == 0:
        #         aL.set_title(r"$D/D_{\mathrm{eq}}$", fontsize=28, pad=20)

            # -------- Right col: SNR --------
            aR = ax[r, 1]
            GK_zip  = zip(vacf_times_high_res_mod_diff[GK_indices_plot], 1 / D_GK_SNR[GK_idxs[r], GK_indices_plot])
            EH_zip  = zip(msd_times_diff[EH_indices_plot],                1 / D_EH_SNR[EH_idxs[r], EH_indices_plot])
            EES_zip = zip(vacf_times_high_res_mod_diff[EES_plot_indices], 1 / D_EES_SNR[EES_idxs[r], EES_plot_indices])

            GK_zip_plot  = [(x, y) for x, y in GK_zip if x >= 0.02]
            EH_zip_plot  = [(x, y) for x, y in EH_zip if x >= 0.02]
            EES_zip_plot = [(x, y) for x, y in EES_zip if x >= 0.02]

            aR.plot([x for x, y in GK_zip_plot],  [1 / y for x, y in GK_zip_plot],
                    color='indigo', linestyle='dashed', marker='o', markersize=4)
            aR.plot([x for x, y in EH_zip_plot],  [1 / y for x, y in EH_zip_plot],
                    color='steelblue', linestyle='dashed', marker='o', markersize=4)
            aR.plot([x for x, y in EES_zip_plot], [y for x, y in EES_zip_plot],
                    color='seagreen', linestyle='dashed', marker='o', markersize=4)

            aR.set_xlim(0.0, None)
            aR.set_ylim(0, 50)
            aR.tick_params(axis='x', labelsize=18)
            aR.tick_params(axis='y', labelsize=18)
            ax[r, 0].set_ylabel(r"$D/D_{\mathrm{eq}}$", fontsize=20, labelpad=5)
            ax[r, 1].set_ylabel(r"$\mathbb{E}[D]/\sqrt{\mathbb{V} \, [D]}$", fontsize=20, labelpad=5)

            

        ax[0, 0].fill_between(-vacf_times_high_res_mod_diff, 1.9*np.ones_like(vacf_times_high_res_mod_diff), 2.1*np.ones_like(vacf_times_high_res_mod_diff), linewidth=2, alpha=0.25, color='grey', label=r'$\pm 2 \sigma$')



        # Common x-labels
        ax[-1, 0].set_xlabel(r"$t_{\mathrm{sample}}/t_D$", fontsize=24, labelpad=5)
        ax[-1, 1].set_xlabel(r"$t_{\mathrm{sample}}/t_D$", fontsize=24, labelpad=5)

        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']


        # Spine formatting
        for i, a in enumerate(ax.flatten()):
            for spine in ["left", "bottom", "top", "right"]:
                a.spines[spine].set_linewidth(2)
                a.text(
                        0.02, 0.95, panel_labels[i], 
                        transform=a.transAxes,
                        fontsize=24, 
                        va='top', ha='left',
                        fontweight='normal'
                )

        plt.subplots_adjust(wspace=0.2)

        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels,
                loc="lower center",
                ncol=4,                # 3 entries per row: GK, EH, EES
                fontsize=24,
                frameon=False,
                bbox_to_anchor=(0.5, -0.04))  # adjust vertical offset as needed


        row_titles = [r"$t/t_D = 0$",
                    r"$t/t_D = 0.01$",
                    r"$t/t_D = 0.025$",
                    r"$t/t_D = 0.05$",
                    r"$t/t_D = 0.10$"]

        for r, title in enumerate(row_titles):
            # Get the y-position of the row using the first column axes
            ax[r, 0].set_title(title, fontsize=24, pad=15)


        plt.tight_layout()

        plt.savefig('fig_4.png', dpi=450, format='png', bbox_inches='tight', pad_inches=0.1)
  

if __name__ == "__main__":
    main()

