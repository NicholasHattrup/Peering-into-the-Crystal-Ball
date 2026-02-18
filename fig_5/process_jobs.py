import matplotlib as mpl
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, njit
import seaborn as sns
import scienceplots
from textwrap import wrap
from matplotlib.colors import TwoSlopeNorm
import time 
plt.style.use(['science', 'no-latex'])
from matplotlib import font_manager as fm
from os.path import join 
from scipy.stats import norm
import scipy.sparse as sp
from scipy.integrate import cumulative_trapezoid
import ot  # POT: Python Optimal Transport
from tqdm import tqdm


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



def compute_T1(delr, Nbins):
    """
    Args:
        delr (float): bin width
        Nbins (int): number of bins

    Returns:
        T1 (array): T1 matrix of size Nbins x Nbins
    """
    T1 = np.zeros((Nbins,Nbins)) 
    for col in range(1, Nbins): # First column is all zeros
        value = (col*delr)**2
        T1[col, col] = value
        T1[col+1:, col] = 2*value
    return T1

def compute_T2(delr, Nbins, w):
    """
    Args:
        delr (float): bin width
        Nbins (int): number of bins

    Returns:
        T2 (array): T2 matrix of size Nbins x Nbins
    """
    m_KL = np.ceil(2*w/delr).astype(int) # number of bins to average over
    k_KL = 2*m_KL-1
    fractions = np.zeros((1,k_KL))
    A1_block = sp.identity(m_KL, format='csr')
    A2_block = sp.lil_matrix((m_KL, Nbins-m_KL))
    fractions[0,m_KL-1:] = norm.cdf(((np.arange(0,m_KL)+0.5)*delr),0,w)-norm.cdf(((np.arange(0,m_KL)-0.5)*delr),0,w)        
    fractions[0,:m_KL-1] = np.flip(fractions[0,m_KL:2*m_KL-1])
    fractions[0, :] *= 1/np.sum(fractions)
    B_block = sp.diags(np.tile(fractions, (Nbins-2*m_KL, 1)).T, np.arange(0, 2*(m_KL-1)+1), shape=(Nbins-2*m_KL, Nbins))
    T2 = sp.vstack((sp.hstack((A1_block, A2_block)), B_block, sp.hstack((A2_block, A1_block))))
    return T2

def compute_T3(delr, Nbins):
    """
    Args:
        delr (float): bin width
        Nbins (int): number of bins
        density (float): density of the system

    Returns:
        T3 (array): T3 matrix of size Nbins x Nbins
    """
    T3 = np.zeros((Nbins,Nbins))
    constant = 1/(delr**2)
    for row in range(1, Nbins):
        T3[row, row] = constant/(row)**2
        factor = 2*constant/(row)**2
        sign = 1 - 2 * (row & 1)
        for col in range(row):
            T3[row, col] = sign*factor
            sign *= -1
    return T3


def KAMEL_LOBE(r,RDF,w=0.015):
    """
    Args:
        r (array): vector of equispaced radii at which RDF is evaluated
        RDF (array): vector of corresponding RDF values
        varargin (float, optional): width of Gaussian kernel (set to 0.015 by default)

    Returns:
        r_tilde (array): vector of equispaced radii at which KAMEL-LOBE RDF is evaluated
        gr_tilde (array): vector of corresponding KAMEL-LOBE RDF values
    """
    Nbins = RDF.shape[0] # number of bins
    delr = r[1]-r[0] # bin width, MATLAB version uses r[2]-r[1]
    m_KL = np.ceil(2*w/delr).astype(int) # number of bins to average over

    if m_KL > 1:
        T1 = compute_T1(delr, Nbins)
        T2 = compute_T2(delr, Nbins, w)
        T3 = compute_T3(delr, Nbins)

        # Computing gr_tilde
        gr_convert = T3@(T2@(T1@RDF)) # Explicit operation order to reduce redundant matrix multiplications
        gr_tilde = gr_convert[:-2*m_KL]    
        r_tilde = r[:-2*m_KL]
        return r_tilde, gr_tilde
    
    else:
        gr_tilde = RDF
        r_tilde = r
        print('w <= delr/2, no averaging is performed')
        return r_tilde, gr_tilde


def earth_movers_distance(array, ref, distances, rho=1):
    """
    Compute the Earth Mover's Distance (EMD) between two distributions.
    """
    del_r = (distances[1]-distances[0])/2 
    factors =  (distances+del_r)**3-(distances-del_r)**3
    deltas = np.abs(array - ref)
    EMD = (4/3)*np.pi*rho*np.sum(deltas * factors)
    return EMD


def s2_integrand(g_r, r, rho=1, tol=1e-8):
    g_r[np.where(g_r < tol)] = 0.0
    s2 = np.ones_like(g_r)
    s2[g_r > tol] = (g_r*np.log(g_r) - g_r + 1)[g_r > tol]
    return -2*np.pi*rho*s2*r**2



def compute_s2_emd(g1, g2, r, reg=1e-1, reg_m=1e-1, tol=1e-8):
    """
    Compute the S₂‐weighted Earth Mover’s Distance between two RDFs g1 and g2
    sampled on the same radial grid r, handling g=0 by setting phi(0)=0.
    (g(r)*ln(g(r)) - g(r) + 1) is the S₂‐mass density.
    """
    dr = r[1] - r[0]
    # Safe phi: zero where g==0
    def phi(g):
        return np.where(g > tol, g * np.log(g) - g + 1, 1)

    # Build S₂‐mass vectors
    mu1 = phi(g1) * r**2 * dr
    mu2 = phi(g2) * r**2 * dr

    # Ground‐distance matrix
    M = ot.dist(r.reshape(-1, 1), r.reshape(-1, 1), metric='euclidean')

    # Compute and return EMD
    # returns a transport matrix gamma of shape (N,N)
    gamma = ot.unbalanced.sinkhorn_unbalanced(mu1, mu2, M, reg, reg_m)

    # the total cost is sum(gamma * M)
    return np.sum(gamma * M)




def compute_rdf_particle_emd_with_creation_penalty(g1, g2, r, rho, D=1.0, tol=1e-8):
    """
    Compute the EMD between two RDF‐derived particle‐count histograms,
    with any excess/deficit particles created/destroyed at unit cost D=1.

    Parameters
    ----------
    g1, g2 : array_like, shape (N,)
        RDF values on the same radial grid r.
    r      : array_like, shape (N,)
        Radial bin centers.
    rho    : float
        Number density of particles.
    D      : float, optional (default=1.0)
        Cost to create or destroy one particle.

    Returns
    -------
    avg_disp   : float
        Average radial displacement per particle (Å).
    total_cost : float
        Total transport cost (Å·particles + D·particles_created).
    """

    # 0) Check for negative 0 values due to KL transform 
    g1[np.where(g1 < -tol)] = 0.0
    g2[np.where(g2 < -tol)] = 0.0


    # 1) Bin width
    dr = r[1] - r[0]

    # 2) Shell counts: dN = 4πρ·r²·g(r)·dr
    shell1 = 4 * np.pi * rho * r**2 * g1 * dr
    shell2 = 4 * np.pi * rho * r**2 * g2 * dr

    # 3) Totals
    N1 = shell1.sum()
    N2 = shell2.sum()

    # 4) Extend to include a dummy bin for creation/destruction
    if N1 < N2:
        delta    = N2 - N1
        a_ext    = np.hstack([shell1, delta])
        b_ext    = np.hstack([shell2,    0.0 ])
    else:
        delta    = N1 - N2
        a_ext    = np.hstack([shell1,    0.0 ])
        b_ext    = np.hstack([shell2, delta])

    # 5) Build extended cost matrix
    N = len(r)
    # cost of moving between real bins: |r_i - r_j|
    M = ot.dist(r.reshape(-1,1), r.reshape(-1,1), metric='euclidean')
    # now make (N+1)x(N+1) and set dummy‐bin costs = D
    M_ext = np.zeros((N+1, N+1))
    M_ext[:N, :N] = M
    M_ext[:N,  N ] = D
    M_ext[ N, :N] = D
    # M_ext[N,N] = 0 already

    # 6) Compute balanced EMD on the extended histograms
    flow      = ot.emd(a_ext, b_ext, M_ext)
    total_cost = np.sum(flow * M_ext)

    # 7) Average per‐particle displacement
    avg_disp   = total_cost / max(N1, N2)

    return avg_disp, total_cost


def compute_rdf_emd_to_next_bin(g1, g2, r, rho, reg=None):
    """
    Balanced EMD that moves any excess/deficit particles out to one bin
    beyond the cutoff (r[-1] + dr).

    Parameters
    ----------
    g1, g2 : array_like, shape (N,)
        Two RDFs on the same radial grid r.
    r      : array_like, shape (N,)
        Monotonic radial bin centers; r[-1] is the cutoff.
    rho    : float
        Number density.
    reg    : float or None
        If None, uses exact EMD. If >0, uses entropic Sinkhorn(reg) but still mass-conserving.

    Returns
    -------
    avg_disp   : float
        Average radial displacement per particle.
    total_cost : float
        Total transport cost = ∑flow·distance.
    """
    dr = r[1] - r[0]
    r_dummy = r[-1] + dr


    # 0) Check for negative 0 values due to KL transform
    g1[np.where(g1 < 0)] = 0.0
    g2[np.where(g2 < 0)] = 0.0
    
    # 1) compute counts in each shell
    shell1 = 4*np.pi * rho * r**2 * g1 * dr
    shell2 = 4*np.pi * rho * r**2 * g2 * dr

    N1, N2 = shell1.sum(), shell2.sum()

    # 2) append one dummy bin to absorb any excess
    # 2) append one dummy bin to absorb any excess
    if N1 < N2:
        # bring a_ext up to N2
        a_ext = np.hstack([shell1, N2 - N1])
        b_ext = np.hstack([shell2,       0.0])
        N_tot = N2
    else:
        # bring b_ext up to N1
        a_ext = np.hstack([shell1,       0.0])
        b_ext = np.hstack([shell2, N1 - N2])
        N_tot = N1

    # 3) build extended r vector and cost matrix
    r_ext = np.concatenate([r, [r_dummy]])
    M_ext = ot.dist(r_ext.reshape(-1,1), r_ext.reshape(-1,1), metric='euclidean')

    # 4) solve balanced OT
    if reg is None:
        flow = ot.emd(a_ext, b_ext, M_ext)
    else:
        # Sinkhorn wants probability vectors, so normalize then scale back:
        a_prob = a_ext / N_tot
        b_prob = b_ext / N_tot
        flow = ot.sinkhorn(a_prob, b_prob, M_ext, reg) * N_tot

    total_cost = np.sum(flow * M_ext)
    avg_disp   = total_cost / N_tot

    return avg_disp, total_cost


def emd_rdf_with_single_pad(g1, g2, r, rho, tol=1e-12):
    """
    Earth-Mover Distance between two RDF histograms, using one dummy
    bin placed just outside r_max to absorb any mass mismatch.

    Parameters
    ----------
    g1, g2 : array_like, shape (N,)
        Two RDFs evaluated at the same radial grid centres r.
    r      : array_like, shape (N,)
        Radial bin centres (ascending, uniform spacing).
    rho    : float
        Number density.
    tol    : float, optional
        Numerical tolerance for deciding the histograms are balanced.

    Returns
    -------
    avg_disp   : float
        Average radial displacement per interacting pair.
    total_cost : float
        Transport cost (= displacement × pairs moved).
    """
    # 1) shell-pair counts  n_i = 4πρ r_i² g(r_i) Δr
    dr      = r[1] - r[0]
    shell1  = 4 * np.pi * rho * r**2 * g1 * dr
    shell2  = 4 * np.pi * rho * r**2 * g2 * dr

    diff    = shell1.sum() - shell2.sum()      # + → shell1 has more mass
    if abs(diff) < tol:                        # already balanced
        a_ext, b_ext = shell1, shell2
        r_ext        = r
    else:
        pad_mass = abs(diff)                   # amount to add
        r_pad    = r[-1] + dr                  # dummy bin centre

        if diff > 0:                           # pad shell2
            a_ext = np.hstack([shell1, 0.0])
            b_ext = np.hstack([shell2, pad_mass])
        else:                                  # pad shell1
            a_ext = np.hstack([shell1, pad_mass])
            b_ext = np.hstack([shell2, 0.0])

        r_ext = np.hstack([r, r_pad])

    # 2) probability vectors
    N_tot  = a_ext.sum()                       # == b_ext.sum()
    a_prob = a_ext / N_tot
    b_prob = b_ext / N_tot

    # 3) cost matrix  |r_i − r_j|
    M = np.abs(r_ext[:, None] - r_ext[None, :])

    γ          = ot.emd(a_prob, b_prob, M)
    total_cost = (γ * M).sum() * N_tot
    avg_disp   = total_cost / N_tot
    return avg_disp, total_cost



def compute_rdf_emd_with_adaptive_capacity(
    g1, g2, r, rho, vacancy_fn, tol=1e-8
):
    """
    EMD between two RDF histograms, adaptively adding overflow bins
    until excess mass is accommodated based on vacancy_fn predictions.

    Parameters
    ----------
    g1, g2 : array_like, shape (N,)
        Two RDFs on the same radial grid r.
    r      : array_like, shape (N,)
        Radial bin centers.
    rho    : float
        Number density.
    vacancy_fn : callable
        vacancy_fn(r) -> fraction [0,1] of free spots in shell at radius r.
    tol    : float, optional
        Tolerance for treating values as zero.

    Returns
    -------
    avg_disp   : float
        Average radial displacement per particle.
    total_cost : float
        Total transport cost (distance · particles).
    """
    # 0) Bin width & clip negatives
    dr = r[1] - r[0]
    g1 = np.where(g1 < tol, 0.0, g1)
    g2 = np.where(g2 < tol, 0.0, g2)

    # 1) Compute shell counts
    shell1 = 4 * np.pi * rho * r**2 * g1 * dr
    shell2 = 4 * np.pi * rho * r**2 * g2 * dr
    N1, N2 = shell1.sum(), shell2.sum()

    # 2) Compute excess mass to allocate
    mass_excess = max(0.0, N1 - N2)

    # 3) Adaptively build overflow bins until capacity ≥ excess
    r_cut = r[-1]
    r_over = []
    cap_over = []
    cumulative = 0.0
    i = 1
    while cumulative < mass_excess:
        r_i = r_cut + i * dr
        cap_i = vacancy_fn(r_i) * (4 * np.pi * rho * r_i**2 * dr)
        r_over.append(r_i)
        cap_over.append(cap_i)
        cumulative += cap_i
        i += 1

    r_over = np.array(r_over)
    cap_over = np.array(cap_over)

    # 4) Build extended histograms
    a_ext = np.hstack([shell1, np.zeros_like(cap_over)])
    b_ext = np.hstack([shell2, cap_over])

    # 5) Check total capacity
    if a_ext.sum() > b_ext.sum() + tol:
        raise ValueError(
            "Insufficient overflow capacity; adjust vacancy_fn or parameters"
        )

    # 6) Normalize to probabilities
    N_tot = a_ext.sum()
    a_prob = a_ext / N_tot
    b_prob = b_ext / N_tot

    # 7) Build cost matrix
    r_ext = np.concatenate([r, r_over])
    M_ext = np.abs(r_ext.reshape(-1,1) - r_ext.reshape(1,-1))

    # 8) Compute balanced EMD
    gamma = ot.emd(a_prob, b_prob, M_ext)
    total_cost = np.sum(gamma * M_ext) * N_tot
    avg_disp = total_cost / N_tot

    return avg_disp, total_cost



from typing import Tuple, Optional, Union


def compute_s2_emd_with_entropy_penalty(
    g1: np.ndarray,
    g2: np.ndarray,
    r: np.ndarray,
    D: float = 1.0,
    reg: Optional[float] = None,
    tol: float = 1e-8
) -> Tuple[float, float]:
    """
    Compute a balanced Earth Mover's Distance (EMD) between the S₂-integrand distributions of two RDFs,
    allowing creation/destruction of entropy with penalty D.
    
    The S₂-integrand φ(g) = g·ln(g) - g + 1 is used to generate mass distributions
    discretized as μ_i(k) = φ(g_i(r_k))·r_k^2·Δr.
    
    Any net excess entropy (sum μ1 - sum μ2) is absorbed into a dummy bin
    at r_dummy = r[-1] + Δr, with cost D per unit S₂-mass.
    
    Parameters
    ----------
    g1, g2 : np.ndarray, shape (N,)
        RDF values on the same radial grid r.
    r : np.ndarray, shape (N,)
        Radial bin centers.
    D : float, default=1.0
        Cost to create or destroy one unit of S₂-mass.
    reg : float or None, default=None
        If None, uses exact EMD; if >0, uses entropic Sinkhorn with regularization reg.
    tol : float, default=1e-8
        Tolerance for clipping small or negative g-values.
        
    Returns
    -------
    avg_cost : float
        Average S₂-mass displacement per unit (in units of r).
    total_cost : float
        Total transport cost (r·S₂-mass + D·mass_created).
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes.
    """
    # Input validation
    g1 = np.asarray(g1)
    g2 = np.asarray(g2)
    r = np.asarray(r)
    
    if g1.shape != r.shape or g2.shape != r.shape:
        raise ValueError(f"Shape mismatch: g1 {g1.shape}, g2 {g2.shape}, r {r.shape}")
    
    if not r.size >= 2:
        raise ValueError("Radial grid must have at least 2 points")
    
    # 1) Calculate bin width
    dr = r[1] - r[0]
    
    # Check for uniform grid
    if not np.allclose(np.diff(r), dr):
        raise ValueError("Radial grid must be uniform")
    
    # 2) Clip small/negative values
    g1_clipped = np.maximum(g1, 0.0)
    g2_clipped = np.maximum(g2, 0.0)
    
    # 3) Define and compute φ(g) = g·ln(g) - g + 1
    def phi(g: np.ndarray) -> np.ndarray:
        """Calculate the S₂-integrand for an RDF."""
        result = np.ones_like(g)
        mask = g > tol
        g_masked = g[mask]
        result[mask] = g_masked * np.log(g_masked) - g_masked + 1
        return result
    
    phi_1 = phi(g1_clipped)
    phi_2 = phi(g2_clipped)
    
    # 4) Build S2 mass vectors
    mu1 = phi_1 * r**2 * dr
    mu2 = phi_2 * r**2 * dr
    N1, N2 = np.sum(mu1), np.sum(mu2)
    
    # 5) Append dummy bin for creation/destruction
    r_dummy = r[-1] + dr
    
    if N1 < N2:
        # Create S2-mass to bring mu1 up to mu2
        a_ext = np.append(mu1, N2 - N1)
        b_ext = np.append(mu2, 0.0)
        N_tot = N2
    else:
        # Destroy S2-mass to bring mu2 up to mu1
        a_ext = np.append(mu1, 0.0)
        b_ext = np.append(mu2, N1 - N2)
        N_tot = N1
    
    # 6) Build extended r-vector and cost matrix
    r_ext = np.append(r, r_dummy)
    M = np.abs(r_ext[:, np.newaxis] - r_ext[np.newaxis, :])
    
    # Set dummy creation/destruction cost = D
    N = len(r)
    M[N, :N] = D
    M[:N, N] = D
    
    # 7) Solve EMD
    if reg is None:
        # Use exact EMD
        flow = ot.emd(a_ext, b_ext, M)
    else:
        # Use regularized Sinkhorn algorithm
        if reg <= 0:
            raise ValueError("Regularization parameter must be positive")
        a_prob = a_ext / N_tot
        b_prob = b_ext / N_tot
        flow = ot.sinkhorn(a_prob, b_prob, M, reg) * N_tot
    
    # 8) Compute costs
    total_cost = np.sum(flow * M)
    avg_cost = total_cost / N_tot
    
    return avg_cost, total_cost

@njit
def create_s2_emd_matrices(g1, g2, r, rho, tol=1e-12):
    mask1 = g1 > tol
    mask2 = g2 > tol
    s1 = np.ones_like(g1)
    s2 = np.ones_like(g2)
    s1[mask1] = g1[mask1] * np.log(g1[mask1]) - g1[mask1] + 1
    s2[mask2] = g2[mask2] * np.log(g2[mask2]) - g2[mask2] + 1
    s1 *= -2 * np.pi * rho * r**2 
    s2 *= -2 * np.pi * rho * r**2
    
    s1 = np.abs(s1)
    s2 = np.abs(s2)

    dr     = r[1] - r[0]
    r_pad  = np.array([r[-1] + dr])           # put the dummy bin just outside the grid
    diff   = np.sum(s1) - np.sum(s2)  # positive ⇒ s1 has more mass
    if np.abs(diff) < tol:           # already balanced
        a_ext, b_ext = s1, s2
        r_ext        = r
    else:
        pad = np.array([np.abs(diff)])          # amount we must add to the lighter side
        if diff > 0:              # s2 needs the pad
            a_ext = np.hstack((s1, np.zeros(1)))
            b_ext = np.hstack((s2, pad))
        else:          
            a_ext = np.hstack((s1, pad))
            b_ext = np.hstack((s2, np.zeros(1)))
        r_ext = np.hstack((r, r_pad))
    # normalise to probability vectors
    N = np.sum(a_ext)
    a_prob = a_ext / N
    b_prob = b_ext / N

    # cost matrix (radial distance)
    M = np.abs(r_ext[:, None] - r_ext[None, :])

    return a_prob, b_prob, M, N

def balanced_emd_s2(g1, g2, r, rho, tol=1e-12):
    """
    EMD between two non-negative histograms of the s₂ integrand.
    The smaller distribution is padded with one 'dummy' bin at r = r_max + Δr
    so that the two sums match exactly.
    """
    a_prob, b_prob, M, N = create_s2_emd_matrices(g1, g2, r, rho, tol)
    γ = ot.emd(a_prob, b_prob, M, numItermax=1000000)
    total_cost = (γ * M).sum() * N
    avg_disp   = total_cost / N
    return avg_disp, total_cost



def compute_KL(p, q, tol=1e-12):
    """
    Compute the Kullback-Leibler divergence D_KL(p || q)
    between two discrete probability distributions p and q.
    Small values below tol are clipped to avoid log(0).
    """

    p_mask = p > tol
    q_mask = q > tol
    mask = p_mask & q_mask
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))




mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman','Palatino','Georgia'],
    'font.size':         12,     # base font size
    'axes.titlesize':    16,
    'axes.labelsize':    16,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'axes.linewidth':    0.5,    # thicker axis lines
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    # 2) pick a clean color cycle (e.g. Colorblind‐friendly)
    'axes.prop_cycle':   mpl.cycler('color', mpl.cm.tab10.colors),
})


rdf_4_160 = np.load('rdfs_4_160.npy')
dists_4_160 = np.load('dist_4_160.npy')

rdf_4_400 = np.load('rdfs_4_400.npy')
dists_4_400 = np.load('dist_4_400.npy')

rdf_4_1600 = np.load('rdfs_4_1600.npy')
dists_4_1600 = np.load('dist_4_1600.npy')


rdf_equil_4_160 = np.mean(parseRDF('rdf_equil_4_160.data')[-1], axis=0)
rdf_equil_4_400 = np.mean(parseRDF('rdf_equil_4_400.data')[-1], axis=0)
rdf_equil_4_1600 = np.mean(parseRDF('rdf_equil_4_1600.data')[-1], axis=0)



rdf_equil_4_160_randoms = parseRDF('rdf_equil_4_160_random.data')[-1]
rdf_equil_4_400_randoms = parseRDF('rdf_equil_4_400_random.data')[-1]
rdf_equil_4_1600_randoms = parseRDF('rdf_equil_4_1600_random.data')[-1]


KL_rdf_equil_4_160 = KAMEL_LOBE(dists_4_160, rdf_equil_4_160)[1]
KL_rdf_equil_4_400 = KAMEL_LOBE(dists_4_400, rdf_equil_4_400)[1]
KL_rdf_equil_4_1600 = KAMEL_LOBE(dists_4_1600, rdf_equil_4_1600)[1]


KL_rdf_equil_4_160_randoms = np.load('KL_rdf_equil_4_160_randoms.npy')
KL_rdf_equil_4_400_randoms = np.load('KL_rdf_equil_4_400_randoms.npy')
KL_rdf_equil_4_1600_randoms = np.load('KL_rdf_equil_4_1600_randoms.npy')

KL_rdf_4_160 = np.load('KL_rdfs_4_160.npy')
KL_dist_4_160 = np.load('KLs_dist_4_160.npy')

KL_rdf_4_400 = np.load('KL_rdfs_4_400.npy')
KL_dist_4_400 = np.load('KLs_dist_4_400.npy')

KL_rdf_4_1600 = np.load('KL_rdfs_4_1600.npy')
KL_dist_4_1600 = np.load('KLs_dist_4_1600.npy')

EMDS_4_160 = np.load('EMDS_4_160.npy')
EMDS_KL_4_160 = np.load('EMDS_KL_4_160.npy')
Totals_4_160 = np.load('Totals_4_160.npy')
Totals_KL_4_160 = np.load('Totals_KL_4_160.npy')

EMDS_4_400 = np.load('EMDS_4_400.npy')
EMDS_KL_4_400 = np.load('EMDS_KL_4_400.npy')
Totals_4_400 = np.load('Totals_4_400.npy')
Totals_KL_4_400 = np.load('Totals_KL_4_400.npy')

EMDS_4_1600 = np.load('EMDS_4_1600.npy')
EMDS_KL_4_1600 = np.load('EMDS_KL_4_1600.npy')
Totals_4_1600 = np.load('Totals_4_1600.npy')
Totals_KL_4_1600 = np.load('Totals_KL_4_1600.npy')

dist_4_160 = np.load('dist_4_160.npy')
dist_4_400 = np.load('dist_4_400.npy')
dist_4_1600 = np.load('dist_4_1600.npy')

dist_KL_4_160 = np.load('KLs_dist_4_160.npy')
dist_KL_4_400 = np.load('KLs_dist_4_400.npy')
dist_KL_4_1600 = np.load('KLs_dist_4_1600.npy')


EMD_4_160 = np.load('EMD_4_160_equil.npy')
EMD_4_400 = np.load('EMD_4_400_equil.npy')
EMD_4_1600_equil = np.load('EMD_4_1600_equil.npy')

EMD_4_160_equil_mean = np.mean(EMD_4_160)
EMD_4_400_equil_mean = np.mean(EMD_4_400)
EMD_4_1600_equil_mean = np.mean(EMD_4_1600_equil)

EMD_4_160_equil_std = np.std(EMD_4_160)
EMD_4_400_equil_std = np.std(EMD_4_400)
EMD_4_1600_equil_std = np.std(EMD_4_1600_equil)



KL_EMD_4_160 = np.load('KL_EMD_4_160_equil.npy')
KL_EMD_4_400 = np.load('KL_EMD_4_400_equil.npy')
KL_EMD_4_1600 = np.load('KL_EMD_4_1600_equil.npy')

KL_EMD_4_160_equil_mean = np.mean(KL_EMD_4_160)
KL_EMD_4_400_equil_mean = np.mean(KL_EMD_4_400)
KL_EMD_4_1600_equil_mean = np.mean(KL_EMD_4_1600)

KL_EMD_4_160_equil_std = np.std(KL_EMD_4_160)
KL_EMD_4_400_equil_std = np.std(KL_EMD_4_400)
KL_EMD_4_1600_equil_std = np.std(KL_EMD_4_1600)


S2_EMDS_4_160 = np.load('S2_EMDS_4_160.npy')
S2_EMDS_KL_4_160 = np.load('S2_EMDS_KL_4_160.npy')
S2_Totals_4_160 = np.load('S2_Totals_4_160.npy')
S2_Totals_KL_4_160 = np.load('S2_Totals_KL_4_160.npy')

S2_EMDS_4_160_balanced = np.load('S2_EMDS_4_160_balanced.npy')
S2_EMDS_KL_4_160_balanced = np.load('S2_EMDS_KL_4_160_balanced.npy')


S2_EMDS_4_400 = np.load('S2_EMDS_4_400.npy')
S2_EMDS_KL_4_400 = np.load('S2_EMDS_KL_4_400.npy')
S2_Totals_4_400 = np.load('S2_Totals_4_400.npy')
S2_Totals_KL_4_400 = np.load('S2_Totals_KL_4_400.npy')
S2_EMDS_4_400_balanced = np.load('S2_EMDS_4_400_balanced.npy')
S2_EMDS_KL_4_400_balanced = np.load('S2_EMDS_KL_4_400_balanced.npy')

S2_EMDS_4_1600 = np.load('S2_EMDS_4_1600.npy')
S2_EMDS_KL_4_1600 = np.load('S2_EMDS_KL_4_1600.npy')
S2_Totals_4_1600 = np.load('S2_Totals_4_1600.npy')
S2_Totals_KL_4_1600 = np.load('S2_Totals_KL_4_1600.npy')
S2_EMDS_4_1600_balanced = np.load('S2_EMDS_4_1600_balanced.npy')
S2_EMDS_KL_4_1600_balanced = np.load('S2_EMDS_KL_4_1600_balanced.npy')

S2_EMDS_4_160_equil = np.load('S2_EMD_4_160_equil.npy')
S2_EMDS_4_400_equil = np.load('S2_EMD_4_400_equil.npy')
S2_EMDS_4_1600_equil = np.load('S2_EMD_4_1600_equil.npy')
S2_EMDS_4_160_equil_balanced = np.load('S2_EMD_4_160_equil_balanced.npy')
S2_EMDS_4_400_equil_balanced = np.load('S2_EMD_4_400_equil_balanced.npy')
S2_EMDS_4_1600_equil_balanced = np.load('S2_EMD_4_1600_equil_balanced.npy')


S2_EMDS_KL_4_160_equil = np.load('KL_S2_EMD_4_160_equil.npy')
S2_EMDS_KL_4_400_equil = np.load('KL_S2_EMD_4_400_equil.npy')
S2_EMDS_KL_4_1600_equil = np.load('KL_S2_EMD_4_1600_equil.npy')
S2_EMDS_KL_4_160_equil_balanced = np.load('KL_S2_EMD_4_160_equil_balanced.npy')
S2_EMDS_KL_4_400_equil_balanced = np.load('KL_S2_EMD_4_400_equil_balanced.npy')
S2_EMDS_KL_4_1600_equil_balanced = np.load('KL_S2_EMD_4_1600_equil_balanced.npy')



S2_EMDS_4_160_equil_avg = np.mean(S2_EMDS_4_160_equil)
S2_EMDS_4_400_equil_avg = np.mean(S2_EMDS_4_400_equil)
S2_EMDS_4_1600_equil_avg = np.mean(S2_EMDS_4_1600_equil)
S2_EMDS_4_160_equil_balanced_avg = np.mean(S2_EMDS_4_160_equil_balanced)
S2_EMDS_4_400_equil_balanced_avg = np.mean(S2_EMDS_4_400_equil_balanced)
S2_EMDS_4_1600_equil_balanced_avg = np.mean(S2_EMDS_4_1600_equil_balanced)

S2_EMDS_4_160_equil_std = np.std(S2_EMDS_4_160_equil)
S2_EMDS_4_400_equil_std = np.std(S2_EMDS_4_400_equil)
S2_EMDS_4_1600_equil_std = np.std(S2_EMDS_4_1600_equil)
S2_EMDS_4_160_equil_balanced_std = np.std(S2_EMDS_4_160_equil_balanced)
S2_EMDS_4_400_equil_balanced_std = np.std(S2_EMDS_4_400_equil_balanced)
S2_EMDS_4_1600_equil_balanced_std = np.std(S2_EMDS_4_1600_equil_balanced)


S2_EMDS_KL_4_160_equil_avg = np.mean(S2_EMDS_KL_4_160_equil)
S2_EMDS_KL_4_400_equil_avg = np.mean(S2_EMDS_KL_4_400_equil)
S2_EMDS_KL_4_1600_equil_avg = np.mean(S2_EMDS_KL_4_1600_equil)
S2_EMDS_KL_4_160_equil_balanced_avg = np.mean(S2_EMDS_KL_4_160_equil_balanced)
S2_EMDS_KL_4_400_equil_balanced_avg = np.mean(S2_EMDS_KL_4_400_equil_balanced)
S2_EMDS_KL_4_1600_equil_balanced_avg = np.mean(S2_EMDS_KL_4_1600_equil_balanced)

S2_EMDS_KL_4_160_equil_std = np.std(S2_EMDS_KL_4_160_equil)
S2_EMDS_KL_4_400_equil_std = np.std(S2_EMDS_KL_4_400_equil)
S2_EMDS_KL_4_1600_equil_std = np.std(S2_EMDS_KL_4_1600_equil)
S2_EMDS_KL_4_160_equil_balanced_std = np.std(S2_EMDS_KL_4_160_equil_balanced)
S2_EMDS_KL_4_400_equil_balanced_std = np.std(S2_EMDS_KL_4_400_equil_balanced)
S2_EMDS_KL_4_1600_equil_balanced_std = np.std(S2_EMDS_KL_4_1600_equil_balanced)



EMDS_4_160_avg = np.mean(EMDS_4_160, axis=0)
EMDS_KL_4_160_avg = np.mean(EMDS_KL_4_160, axis=0)
Totals_4_160_avg = np.mean(Totals_4_160, axis=0)
Totals_KL_4_160_avg = np.mean(Totals_KL_4_160, axis=0)



EMDS_4_400_avg = np.mean(EMDS_4_400, axis=0)
EMDS_KL_4_400_avg = np.mean(EMDS_KL_4_400, axis=0)
Totals_4_400_avg = np.mean(Totals_4_400, axis=0)
Totals_KL_4_400_avg = np.mean(Totals_KL_4_400, axis=0)


EMDS_4_1600_avg = np.mean(EMDS_4_1600, axis=0)
EMDS_KL_4_1600_avg = np.mean(EMDS_KL_4_1600, axis=0)
Totals_4_1600_avg = np.mean(Totals_4_1600, axis=0)
Totals_KL_4_1600_avg = np.mean(Totals_KL_4_1600, axis=0)


EMDS_4_160_std = np.std(EMDS_4_160, axis=0)
EMDS_KL_4_160_std = np.std(EMDS_KL_4_160, axis=0)
Totals_4_160_std = np.std(Totals_4_160, axis=0)
Totals_KL_4_160_std = np.std(Totals_KL_4_160, axis=0)
EMDS_4_400_std = np.std(EMDS_4_400, axis=0)
EMDS_KL_4_400_std = np.std(EMDS_KL_4_400, axis=0)
Totals_4_400_std = np.std(Totals_4_400, axis=0)
Totals_KL_4_400_std = np.std(Totals_KL_4_400, axis=0)
EMDS_4_1600_std = np.std(EMDS_4_1600, axis=0)
EMDS_KL_4_1600_std = np.std(EMDS_KL_4_1600, axis=0)
Totals_4_1600_std = np.std(Totals_4_1600, axis=0)
Totals_KL_4_1600_std = np.std(Totals_KL_4_1600, axis=0)
EMDS_4_160_balanced_std = np.std(S2_EMDS_4_160_balanced, axis=0)
EMDS_KL_4_160_balanced_std = np.std(S2_EMDS_KL_4_160_balanced, axis=0)
EMDS_4_400_balanced_std = np.std(S2_EMDS_4_400_balanced, axis=0)
EMDS_KL_4_400_balanced_std = np.std(S2_EMDS_KL_4_400_balanced, axis=0)
EMDS_4_1600_balanced_std = np.std(S2_EMDS_4_1600_balanced, axis=0)
EMDS_KL_4_1600_balanced_std = np.std(S2_EMDS_KL_4_1600_balanced, axis=0)


S2_EMDS_4_160_avg = np.mean(S2_EMDS_4_160, axis=0)
S2_EMDS_KL_4_160_avg = np.mean(S2_EMDS_KL_4_160, axis=0)
S2_Totals_4_160_avg = np.mean(S2_Totals_4_160, axis=0)
S2_Totals_KL_4_160_avg = np.mean(S2_Totals_KL_4_160, axis=0)
S2_EMDS_4_400_avg = np.mean(S2_EMDS_4_400, axis=0)
S2_EMDS_KL_4_400_avg = np.mean(S2_EMDS_KL_4_400, axis=0)
S2_Totals_4_400_avg = np.mean(S2_Totals_4_400, axis=0)
S2_Totals_KL_4_400_avg = np.mean(S2_Totals_KL_4_400, axis=0)
S2_EMDS_4_1600_avg = np.mean(S2_EMDS_4_1600, axis=0)
S2_EMDS_KL_4_1600_avg = np.mean(S2_EMDS_KL_4_1600, axis=0)
S2_Totals_4_1600_avg = np.mean(S2_Totals_4_1600, axis=0)
S2_Totals_KL_4_1600_avg = np.mean(S2_Totals_KL_4_1600, axis=0)
S2_EMDS_4_160_std = np.std(S2_EMDS_4_160, axis=0)
S2_EMDS_KL_4_160_std = np.std(S2_EMDS_KL_4_160, axis=0)
S2_Totals_4_160_std = np.std(S2_Totals_4_160, axis=0)
S2_Totals_KL_4_160_std = np.std(S2_Totals_KL_4_160, axis=0)
S2_EMDS_4_400_std = np.std(S2_EMDS_4_400, axis=0)
S2_EMDS_KL_4_400_std = np.std(S2_EMDS_KL_4_400, axis=0)
S2_Totals_4_400_std = np.std(S2_Totals_4_400, axis=0)
S2_Totals_KL_4_400_std = np.std(S2_Totals_KL_4_400, axis=0)
S2_EMDS_4_1600_std = np.std(S2_EMDS_4_1600, axis=0)
S2_EMDS_KL_4_1600_std = np.std(S2_EMDS_KL_4_1600, axis=0)
S2_Totals_4_1600_std = np.std(S2_Totals_4_1600, axis=0)
S2_Totals_KL_4_1600_std = np.std(S2_Totals_KL_4_1600, axis=0)
S2_EMDS_4_160_balanced_avg = np.mean(S2_EMDS_4_160_balanced, axis=0)
S2_EMDS_KL_4_160_balanced_avg = np.mean(S2_EMDS_KL_4_160_balanced, axis=0)
S2_EMDS_4_400_balanced_avg = np.mean(S2_EMDS_4_400_balanced, axis=0)
S2_EMDS_KL_4_400_balanced_avg = np.mean(S2_EMDS_KL_4_400_balanced, axis=0)
S2_EMDS_4_1600_balanced_avg = np.mean(S2_EMDS_4_1600_balanced, axis=0)
S2_EMDS_KL_4_1600_balanced_avg = np.mean(S2_EMDS_KL_4_1600_balanced, axis=0)
S2_EMDS_4_160_balanced_std = np.std(S2_EMDS_4_160_balanced, axis=0)
S2_EMDS_KL_4_160_balanced_std = np.std(S2_EMDS_KL_4_160_balanced, axis=0)
S2_EMDS_4_400_balanced_std = np.std(S2_EMDS_4_400_balanced, axis=0)
S2_EMDS_KL_4_400_balanced_std = np.std(S2_EMDS_KL_4_400_balanced, axis=0)
S2_EMDS_4_1600_balanced_std = np.std(S2_EMDS_4_1600_balanced, axis=0)
S2_EMDS_KL_4_1600_balanced_std = np.std(S2_EMDS_KL_4_1600_balanced, axis=0)


times = np.arange(0, len(EMDS_4_160_avg))*0.02


rdf_4_160_ave = np.mean(rdf_4_160, axis=0)
rdf_4_400_ave = np.mean(rdf_4_400, axis=0)
rdf_4_1600_ave = np.mean(rdf_4_1600, axis=0)

rdf_4_160_ave_KL = np.mean(KL_rdf_4_160, axis=0)
rdf_4_400_ave_KL = np.mean(KL_rdf_4_400, axis=0)
rdf_4_1600_ave_KL = np.mean(KL_rdf_4_1600, axis=0)




bootstraps = 1000
D_ref = 0.0512
N = rdf_4_160.shape[0]

errors_4_160 = []
errors_4_160_std = [] 

errors_4_160_KL = []
errors_4_160_KL_std = []

for idx in range(rdf_4_160_ave.shape[0]):
    rdf_4_160_choices = np.random.choice(N, size=(bootstraps), replace=True)
    errors_time = [] 
    for choice in rdf_4_160_choices:
        s2_4_160 = np.trapz(y = s2_integrand(rdf_4_160[choice, idx], dist_4_160, rho = .85), x = dist_4_160)
        D_4_160 = c1 * np.exp(c2 * s2_4_160)
        D_4_160 *= scaling_factor
        error_4_160 = 100*(D_4_160-D_ref)/D_ref
        errors_time.append(error_4_160)
    errors_4_160.append(np.mean(errors_time))
    errors_4_160_std.append(np.std(errors_time, ddof=1))

    rdf_4_160_KL_choices = np.random.choice(N, size=(bootstraps), replace=True)
    errors_time_KL = []
    for choice in rdf_4_160_KL_choices:
        s2_4_160_KL = np.trapz(y = s2_integrand(KL_rdf_4_160[choice, idx], dist_KL_4_160, rho = .85), x = dist_KL_4_160)
        D_4_160_KL = c1 * np.exp(c2 * s2_4_160_KL)
        D_4_160_KL *= scaling_factor
        error_4_160_KL = 100*(D_4_160_KL-D_ref)/D_ref
        errors_time_KL.append(error_4_160_KL)
    errors_4_160_KL.append(np.mean(errors_time_KL))
    errors_4_160_KL_std.append(np.std(errors_time_KL, ddof=1))


errors_4_400 = []
errors_4_400_std = []
errors_4_400_KL = []
errors_4_400_KL_std = []
for idx in range(rdf_4_400_ave.shape[0]):
    rdf_4_400_choices = np.random.choice(N, size=(bootstraps), replace=True)
    errors_time = [] 
    for choice in rdf_4_400_choices:
        s2_4_400 = np.trapz(y = s2_integrand(rdf_4_400[choice, idx], dist_4_400, rho = .85), x = dist_4_400)
        D_4_400 = c1 * np.exp(c2 * s2_4_400)
        D_4_400 *= scaling_factor
        error_4_400 = 100*(D_4_400-D_ref)/D_ref
        errors_time.append(error_4_400)
    errors_4_400.append(np.mean(errors_time))
    errors_4_400_std.append(np.std(errors_time, ddof=1))

    rdf_4_400_KL_choices = np.random.choice(N, size=(bootstraps), replace=True)
    errors_time_KL = []
    for choice in rdf_4_400_KL_choices:
        s2_4_400_KL = np.trapz(y = s2_integrand(KL_rdf_4_400[choice, idx], dist_KL_4_400, rho = .85), x = dist_KL_4_400)
        D_4_400_KL = c1 * np.exp(c2 * s2_4_400_KL)
        D_4_400_KL *= scaling_factor
        error_KL = 100*(D_4_400_KL-D_ref)/D_ref
        errors_time_KL.append(error_KL)
    errors_4_400_KL.append(np.mean(errors_time_KL))
    errors_4_400_KL_std.append(np.std(errors_time_KL, ddof=1))


errors_4_1600 = []
errors_4_1600_std = []
errors_4_1600_KL = []
errors_4_1600_KL_std = []

for idx in range(rdf_4_1600_ave.shape[0]):
    rdf_4_1600_choices = np.random.choice(N, size=(bootstraps), replace=True)
    errors_time = [] 
    for choice in rdf_4_1600_choices:
        s2_4_1600 = np.trapz(y = s2_integrand(rdf_4_1600[choice, idx], dist_4_1600, rho = .85), x = dist_4_1600)
        D_4_1600 = c1 * np.exp(c2 * s2_4_1600)
        D_4_1600 *= scaling_factor
        error_4_1600 = 100*(D_4_1600-D_ref)/D_ref
        errors_time.append(error_4_1600)
    errors_4_1600.append(np.mean(errors_time))
    errors_4_1600_std.append(np.std(errors_time, ddof=1))

    rdf_4_1600_KL_choices = np.random.choice(N, size=(bootstraps), replace=True)
    errors_time_KL = []
    for choice in rdf_4_1600_KL_choices:
        s2_4_1600_KL = np.trapz(y = s2_integrand(KL_rdf_4_1600[choice, idx], dist_KL_4_1600, rho = .85), x = dist_KL_4_1600)
        D_KL = c1 * np.exp(c2 * s2_4_1600_KL)
        D_KL *= scaling_factor
        error_KL = 100*(D_KL-D_ref)/D_ref
        errors_time_KL.append(error_KL)
    errors_4_1600_KL.append(np.mean(errors_time_KL))
    errors_4_1600_KL_std.append(np.std(errors_time_KL, ddof=1))


    
errors = [errors_4_160, errors_4_160_KL, errors_4_400, errors_4_400_KL, errors_4_1600, errors_4_1600_KL]

errors_std = [errors_4_160_std, errors_4_160_KL_std, errors_4_400_std, errors_4_400_KL_std, errors_4_1600_std, errors_4_1600_KL_std]

errors = np.array(errors).T
errors_std = np.array(errors_std).T






fig, axs = plt.subplots(3, 2, figsize=(8, 9))


axs[0, 0].scatter(times[::10]*0.0512, EMDS_4_160_avg[::10], label=r'$ \Delta r = 4 \times 10^{-2}$', linewidth=.75, zorder=2, marker='s', facecolors='none', edgecolors='#2ca02c', s=40, linewidths=.75)
axs[0, 0].fill_between(times*0.0512, EMDS_4_160_avg-2*EMDS_4_160_std, EMDS_4_160_avg+2*EMDS_4_160_std, alpha=0.1, color='#2ca02c', zorder=1, linewidths=.75)
axs[0, 0].scatter(times[::10]*0.0512, EMDS_4_400_avg[::10], label=r'$ \Delta r = 1 \times 10^{-2}$', color='#b22222', linewidth=.75, zorder=2, marker='o', facecolors='none', edgecolors='#b22222', s=40, linewidths=.75)
axs[0, 0].fill_between(times*0.0512, EMDS_4_400_avg-2*EMDS_4_400_std, EMDS_4_400_avg+2*EMDS_4_400_std, alpha=0.1, color='#b22222', zorder=1, linewidths=.75)
axs[0, 0].scatter(times[::10]*0.0512, EMDS_4_1600_avg[::10], label=r'$ \Delta r = 2.5 \times 10^{-3}$', color='#d95f02', linewidth=.75, zorder=2, marker='X', facecolors='none', edgecolors='#d95f02', s=40, linewidths=.75)
axs[0, 0].fill_between(times*0.0512, EMDS_4_1600_avg-2*EMDS_4_1600_std, EMDS_4_1600_avg+2*EMDS_4_1600_std, alpha=0.1, color='#d95f02', zorder=1, linewidths=.75)

# Plot 'equilibration values 'values' 
# axs[0, 0].scatter(times[::10]*0.0512, [EMD_4_160_equil_mean] * len(times[::10]), label=r'$ \Delta r = 4 \times 10^{-2}$', facecolors='none', edgecolors='seagreen', marker='o', zorder=2, s=7.5)
# axs[0, 0].scatter(times[::10]*0.0512, [EMD_4_400_equil_mean] * len(times[::10]), label=r'$ \Delta r = 1 \times 10^{-2}$', facecolors='none', edgecolors='salmon', marker='s', zorder=2, s=7.5)
# axs[0, 0].scatter(times[::10]*0.0512, [EMD_4_1600_equil_mean] * len(times[::10]), label=r'$ \Delta r = 2.5 \times 10^{-3}$', facecolors='none', edgecolors='slateblue', marker='o', zorder=2, s=7.5)

axs[0, 0].plot(times[::10]*0.0512, [EMD_4_160_equil_mean] * len(times[::10]), label=r'$ \Delta r = 4 \times 10^{-2}$', zorder=2, linestyle='-', color='#2ca02c')
axs[0, 0].plot(times[::10]*0.0512, [EMD_4_400_equil_mean] * len(times[::10]), label=r'$ \Delta r = 1 \times 10^{-2}$', zorder=2, linestyle='-', color='#b22222')
axs[0, 0].plot(times[::10]*0.0512, [EMD_4_1600_equil_mean] * len(times[::10]), label=r'$ \Delta r = 2.5 \times 10^{-3}$', zorder=2, linestyle='-', color='#d95f02')


axs[0, 1].scatter(times[::10]*0.0512, EMDS_KL_4_160_avg[::10], label=r'$ \Delta r = 4 \times 10^{-2}$', color='#2ca02c', linewidth=.75, zorder=2, marker='s', facecolors='none', edgecolors='#2ca02c', s=40)
axs[0, 1].fill_between(times*0.0512, EMDS_KL_4_160_avg-2*EMDS_KL_4_160_std, EMDS_KL_4_160_avg+2*EMDS_KL_4_160_std, alpha=0.1, color='#2ca02c', zorder=1)
axs[0, 1].scatter(times[::10]*0.0512, EMDS_KL_4_400_avg[::10], label=r'$ \Delta r = 1 \times 10^{-2}$', color='#b22222', linewidth=.75, zorder=2, marker='o', facecolors='none', edgecolors='#b22222', s=40)
axs[0, 1].fill_between(times*0.0512, EMDS_KL_4_400_avg-2*EMDS_KL_4_400_std, EMDS_KL_4_400_avg+2*EMDS_KL_4_400_std, alpha=0.1, color='#b22222', zorder=1);
axs[0, 1].scatter(times[::10]*0.0512, EMDS_KL_4_1600_avg[::10], label=r'$ \Delta r = 2.5 \times 10^{-3}$', color='#d95f02', linewidth=.75, zorder=2, marker='X', facecolors='none', edgecolors='#d95f02', s=40);
axs[0, 1].fill_between(times*0.0512, EMDS_KL_4_1600_avg-2*EMDS_KL_4_1600_std, EMDS_KL_4_1600_avg+2*EMDS_KL_4_1600_std, alpha=0.1, color='#d95f02', zorder=1);


axs[0, 1].plot(times[::10]*0.0512, [KL_EMD_4_160_equil_mean] * len(times[::10]), label=r'$ \Delta r = 4 \times 10^{-2}$', zorder=2, linestyle='-', color='#2ca02c')
axs[0, 1].plot(times[::10]*0.0512, [KL_EMD_4_400_equil_mean] * len(times[::10]), label=r'$ \Delta r = 1 \times 10^{-2}$', zorder=2, linestyle='-', color='#b22222')
axs[0, 1].plot(times[::10]*0.0512, [KL_EMD_4_1600_equil_mean] * len(times[::10]), label=r'$ \Delta r = 2.5 \times 10^{-3}$', zorder=2, linestyle='-', color='#d95f02')

axs[0, 0].set_yscale('log')
axs[0, 1].set_yscale('log')

y_min = np.min([axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0]])
y_max = np.max([axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1]])

axs[0, 0].set_ylim(y_min, y_max*2)
axs[0, 1].set_ylim(y_min, y_max*2)


axs[0, 0].set_xlabel(r'$t / t_D$', fontsize=16)
axs[0, 1].set_xlabel(r'$t / t_D$', fontsize=16)


# Adds tickers to y-axis
axs[0, 0].set_yticks([1e-3, 1e-2, 1e-1])
axs[0, 1].set_yticks([1e-3, 1e-2, 1e-1])






# Make y-ticks bigger
axs[0, 0].tick_params(axis='y', labelsize=16)
axs[0, 1].tick_params(axis='y', labelsize=16)
# Make x-ticks bigger
axs[0, 0].tick_params(axis='x', labelsize=16)
axs[0, 1].tick_params(axis='x', labelsize=16)

# Make top and right ticks invisible



axs[0, 0].set_ylabel(r'$ \xi (g_t(r), g_{\mathrm{eq}}(r))$', fontsize=16)



log_EMDS_vals = -np.log(EMDS_4_160_avg)
log_min = np.min(log_EMDS_vals)
log_max = np.max(log_EMDS_vals)
log_space = np.linspace(log_min, log_max, 10)

closest_idxs = []
for v in log_space:
    closest_idx = np.argmin(np.abs(log_EMDS_vals - v))
    closest_idxs.append(closest_idx)

axs[1, 0].scatter(EMDS_4_160_avg[closest_idxs], errors[closest_idxs, 0], color='#2ca02c', linewidth=.75, zorder=2, marker='s', facecolors='none', edgecolors='#2ca02c', s=40, linewidths=0.75)
axs[1, 1].scatter(EMDS_KL_4_160_avg[closest_idxs], errors[closest_idxs, 1], color='#2ca02c', linewidth=.75, zorder=2, marker='s', facecolors='none', edgecolors='#2ca02c', s=40, linewidths=0.75)


log_EMDS_vals = -np.log(EMDS_4_400_avg)
log_min = np.min(log_EMDS_vals)
log_max = np.max(log_EMDS_vals)
log_space = np.linspace(log_min, log_max, 10)

closest_idxs = []
for v in log_space:
    closest_idx = np.argmin(np.abs(log_EMDS_vals - v))
    closest_idxs.append(closest_idx)

axs[1, 0].scatter(EMDS_4_400_avg[closest_idxs], errors[closest_idxs, 2], color='#b22222', linewidth=.75, zorder=2, marker='o', facecolors='none', edgecolors='#b22222', s=40, linewidths=0.75)
axs[1, 1].scatter(EMDS_KL_4_400_avg[closest_idxs], errors[closest_idxs, 3], color='#b22222', linewidth=.75, zorder=2, marker='o', facecolors='none', edgecolors='#b22222', s=40, linewidths=0.75)

log_EMDS_vals = -np.log(EMDS_4_1600_avg)
log_min = np.min(log_EMDS_vals)
log_max = np.max(log_EMDS_vals)
log_space = np.linspace(log_min, log_max, 10)

closest_idxs = []
for v in log_space:
    closest_idx = np.argmin(np.abs(log_EMDS_vals - v))
    closest_idxs.append(closest_idx)

axs[1, 0].scatter(EMDS_4_1600_avg[closest_idxs], errors[closest_idxs, 4], color='#d95f02', linewidth=.75, zorder=2, marker='X', facecolors='none', edgecolors='#d95f02', s=40, linewidths=0.75)
axs[1, 1].scatter(EMDS_KL_4_1600_avg[closest_idxs], errors[closest_idxs, 5], color='#d95f02', linewidth=.75, zorder=2, marker='X', facecolors='none', edgecolors='#d95f02', s=40, linewidths=0.75)

axs[1, 0].fill_between(EMDS_4_160_avg, errors[:, 0]-2*errors_std[:, 0], errors[:, 0]+2*errors_std[:, 0], alpha=0.1, color='#2ca02c', zorder=1)
axs[1, 0].fill_between(EMDS_4_400_avg, errors[:, 2]-2*errors_std[:, 2], errors[:, 2]+2*errors_std[:, 2], alpha=0.1, color='#b22222', zorder=1)
axs[1, 0].fill_between(EMDS_4_1600_avg, errors[:, 4]-2*errors_std[:, 4], errors[:, 4]+2*errors_std[:, 4], alpha=0.1, color='#d95f02', zorder=1)
# axs[1, 0].scatter(EMDS_4_400_avg, errors[:, 2], color='salmon', s=75, marker='s', alpha=0.7, edgecolor='black', linewidth=0.25, label=r'$ \Delta r = 1 \times 10^{-2}$', zorder=2)

# axs[1, 0].scatter(EMDS_4_1600_avg, errors[:, 4], color='slateblue', s=75, marker='o', alpha=0.7, edgecolor='black', linewidth=0.25, label=r'$ \Delta r = 2.5 \times 10^{-3}$', zorder=2)

axs[1, 0].plot(EMDS_4_160_avg, np.zeros_like(EMDS_4_160_avg), color='black', linewidth=1.5, linestyle='--', zorder=3)


axs[1, 1].fill_between(EMDS_KL_4_160_avg, errors[:, 1]-2*errors_std[:, 1], errors[:, 1]+2*errors_std[:, 1], alpha=0.1, color='#2ca02c', zorder=1)
axs[1, 1].fill_between(EMDS_KL_4_400_avg, errors[:, 3]-2*errors_std[:, 3], errors[:, 3]+2*errors_std[:, 3], alpha=0.1, color='#b22222', zorder=1)
axs[1, 1].fill_between(EMDS_KL_4_1600_avg, errors[:, 5]-2*errors_std[:, 5], errors[:, 5]+2*errors_std[:, 5], alpha=0.1, color='#d95f02', zorder=1)

axs[1, 1].plot(EMDS_KL_4_160_avg, np.zeros_like(EMDS_KL_4_160_avg), color='black', linewidth=1.5, linestyle='--', zorder=3)

axs[1, 0].set_xscale('log')
axs[1, 1].set_xscale('log')


x_min = np.min([axs[1, 0].get_xlim()[0], axs[1, 1].get_xlim()[0]])
x_max = np.max([axs[1, 0].get_xlim()[1], axs[1, 1].get_xlim()[1]])

axs[1, 0].set_xlim(x_max, x_min)
axs[1, 1].set_xlim(x_max, x_min)

# Make y-ticks bigger
axs[1, 0].tick_params(axis='y', labelsize=16)
axs[1, 1].tick_params(axis='y', labelsize=16)
# Make x-ticks bigger
axs[1, 0].tick_params(axis='x', labelsize=16)
axs[1, 1].tick_params(axis='x', labelsize=16)

axs[1, 0].set_ylabel(r'$\%$ Error', fontsize=16)




axs[1, 0].set_xlabel(r'$ \xi (g_t(r), g_{\mathrm{eq}}(r))$', fontsize=16)
axs[1, 1].set_xlabel(r'$ \xi (g_t(r), g_{\mathrm{eq}}(r))$', fontsize=16)




axs[2, 0].scatter(times[::10]*0.0512, S2_EMDS_4_160_balanced_avg[::10], label=r'$ \Delta r = 4 \times 10^{-2}$', linewidth=.75, zorder=2, marker='s', facecolors='none', edgecolors='#2ca02c', s=40, linewidths=3.5)
axs[2, 0].fill_between(times*0.0512, S2_EMDS_4_160_balanced_avg-2*S2_EMDS_4_160_balanced_std, S2_EMDS_4_160_balanced_avg+2*S2_EMDS_4_160_balanced_std, alpha=0.1, color='#2ca02c', zorder=1)
axs[2, 0].scatter(times[::10]*0.0512, S2_EMDS_4_400_balanced_avg[::10], label=r'$ \Delta r = 1 \times 10^{-2}$', linewidth=.75, linestyle='-.', zorder=2, marker='o', facecolors='none', edgecolors='#b22222', s=40,  linewidths=3.5)
axs[2, 0].fill_between(times*0.0512, S2_EMDS_4_400_balanced_avg-2*S2_EMDS_4_400_balanced_std, S2_EMDS_4_400_balanced_avg+2*S2_EMDS_4_400_balanced_std, alpha=0.1, color='#b22222', zorder=1)
axs[2, 0].scatter(times[::10]*0.0512, S2_EMDS_4_1600_balanced_avg[::10], label=r'$ \Delta r = 2.5 \times 10^{-3}$', linewidth=.75, linestyle='-.', zorder=2, marker='X', facecolors='none', edgecolors='#d95f02', s=40 , linewidths=3.5)
axs[2, 0].fill_between(times*0.0512, S2_EMDS_4_1600_balanced_avg-2*S2_EMDS_4_1600_balanced_std, S2_EMDS_4_1600_balanced_avg+2*S2_EMDS_4_1600_balanced_std, alpha=0.1, color='#d95f02', zorder=1)


axs[2, 0].plot(times[::10]*0.0512, [S2_EMDS_4_160_equil_balanced_avg] * len(times[::10]), label=r'$ \Delta r = 4 \times 10^{-2}$', linestyle='-', zorder=2, color='#2ca02c')
axs[2, 0].plot(times[::10]*0.0512, [S2_EMDS_4_400_equil_balanced_avg] * len(times[::10]), label=r'$ \Delta r = 1 \times 10^{-2}$', linestyle='-', zorder=2, color='#b22222')
axs[2, 0].plot(times[::10]*0.0512, [S2_EMDS_4_1600_equil_balanced_avg] * len(times[::10]), label=r'$ \Delta r = 2.5 \times 10^{-3}$', linestyle='-', zorder=2, color='#d95f02')

axs[2, 1].scatter(times[::10]*0.0512, S2_EMDS_KL_4_160_balanced_avg[::10], label=r'$ \Delta r = 4 \times 10^{-2}$', zorder=2, marker='s', facecolors='none', edgecolors='#2ca02c', s=40)
axs[2, 1].fill_between(times*0.0512, S2_EMDS_KL_4_160_balanced_avg-2*S2_EMDS_KL_4_160_balanced_std, S2_EMDS_KL_4_160_balanced_avg+2*S2_EMDS_KL_4_160_balanced_std, alpha=0.1, color='#2ca02c', zorder=1)
axs[2, 1].scatter(times[::10]*0.0512, S2_EMDS_KL_4_400_balanced_avg[::10], label=r'$ \Delta r = 1 \times 10^{-2}$', zorder=2, marker='o', facecolors='none', edgecolors='#b22222', s=40)
axs[2, 1].fill_between(times*0.0512, S2_EMDS_KL_4_400_balanced_avg-2*S2_EMDS_KL_4_400_balanced_std, S2_EMDS_KL_4_400_balanced_avg+2*S2_EMDS_KL_4_400_balanced_std, alpha=0.1, color='#b22222', zorder=1)
axs[2, 1].scatter(times[::10]*0.0512, S2_EMDS_KL_4_1600_balanced_avg[::10], label=r'$ \Delta r = 2.5 \times 10^{-3}$', zorder=2, marker='X', facecolors='none', edgecolors='#d95f02', s=40)
axs[2, 1].fill_between(times*0.0512, S2_EMDS_KL_4_1600_balanced_avg-2*S2_EMDS_KL_4_1600_balanced_std, S2_EMDS_KL_4_1600_balanced_avg+2*S2_EMDS_KL_4_1600_balanced_std, alpha=0.1, color='#d95f02', zorder=1)
# axs[2, 1].scatter(times[::10]*0.0512, [S2_EMDS_KL_4_160_equil_avg] * len(times[::10]), label=r'$ \Delta r = 4 \times 10^{-2}$', facecolors='none', edgecolors='seagreen', marker='o', zorder=2, s=7.5)
# axs[2, 1].scatter(times[::10]*0.0512, [S2_EMDS_KL_4_400_equil_avg] * len(times[::10]), label=r'$ \Delta r = 1 \times 10^{-2}$', facecolors='none', edgecolors='salmon', marker='s', zorder=2, s=7.5)
# axs[2, 1].scatter(times[::10]*0.0512, [S2_EMDS_KL_4_1600_equil_avg] * len(times[::10]), label=r'$ \Delta r = 2.5 \times 10^{-3}$', facecolors='none', edgecolors='slateblue', marker='o', zorder=2, s=7.5)

axs[2, 1].plot(times[::10]*0.0512, [S2_EMDS_KL_4_160_equil_balanced_avg] * len(times[::10]), label=r'$ \Delta r = 4 \times 10^{-2}$', zorder=2, linestyle='-', color='#2ca02c')
axs[2, 1].plot(times[::10]*0.0512, [S2_EMDS_KL_4_400_equil_balanced_avg] * len(times[::10]), label=r'$ \Delta r = 1 \times 10^{-2}$',  zorder=2, linestyle='-', color='#b22222')
axs[2, 1].plot(times[::10]*0.0512, [S2_EMDS_KL_4_1600_equil_balanced_avg] * len(times[::10]), label=r'$ \Delta r = 2.5 \times 10^{-3}$', zorder=2, linestyle='-', color='#d95f02')


# axs[2, 0].text(.85, 0.875, 'raw', fontsize=12, transform=axs[2, 0].transAxes, ha='right')
# axs[2, 1].text(.65, 0.875, 'KAMEL-LOBE', fontsize=12, transform=axs[2, 1].transAxes, ha='right')

# choose a uniform offset in points (1 point = 1/72 inch)
offset_pts = 6
fs         = 12
y_pos      = 0.88  # vertical in axes‐fraction coordinates


for ax, txt in zip(axs[0, :], ['raw', 'KAMEL-LOBE']):
    ax.annotate(
        txt,
        xy=(1, y_pos),              # anchor at right border of axes
        xycoords='axes fraction',
        xytext=(-offset_pts, 0),    # shift left by 6 points
        textcoords='offset points',
        ha='right', 
        va='center',
        fontsize=fs
    )

for ax, txt in zip(axs[1, :], ['raw', 'KAMEL-LOBE']):
    ax.annotate(
        txt,
        xy=(1, y_pos),              # anchor at right border of axes
        xycoords='axes fraction',
        xytext=(-offset_pts, 0),    # shift left by 6 points
        textcoords='offset points',
        ha='right', 
        va='center',
        fontsize=fs
    )

for ax, txt in zip(axs[2, :], ['raw', 'KAMEL-LOBE']):
    ax.annotate(
        txt,
        xy=(1, y_pos),              # anchor at right border of axes
        xycoords='axes fraction',
        xytext=(-offset_pts, 0),    # shift left by 6 points
        textcoords='offset points',
        ha='right', 
        va='center',
        fontsize=fs
    )



y_min = np.min([axs[2, 0].get_ylim()[0], axs[2, 1].get_ylim()[0]])
y_max = np.max([axs[2, 0].get_ylim()[1], axs[2, 1].get_ylim()[1]])
axs[2, 0].set_ylim(y_min, y_max)
axs[2, 1].set_ylim(y_min, y_max)



# Make y-ticks bigger
axs[2, 0].tick_params(axis='y', labelsize=16)
axs[2, 1].tick_params(axis='y', labelsize=16)
# Make x-ticks bigger
axs[2, 0].tick_params(axis='x', labelsize=16)
axs[2, 1].tick_params(axis='x', labelsize=16)

axs[2, 0].set_ylabel(r'$ \xi (\phi_t(r), \phi_{\mathrm{eq}}(r))$', fontsize=16)


axs[2, 0].set_xlabel(r'$t / t_D$', fontsize=16)
axs[2, 1].set_xlabel(r'$t / t_D$', fontsize=16)







# — after your fig, axs = plt.subplots(…) —
for ax in axs.flatten():

    # 4) subtly lighten the panel background (optional)
    ax.patch.set_facecolor('white')
    # 5) tighten up tick placement
    ax.tick_params(which='major', width=1.0)
    ax.tick_params(which='minor', length=3)
    ax.tick_params(which='both', top=False, right=False)

    # No ticks for top and right

# Plot legend axis at the very bottom of the plot 


handles, labels = axs[2,0].get_legend_handles_labels()



h_top,  l_top  = handles[:3], labels[:3]
h_bot,  l_bot  = handles[3:], labels[3:]

# 3) first legend (row 1)
leg1 = fig.legend(
    h_top, l_top,
    ncol=3,
    fontsize=16,
    frameon=False,
    title='Out of Equilibrium',            # ← your “row 1” label
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05)        # tweak the y to sit just under the axes
)
fig.add_artist(leg1)                    # keep it around

# 4) second legend (row 2)
leg2 = fig.legend(
    h_bot, l_bot,
    ncol=3,
    fontsize=16,
    frameon=False,
    title='Equilibrium',           # ← your “row 2” label
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12)         # a bit lower than the first one
)


# After creating leg1 and leg2:




# give extra room at the bottom so the legend isn’t clipped
plt.subplots_adjust(bottom=0.25,
    hspace=0.15)


for ax in axs.flatten():
    for spine in ["left","bottom", "right", "top"]:
        ax.spines[spine].set_linewidth(2.0)

row_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
label_fs   = mpl.rcParams['axes.titlesize']  # 16 pt in your rcParams
for i, lbl in enumerate(row_labels):
    ax = axs[i//2, i%2]
    ax.text(
        x   = 0.05,            # a little left of the plotting area
        y   = 1.025,             # just above the top
        s   = lbl, 
        transform = ax.transAxes,
        fontsize  = 20,
        va         = 'bottom',
        # ha         = 'right',
    )
plt.tight_layout()


# plt.savefig("EES_2.svg", format="svg", dpi=900, bbox_inches="tight")
