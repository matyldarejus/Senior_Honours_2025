import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
import h5py
import pygad as pg
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import *
from physics import *
sys.path.append('/home/matylda/sh/tools/')
import plotmedian as pm

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 27
plt.rcParams['axes.titlesize'] = 27
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
    cmap_list = cmap(np.linspace(minval, maxval, n))
    cmap_list[:, -1] = alpha
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap_list)
    return new_cmap

def quench_thresh(z):
    return -1.8 + 0.3*z - 9.

def ssfr_type_check(ssfr_thresh, ssfr):
    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh - 1)
    q_mask  = ssfr == -14.0
    return sf_mask, gv_mask, q_mask

def median_with_percentiles(x, y, nmin=10):
    """Compute running median and 16th-84th percentile bands."""
    # Remove NaNs and check we have enough data
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < nmin:
        return np.array([]), np.array([]), np.array([]), np.array([])
    bin_cent, ymed, ysiglo, ysighi, ndata = pm.runningmedian(x, y, stat='median')
    mask = ndata > nmin
    return bin_cent[mask], ymed[mask], ysiglo[mask], ysighi[mask]


if __name__ == '__main__':

    model = sys.argv[1]
    wind  = sys.argv[2]
    snap  = sys.argv[3]

    zsolar    = 5.79e-3
    chisq_lim = 4.5
    N_min     = 13.2
    nmin      = 10  # minimum galaxies per bin for median

    delta_fr200 = 0.25
    min_fr200   = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200 + 1) * delta_fr200, delta_fr200)

    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s        = pg.Snapshot(snapfile)
    redshift = s.redshift
    quench   = quench_thresh(redshift)

    plot_dir   = '/home/matylda/data/plots/'
    sample_dir = '/disk04/mrejus/sh/samples/'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_OVI1031.h5'
    sample_file  = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'

    # --- Load galaxy sample ---
    with h5py.File(sample_file, 'r') as sf:
        gal_ids        = sf['gal_ids'][:]
        mass           = sf['mass'][:]
        ssfr           = sf['ssfr'][:]
        Z_sfr_weighted = sf['Z_sfr_weighted'][:]
        Z_cgm_mw       = sf['Z_cgm_mw'][:]

    # --- Load absorbers ---
    all_Z_abs   = []
    all_N_abs   = []
    all_chisq   = []
    all_ids_abs = []

    Z_ism_sol = np.log10(Z_sfr_weighted) - np.log10(zsolar)
    Z_cgm_sol = np.log10(Z_cgm_mw)       - np.log10(zsolar)

    with h5py.File(results_file, 'r') as hf:
        for r in fr200:
            try:
                all_Z_abs.extend(hf[f'log_Z_{r}r200'][:])
                all_N_abs.extend(hf[f'log_N_{r}r200'][:])
                all_chisq.extend(hf[f'chisq_{r}r200'][:])
                all_ids_abs.extend(hf[f'ids_{r}r200'][:])
            except KeyError:
                continue

    all_Z_abs   = np.array(all_Z_abs)
    all_N_abs   = np.array(all_N_abs)
    all_chisq   = np.array(all_chisq)
    all_ids_abs = np.array(all_ids_abs)

    mask        = (all_N_abs > N_min) & (all_chisq < chisq_lim)
    all_Z_abs   = all_Z_abs[mask] - np.log10(zsolar)
    all_ids_abs = all_ids_abs[mask]

    valid       = np.isin(all_ids_abs, gal_ids)
    all_Z_abs   = all_Z_abs[valid]
    all_ids_abs = all_ids_abs[valid]
    idx         = np.array([np.where(gal_ids == gid)[0][0] for gid in all_ids_abs])
    abs_mass    = mass[idx]
    abs_ssfr    = ssfr[idx]

    # PLOT 1: Stellar mass bins    
    delta_m   = 0.5
    min_m     = 10.
    nbins_m   = 3
    mass_bins = np.arange(min_m, min_m + (nbins_m + 1) * delta_m, delta_m)
    mass_bin_labels = [f'{mass_bins[i]:.1f}'+r'$ < \log M_\star < $'+f'{mass_bins[i+1]:.1f}'
                       for i in range(nbins_m)]

    cmap_mass  = truncate_colormap(cm.get_cmap('magma'), 0.2, 0.8)
    idelta     = 1. / (len(mass_bins) - 1)
    icolor     = np.arange(0., 1. + idelta, idelta)
    mass_colors = [cmap_mass(i) for i in icolor]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ['ISM (SFR-weighted)', 'CGM (mass-weighted)', r'OVI Absorbers']
    
    plot_sets = [
        (Z_ism_sol, mass,     'ISM (SFR-weighted)'),
        (Z_cgm_sol, mass,     'CGM (mass-weighted)'),
        (all_Z_abs, abs_mass, r'OVI Absorbers'),
    ]

    for ax, (Z_data, x_data, title) in zip(axes, plot_sets):

        # All galaxies in grey
        bc, ymed, ylo, yhi = median_with_percentiles(x_data, Z_data, nmin)
        if len(bc) > 0:
            ax.plot(bc, ymed, c='dimgrey', lw=2, ls='-', label='All')
            ax.fill_between(bc, ylo, yhi, color='dimgrey', alpha=0.15)

        # Mass bins
        for k in range(nbins_m):
            mlo, mhi = mass_bins[k], mass_bins[k+1]
            if title == r'OVI Absorbers':
                bin_mask = (abs_mass >= mlo) & (abs_mass < mhi)
            else:
                bin_mask = (mass >= mlo) & (mass < mhi)

            if np.sum(bin_mask) < nmin:
                continue

            bc, ymed, ylo, yhi = median_with_percentiles(
                x_data[bin_mask], Z_data[bin_mask], nmin)
            if len(bc) == 0:
                continue
            ax.plot(bc, ymed, c=mass_colors[k], lw=2, ls='-')
            ax.fill_between(bc, ylo, yhi, color=mass_colors[k], alpha=0.2)

        ax.axhline(0, ls=':', c='k', lw=1.5)
        ax.set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
        ax.set_title(title)
        ax.set_xlim(9.75, 11.75)
        ax.set_ylim(-2, 1)

    axes[0].set_ylabel(r'$\log\ (Z / Z_{\odot})$')

    # Legend
    legend_lines = [Line2D([0,1],[0,1], color='dimgrey', lw=2, label='All')]
    for k in range(nbins_m):
        legend_lines.append(Line2D([0,1],[0,1], color=mass_colors[k], lw=2,
                                   label=mass_bin_labels[k]))
    axes[0].legend(handles=legend_lines, fontsize=16, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_median_mass_bins.png', format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_median_mass_bins.pdf', format='pdf')
    plt.close()
    print(f'Saved: Z_median_mass_bins')

    # PLOT 2: Split by SF/GV/Q
    sf_mask_gal, gv_mask_gal, q_mask_gal = ssfr_type_check(quench, ssfr)
    sf_mask_abs, gv_mask_abs, q_mask_abs = ssfr_type_check(quench, abs_ssfr)

    ssfr_colors = ['#5289C7', '#90C987', '#E26F72']  # blue=SF, green=GV, red=Q
    ssfr_labels = ['Star-forming', 'Green valley', 'Quenched']
    gal_masks   = [sf_mask_gal, gv_mask_gal, q_mask_gal]
    abs_masks   = [sf_mask_abs, gv_mask_abs, q_mask_abs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    plot_sets = [
        (Z_ism_sol, mass,     'ISM (SFR-weighted)',  False),
        (Z_cgm_sol, mass,     'CGM (mass-weighted)', False),
        (all_Z_abs, abs_mass, r'OVI Absorbers',       True),
    ]

    for ax, (Z_data, x_data, title, is_abs) in zip(axes, plot_sets):
        # All in grey
        bc, ymed, ylo, yhi = median_with_percentiles(x_data, Z_data, nmin)
        if len(bc) > 0:pu
            ax.plot(bc, ymed, c='dimgrey', lw=2, ls='-', label='All')
            ax.fill_between(bc, ylo, yhi, color='dimgrey', alpha=0.15)

        # SF / GV / Q
        masks = abs_masks if is_abs else gal_masks
        for color, label, smask in zip(ssfr_colors, ssfr_labels, masks):
            if np.sum(smask) < nmin:
                continue
            bc, ymed, ylo, yhi = median_with_percentiles(
                x_data[smask], Z_data[smask], nmin)
            if len(bc) == 0:
                continue
            ax.plot(bc, ymed, c=color, lw=2, ls='-', label=label)
            ax.fill_between(bc, ylo, yhi, color=color, alpha=0.2)

        ax.axhline(0, ls=':', c='k', lw=1.5)
        ax.set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
        ax.set_title(title)
        ax.set_xlim(9.75, 11.75)
        ax.set_ylim(-2, 1)

    axes[0].set_ylabel(r'$\log\ (Z / Z_{\odot})$')

    legend_lines = [Line2D([0,1],[0,1], color='dimgrey', lw=2, label='All')]
    for color, label in zip(ssfr_colors, ssfr_labels):
        legend_lines.append(Line2D([0,1],[0,1], color=color, lw=2, label=label))
    axes[0].legend(handles=legend_lines, fontsize=16, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_median_ssfr_bins.png', format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_median_ssfr_bins.pdf', format='pdf')
    plt.close()
    print(f'Saved: Z_median_ssfr_bins')