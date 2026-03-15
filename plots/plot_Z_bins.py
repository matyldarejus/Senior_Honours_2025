import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import pygad as pg
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import *
from physics import *
 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130
 
def quench_thresh(z):  # in units of yr^-1
    return -1.8 + 0.3*z - 9.
 
def ssfr_type_check(ssfr_thresh, ssfr):
    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh - 1)
    q_mask  = ssfr == -14.0
    return sf_mask, gv_mask, q_mask
 
 
if __name__ == '__main__':
 
    model = sys.argv[1]
    wind  = sys.argv[2]
    snap  = sys.argv[3]
 
    zsolar    = 5.79e-3   # Solar oxygen abundance
    chisq_lim = 4.5
    N_min     = 13.2
 
    ssfr_labels = ['All', 'Star-forming', 'Green valley', 'Quenched']
 
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
 
    # Load galaxy sample
    with h5py.File(sample_file, 'r') as sf:
        gal_ids         = sf['gal_ids'][:]
        mass            = sf['mass'][:]
        ssfr            = sf['ssfr'][:]
        Z_sfr_weighted  = sf['Z_sfr_weighted'][:]   # ISM metallicity proxy
        Z_cgm_mw        = sf['Z_cgm_mw'][:]         # CGM mass-weighted metallicity
 
    # Convert to log Z/Zsolar
    Z_ism_sol = np.log10(Z_sfr_weighted) - np.log10(zsolar)
    Z_cgm_sol = np.log10(Z_cgm_mw)       - np.log10(zsolar)
 
    # Load absorber data
    all_Z_abs   = []
    all_N_abs   = []
    all_chisq   = []
    all_ids_abs = []
 
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
 
    # Quality cuts
    mask        = (all_N_abs > N_min) & (all_chisq < chisq_lim)
    all_Z_abs   = all_Z_abs[mask]   - np.log10(zsolar)
    all_ids_abs = all_ids_abs[mask]
 
    # Match absorbers to galaxy properties
    valid       = np.isin(all_ids_abs, gal_ids)
    all_ids_abs = all_ids_abs[valid]
    all_Z_abs   = all_Z_abs[valid]
    idx         = np.array([np.where(gal_ids == gid)[0][0] for gid in all_ids_abs])
    abs_mass    = mass[idx]
    abs_ssfr    = ssfr[idx]
 
    sf_mask_abs, gv_mask_abs, q_mask_abs = ssfr_type_check(quench, abs_ssfr)
    sf_mask_gal, gv_mask_gal, q_mask_gal = ssfr_type_check(quench, ssfr)
 
    # Masks per column: [All, SF, GV, Q]
    abs_masks = [np.ones(len(all_Z_abs), dtype=bool), sf_mask_abs, gv_mask_abs, q_mask_abs]
    gal_masks = [np.ones(len(mass),      dtype=bool), sf_mask_gal, gv_mask_gal, q_mask_gal]
 
    # Colour limits
    vmin_mass = 9.75
    vmax_mass = 11.75
    vmin_Z    = -2.0
    vmax_Z    =  1.0
 
    # Figure: 2 rows x 4 cols
    #   Row 0 (top)    : ISM SFR-weighted metallicity vs stellar mass
    #   Row 1 (bottom) : CGM mass-weighted metallicity vs stellar mass
    #   Colour         : absorber Z/Zsolar
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharey='row', sharex='col')
 
    for col, (amask, gmask, label) in enumerate(zip(abs_masks, gal_masks, ssfr_labels)):
 
        # Row 0: ISM
        sc0 = ax[0][col].scatter(mass[gmask], Z_ism_sol[gmask],
                                 c=mass[gmask], cmap='plasma',
                                 s=8, alpha=0.5, vmin=vmin_mass, vmax=vmax_mass,
                                 zorder=1, rasterized=True)
 
        # Overplot absorbers coloured by Z
        sc0_abs = ax[0][col].scatter(abs_mass[amask], all_Z_abs[amask],
                                     c=all_Z_abs[amask], cmap='RdYlBu',
                                     s=12, alpha=0.8, vmin=vmin_Z, vmax=vmax_Z,
                                     edgecolors='k', linewidths=0.2,
                                     zorder=2, marker='D')
 
        # Row 1: CGM
        sc1 = ax[1][col].scatter(mass[gmask], Z_cgm_sol[gmask],
                                 c=mass[gmask], cmap='plasma',
                                 s=8, alpha=0.5, vmin=vmin_mass, vmax=vmax_mass,
                                 zorder=1, rasterized=True)
 
        sc1_abs = ax[1][col].scatter(abs_mass[amask], all_Z_abs[amask],
                                     c=all_Z_abs[amask], cmap='RdYlBu',
                                     s=12, alpha=0.8, vmin=vmin_Z, vmax=vmax_Z,
                                     edgecolors='k', linewidths=0.2,
                                     zorder=2, marker='D')
 
        # Solar metallicity line
        for row in range(2):
            ax[row][col].axhline(0, ls=':', c='k', lw=1.5)
            ax[row][col].set_xlim(9.75, 11.75)
            ax[row][col].set_ylim(vmin_Z, vmax_Z)
 
        if col == 0:
            ax[0][0].set_title(ssfr_labels[0])
        ax[0][col].set_title(label)
 
    # Row labels
    ax[0][0].set_ylabel(r'$\log\ (Z_{\rm ISM} / Z_{\odot})$')
    ax[1][0].set_ylabel(r'$\log\ (Z_{\rm CGM} / Z_{\odot})$')
 
    # x-axis labels on bottom row only
    for col in range(4):
        ax[1][col].set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
 
    # Stellar mass colorbar (right side, galaxies)
    cax_mass = fig.add_axes([0.92, 0.55, 0.012, 0.35])
    cb_mass  = fig.colorbar(sc0, cax=cax_mass)
    cb_mass.set_label(r'$\log\ (M_{\star} / M_{\odot})$', fontsize=18)
    cb_mass.ax.tick_params(labelsize=16)
 
    # Absorber Z colorbar (right side, absorbers)
    cax_Z   = fig.add_axes([0.92, 0.12, 0.012, 0.35])
    cb_Z    = fig.colorbar(sc0_abs, cax=cax_Z)
    cb_Z.set_label(r'$\log\ (Z_{\rm abs} / Z_{\odot})$', fontsize=18)
    cb_Z.ax.tick_params(labelsize=16)
 
    fig.subplots_adjust(wspace=0.05, hspace=0.1, right=0.90)
 
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_ISM_CGM_ssfr_bins.png', format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_ISM_CGM_ssfr_bins.pdf', format='pdf')
    plt.close()
 
    print(f'Saved: {plot_dir}{model}_{wind}_{snap}_Z_ISM_CGM_ssfr_bins.png')