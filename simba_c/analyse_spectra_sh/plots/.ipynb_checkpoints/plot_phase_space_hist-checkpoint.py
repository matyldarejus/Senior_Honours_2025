# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025 – single-ion version

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'

def quench_thresh(z):
    return -1.8 + 0.3*z - 9.

def ssfr_type_check(ssfr_thresh, ssfr):
    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh - 1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    line = "OVI1031"
    plot_line = r'${\rm OVI}\ 1031$'
    Nlabel = r'${\rm log }(N_{\rm OVI} / {\rm cm}^{-2})$'
    chisq_lim_dict = {'snap_151': 4.5,
                      'snap_137': 4.5,
                      'snap_125': 5.6,
                      'snap_105': 7.1}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = 13.2
    N_max = 18.0

    dT = 0.1
    T_min, T_max = 3., 6.
    delta_rho_min, delta_rho_max, ddelta = -1., 5., 0.2
    dN = 0.2

    T_bins = np.arange(T_min, T_max+dT, dT)
    delta_rho_bins = np.arange(delta_rho_min, delta_rho_max+ddelta, ddelta)
    N_bins = np.arange(N_min, N_max+dN, dN)

    inner_outer = [[0.25, 0.5], [0.75, 1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    rho_ls = ['--', ':']
    rho_lw = [1, 2]
    rho_y = [0.8, 0.9]
    ssfr_labels = ['Star forming', 'Green valley', 'Quenched']
    ssfr_colors = [cb_blue, cb_green, cb_red]

    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    plot_dir = f'/home/matylda/data/plots/'
    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_OVI1031.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    fig, ax = plt.subplots(figsize=(7, 5))

    ssfr_lines = [Line2D([0,1],[0,1], color=c, lw=1) for c in ssfr_colors]
    ax.legend(ssfr_lines, ssfr_labels, loc='upper right', fontsize=12)

    all_delta_rho = np.array([])
    all_ids = np.array([])
    sf_median = np.zeros(len(inner_outer))
    gv_median = np.zeros(len(inner_outer))
    q_median = np.zeros(len(inner_outer))

    for k in range(len(inner_outer)):
        rho, N, chisq, ids = [], [], [], []
        for l in range(len(inner_outer[k])):
            with h5py.File(results_file, 'r') as hf:
                rho.extend(hf[f'log_rho_{inner_outer[k][l]}r200'][:])
                N.extend(hf[f'log_N_{inner_outer[k][l]}r200'][:])
                chisq.extend(hf[f'chisq_{inner_outer[k][l]}r200'][:])
                ids.extend(hf[f'ids_{inner_outer[k][l]}r200'][:])

        rho, N, chisq, ids = map(np.array, [rho, N, chisq, ids])
        mask = (N > N_min) * (chisq < chisq_lim)
        delta_rho = rho[mask] - np.log10(cosmic_rho)
        ids = ids[mask]

        mask = (delta_rho > delta_rho_bins[0]) & (delta_rho < delta_rho_bins[-1])
        idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr[idx])

        sf_median[k] = np.nanmedian(delta_rho[sf_mask*mask])
        gv_median[k] = np.nanmedian(delta_rho[gv_mask*mask])
        q_median[k] = np.nanmedian(delta_rho[q_mask*mask])

        all_delta_rho = np.append(all_delta_rho, delta_rho)
        all_ids = np.append(all_ids, ids)

    idx = np.array([np.where(gal_ids == l)[0] for l in all_ids]).flatten()
    sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr[idx])

    ax.hist(all_delta_rho[sf_mask], bins=delta_rho_bins, density=True, color=cb_blue, histtype='step', lw=1)
    ax.hist(all_delta_rho[gv_mask], bins=delta_rho_bins, density=True, color=cb_green, histtype='step', lw=1)
    ax.hist(all_delta_rho[q_mask], bins=delta_rho_bins, density=True, color=cb_red, histtype='step', lw=1)

    for k in range(len(inner_outer)):
        ax.axvline(sf_median[k], ymin=rho_y[k], color=cb_blue, ls=rho_ls[k], lw=rho_lw[k])
        ax.axvline(gv_median[k], ymin=rho_y[k], color=cb_green, ls=rho_ls[k], lw=rho_lw[k])
        ax.axvline(q_median[k], ymin=rho_y[k], color=cb_red, ls=rho_ls[k], lw=rho_lw[k])

    ax.set_xlim(delta_rho_min, delta_rho_max)
    ax.set_xlabel(r'${\rm log }\delta$')
    ax.set_ylabel('Frequency')
    ax.annotate(plot_line, xy=(0.05, 0.9), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))
    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_{line}_delta_hist.png', dpi=300)
    plt.close()