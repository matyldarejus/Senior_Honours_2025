# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025
# Plots CDDFs for different azimuthal angle bins (major vs minor axis)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'
cb_orange = '#E1BE6A'
cb_purple = '#B497E7'

def stop_array_after_inf(array):
    mask = np.isinf(array)
    if len(array[mask]) > 0:
        inf_start = np.where(mask)[0][0]
        array[inf_start:] = np.inf
        return array
    else:
        return array


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']

    # These mimic inner/outer but for major/minor axes
    az_labels = ['major', 'minor']
    az_colors = [cb_blue, cb_red]
    az_ls = ['-', '--']

    rho_labels = ['All CGM', 'Major Axis', 'Minor Axis']
    ssfr_labels = ['All galaxies', 'Star forming', 'Green valley', 'Quenched']
    ssfr_colors = ['dimgrey', cb_blue, cb_green, cb_red]
    rho_ls = ['-', '--', ':']
    rho_lw = [1, 1.5, 2]
    logN_min = 11.
    x = [0.79, 0.76, 0.77, 0.75, 0.755, 0.76]
    ncells = 16

    plot_dir = f'/home/matylda/data/plots/'

    fig, ax = plt.subplots(2, 1, figsize=(15, 10),
                           gridspec_kw={'height_ratios': [2, 1]},
                           sharey='row', sharex='col')

    # Legend setup
    rho_lines = [Line2D([0, 1], [0, 1], color=c, ls=ls, lw=1.5)
                 for c, ls in zip([ssfr_colors[0], cb_blue, cb_red], rho_ls[:3])]
    leg = ax[0].legend(rho_lines, rho_labels, loc=3, fontsize=14)
    ax[0].add_artist(leg)

    i = 0

    for l, line in enumerate(lines):

        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion.h5'
        plot_data = read_h5_into_dict(cddf_file)
        completeness = plot_data['completeness']
        print(f'Line {line}: {completeness}')

        xerr = np.array([
            (plot_data['bin_edges_logN'][k + 1] - plot_data['bin_edges_logN'][k]) * 0.5
            for k in range(len(plot_data['plot_logN']))
        ])

        ax[i + 1].axhline(0, c='k', lw=0.8, ls='-')

        for key in ['cddf_all_poisson', 'cddf_sf_poisson',
                    'cddf_gv_poisson', 'cddf_q_poisson']:
            plot_data[key][np.isnan(plot_data[key])] = 0

        # Combine Poisson + CV errors
        plot_data[f'cddf_all_err'] = np.sqrt(
            plot_data[f'cddf_all_cv_{ncells}']**2. + plot_data[f'cddf_all_poisson']**2.)
        plot_data[f'cddf_sf_err'] = np.sqrt(
            plot_data[f'cddf_sf_cv_{ncells}']**2. + plot_data[f'cddf_sf_poisson']**2.)
        plot_data[f'cddf_gv_err'] = np.sqrt(
            plot_data[f'cddf_gv_cv_{ncells}']**2. + plot_data[f'cddf_gv_poisson']**2.)
        plot_data[f'cddf_q_err'] = np.sqrt(
            plot_data[f'cddf_q_cv_{ncells}']**2. + plot_data[f'cddf_q_poisson']**2.)

        # --- Plot ALL ---
        ax[i].errorbar(plot_data['plot_logN'], plot_data[f'cddf_all'],
                       c=ssfr_colors[0], yerr=plot_data[f'cddf_all_err'],
                       xerr=xerr, capsize=4, ls='-', lw=1)
        ax[i].axvline(completeness, c='k', ls=':', lw=1)
        ax[i + 1].axvline(completeness, c='k', ls=':', lw=1)

        for az_idx, az_label in enumerate(az_labels):
            offset = (-1)**az_idx * 0.05
            mock_cddf = mock_cddf = plot_data[f'cddf_{az_label}']
            ax[i].plot(plot_data['plot_logN'], mock_cddf,
                       c=az_colors[az_idx], ls=az_ls[az_idx], lw=1.5,
                       label=f'{az_label.capitalize()} axis')

            # fractional difference relative to all
            ax[i + 1].plot(plot_data['plot_logN'],
                           mock_cddf - plot_data['cddf_all'],
                           c=az_colors[az_idx], ls=az_ls[az_idx], lw=1.5)

        ax_top = ax[i].secondary_xaxis('top')

        ax[i].set_xlim(logN_min, 18)
        ax[i].set_ylim(-19, -9)
        ax[i + 1].set_xlim(logN_min, 18)
        ax[i + 1].set_ylim(-1.25, 1.25)

        ax[i + 1].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
        ax[i].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
        ax[i + 1].set_ylabel(r'$\Delta {\rm CDDF}$')

        ax[i].annotate(plot_lines[l], xy=(x[l], 0.86), xycoords='axes fraction',
                       bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if line in ['OVI1031']:
            ax[i].set_xticks(range(11, 19))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    outname = f'{plot_dir}{model}_{wind}_{snap}_cddf_major_minor_mock_{ncells}.png'
    plt.savefig(outname, format='png')
    plt.close()

    print(f"Saved plot to {outname}")