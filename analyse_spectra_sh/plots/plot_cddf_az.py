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

# --- Colour palette ---
cb_blue = '#5289C7'
cb_red = '#E26F72'
cb_grey = 'dimgrey'

def stop_array_after_inf(array):
    mask = np.isinf(array)
    if np.any(mask):
        inf_start = np.where(mask)[0][0]
        array[inf_start:] = np.inf
    return array


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    # Which lines to plot
    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']

    # Azimuthal bins (major vs minor)
    az_labels = ['major', 'minor']
    az_colors = [cb_blue, cb_red]
    az_ls = ['-', '--']

    logN_min = 11.
    ncells = 16
    plot_dir = f'/home/matylda/data/plots/'

    fig, ax = plt.subplots(2, 1, figsize=(15, 10),
                           gridspec_kw={'height_ratios': [2, 1]},
                           sharey='row', sharex='col')

    # Legend setup
    rho_labels = ['All CGM', 'Major Axis', 'Minor Axis']
    rho_colors = [cb_grey, cb_blue, cb_red]
    rho_ls = ['-', '--', '--']
    rho_lines = [Line2D([0, 1], [0, 1], color=c, ls=ls, lw=1.5)
                 for c, ls in zip(rho_colors, rho_ls)]
    leg = ax[0].legend(rho_lines, rho_labels, loc=3, fontsize=14)
    ax[0].add_artist(leg)

    for l, line in enumerate(lines):

        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_azimuth.h5'
        plot_data = read_h5_into_dict(cddf_file)
        completeness = plot_data['completeness']

        xerr = np.array([
            (plot_data['bin_edges_logN'][k + 1] - plot_data['bin_edges_logN'][k]) * 0.5
            for k in range(len(plot_data['plot_logN']))
        ])

        ax[1].axhline(0, c='k', lw=0.8, ls='-')


        for key in ['cddf_all_poisson', 'cddf_major_poisson', 'cddf_minor_poisson']:
            plot_data[key][np.isnan(plot_data[key])] = 0

        # Combined errors (Poisson + cosmic variance)
        plot_data['cddf_all_err'] = np.sqrt(
            plot_data[f'cddf_all_cv_{ncells}']**2. + plot_data['cddf_all_poisson']**2.)
        plot_data['cddf_major_err'] = np.sqrt(
            plot_data[f'cddf_major_cv_{ncells}']**2. + plot_data['cddf_major_poisson']**2.)
        plot_data['cddf_minor_err'] = np.sqrt(
            plot_data[f'cddf_minor_cv_{ncells}']**2. + plot_data['cddf_minor_poisson']**2.)
        
        ax[0].axvline(plot_data['completeness'], c='k', ls=':', lw=1)
        ax[1].axvline(plot_data['completeness'], c='k', ls=':', lw=1)


    
        ax[0].errorbar(plot_data['plot_logN'], plot_data['cddf_all'],
                       c=cb_grey, yerr=plot_data['cddf_all_err'],
                       xerr=xerr, capsize=4, ls='-', lw=1.5,
                       label='All CGM')

        
        for az_idx, az_label in enumerate(az_labels):
            mock_cddf = plot_data[f'cddf_{az_label}']
            ax[0].plot(plot_data['plot_logN'], mock_cddf,
                       c=az_colors[az_idx], ls=az_ls[az_idx], lw=1.5,
                       label=f'{az_label.capitalize()} axis')

            # fractional difference relative to all
            ax[1].plot(plot_data['plot_logN'],
                       mock_cddf - plot_data['cddf_all'],
                       c=az_colors[az_idx], ls=az_ls[az_idx], lw=1.5)

        # Axes & labels
        ax[0].set_xlim(logN_min, 18)
        ax[0].set_ylim(-19, -9)
        ax[1].set_xlim(logN_min, 18)
        ax[1].set_ylim(-1.25, 1.25)

        ax[1].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
        ax[0].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
        ax[1].set_ylabel(r'$\Delta {\rm CDDF}$')

        ax[0].annotate(plot_lines[l], xy=(0.76, 0.86), xycoords='axes fraction',
                       bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if line in ['OVI1031']:
            ax[0].set_xticks(range(11, 19))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)

    outname = f'{plot_dir}{model}_{wind}_{snap}_cddf_major_minor_{ncells}.png'
    plt.savefig(outname, format='png', dpi=200)
    plt.close()