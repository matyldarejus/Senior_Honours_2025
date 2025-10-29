# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as colors
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

def stop_array_after_inf(array):
    mask = np.isinf(array)
    if len(array[mask]) > 0:
        inf_start = np.where(mask)[0][0]
        array[inf_start:] = np.inf
        return array
    else:
        return array


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    """
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    """

    # IGNORE ABOVE AND FOCUS ON OVI1031 ONLY

    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']
    
    logN_min = 11.
    x = [0.79, 0.74, 0.77, 0.75, 0.755, 0.76]
    ncells = 16

    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    mass_bin_labels = []
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_\star / M_{\odot}) < $' + f'{mass_bins[i+1]}')
    mass_plot_titles.insert(0, 'All galaxies')

    idelta = 1. / (len(mass_bins) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('plasma')
    cmap = truncate_colormap(cmap, 0.2, .8)
    mass_colors = [cmap(i) for i in icolor]

    plot_dir = f'/home/matylda/data/plots/'

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]}, sharey='row', sharex='col')

    mass_lines = []
    mass_lines.append(Line2D([0,1],[0,1], color='dimgrey'))
    for i in range(len(mass_colors)):
        mass_lines.append(Line2D([0,1],[0,1], color=mass_colors[i]))
    leg = ax[0].legend(mass_lines, mass_plot_titles, loc=3, fontsize=14)
    ax[0].add_artist(leg)

    i = 0
    j = 0

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_fit_lines_OVI1031.h5'
        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_OVI1031_cddf_mass.h5'

        plot_data = read_h5_into_dict(cddf_file)
        completeness = plot_data['completeness']
        print(f'Line OVI1031: {completeness}')

        xerr = np.zeros(len(plot_data['plot_logN']))
        for k in range(len(plot_data['plot_logN'])):
            xerr[k] = (plot_data['bin_edges_logN'][k+1] - plot_data['bin_edges_logN'][k])*0.5

        ax[i+1].axhline(0, c='k', lw=0.8, ls='-')

        plot_data[f'cddf_all_err'] = np.sqrt(plot_data[f'cddf_all_cv_{ncells}']**2. + plot_data[f'cddf_all_poisson']**2.)
        ax[i].errorbar(plot_data['plot_logN'], plot_data[f'cddf_all'], c='dimgrey', yerr=plot_data[f'cddf_all_err'],
                          xerr=xerr, capsize=4, ls='-', lw=1)
        ax[i].axvline(plot_data['completeness'], c='k', ls=':', lw=1)
        ax[i+1].axvline(plot_data['completeness'], c='k', ls=':', lw=1)

        for k in range(len(mass_bin_labels)):
            ax[i].plot(plot_data['plot_logN'], plot_data[f'cddf_{mass_bin_labels[k]}'], c=mass_colors[k], ls='-', lw=1)

            ax[i+1].plot(plot_data['plot_logN'], (plot_data[f'cddf_{mass_bin_labels[k]}'] - plot_data[f'cddf_all']),
                            c=mass_colors[k], ls='-', lw=1)
 
        ax_top = ax[i].secondary_xaxis('top')
        ax_top.set_xticks(np.arange(logN_min, 18), labels=[])

        ax[i].set_xlim(logN_min, 18)
        ax[i].set_ylim(-19, -9)

        ax[i+1].set_xlim(logN_min, 18)
        ax[i+1].set_ylim(-0.75, 0.75)

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i+1].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

        if line in ['H1215', "SiIII1206"]:
            ax[i].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
            ax[i+1].set_ylabel(r'${\rm log}\ f_{\rm CDDF\ All}$')
        ax[i].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.86), xycoords='axes fraction',
                          bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if line in ['SiIII1206', 'CIV1548']:
            ax[i].set_xticks(range(11, 18))
        elif line in ['OVI1031']:
            ax[i].set_xticks(range(11, 19))

        j += 1
        if line == 'CII1334':
            i += 2
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_mass.png', format='png')
    plt.close()

