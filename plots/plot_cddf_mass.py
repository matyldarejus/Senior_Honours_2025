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
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.6
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130

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
    cmap = cm.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.2, .8)
    mass_colors = [cmap(i) for i in icolor]

    plot_dir = f'/home/matylda/data/plots/'

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]}, sharey='row', sharex='col')

    mass_lines = []
    mass_lines.append(Line2D([0,1],[0,1], color='dimgrey'))
    for i in range(len(mass_colors)):
        mass_lines.append(Line2D([0,1],[0,1], color=mass_colors[i]))
    leg = ax[0].legend(mass_lines, mass_plot_titles, loc=1, fontsize=20)
    ax[0].add_artist(leg)

    i = 0
    j = 0

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_OVI1031.h5'
        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_OVI1031_cddf_mass.h5'

        plot_data = read_h5_into_dict(cddf_file)
        completeness = plot_data['completeness']
        print(f'Line OVI1031: {completeness}')

        xerr = np.zeros(len(plot_data['plot_logN']))
        for k in range(len(plot_data['plot_logN'])):
            xerr[k] = (plot_data['bin_edges_logN'][k+1] - plot_data['bin_edges_logN'][k])*0.5

        ax[i+1].axhline(0, c='k', lw=1.5, ls='-')

        plot_data[f'cddf_all_err'] = np.sqrt(plot_data[f'cddf_all_cv_{ncells}']**2. + plot_data[f'cddf_all_poisson']**2.)
        ax[i].errorbar(plot_data['plot_logN'], plot_data[f'cddf_all'], c='dimgrey', yerr=plot_data[f'cddf_all_err'],
                          xerr=xerr, capsize=4, ls='-', lw=1.3)
        ax[i].axvline(plot_data['completeness'], c='k', ls=':', lw=1.5)
        ax[i+1].axvline(plot_data['completeness'], c='k', ls=':', lw=1.5)

        for k in range(len(mass_bin_labels)):
            ax[i].plot(plot_data['plot_logN'], plot_data[f'cddf_{mass_bin_labels[k]}'], c=mass_colors[k], ls='-', lw=1.5)

            ax[i+1].plot(plot_data['plot_logN'], (plot_data[f'cddf_{mass_bin_labels[k]}'] - plot_data[f'cddf_all']),
                            c=mass_colors[k], ls='-', lw=1.5)
 
    # Axes & labels
        ax[0].set_xlim(logN_min, 16.5)
        ax[0].set_ylim(-19, -9)
        ax[1].set_xlim(logN_min, 16.5)
        ax[1].set_ylim(-1.25, 1.25)

        ax[1].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
        ax[0].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
        ax[1].set_ylabel(r'$\Delta {\rm CDDF}$')

        #ax[0].annotate(plot_lines[l], xy=(0.76, 0.86), xycoords='axes fraction',
        #               bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if line in ['OVI1031']:
            ax[0].set_xticks(range(11, 19))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_mass.png', format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_mass.pdf', format='pdf')
    plt.close()

