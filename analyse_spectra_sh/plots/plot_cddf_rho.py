# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025

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

    """
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    """

    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']
    
    labels = ['inner', 'outer']
    rho_labels = ['All CGM', 'Inner CGM', 'Outer CGM']
    ssfr_labels = ['All galaxies', 'Star forming', 'Green valley', 'Quenched']
    ssfr_colors = ['dimgrey', cb_blue, cb_green, cb_red]
    rho_ls = ['-', '--', ':']
    rho_lw = [1, 1.5, 2]
    logN_min = 11.
    x = [0.79, 0.74, 0.77, 0.75, 0.755, 0.76]
    ncells = 16

    plot_dir = f'/home/matylda/data/plots/'

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]}, sharey='row', sharex='col')

    #ssfr_lines = []
    #for i in range(len(ssfr_colors)):
    #    ssfr_lines.append(Line2D([0,1],[0,1], color=ssfr_colors[i]))
    #leg = ax[0].legend(ssfr_lines, ssfr_labels, loc=3, fontsize=14)
    #ax[0].add_artist(leg)

    rho_lines = []
    for i in range(len(rho_ls)):
        rho_lines.append(Line2D([0,1],[0,1], color=ssfr_colors[0], ls=rho_ls[i], lw=rho_lw[i]))
    leg = ax[0].legend(rho_lines, rho_labels, loc=3, fontsize=14)
    ax[0].add_artist(leg)

    i = 0
    j = 0

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'
        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion.h5'

        #results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}_extras.h5'
        #cddf_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion_extras.h5'

        plot_data = read_h5_into_dict(cddf_file)
        completeness = plot_data['completeness']
        print(f'Line {line}: {completeness}')

        xerr = np.zeros(len(plot_data['plot_logN']))
        for k in range(len(plot_data['plot_logN'])):
            xerr[k] = (plot_data['bin_edges_logN'][k+1] - plot_data['bin_edges_logN'][k])*0.5

        ax[i+1].axhline(0, c='k', lw=0.8, ls='-')

        plot_data[f'cddf_all_poisson'][np.isnan(plot_data[f'cddf_all_poisson'])] = 0
        plot_data[f'cddf_sf_poisson'][np.isnan(plot_data[f'cddf_sf_poisson'])] = 0
        plot_data[f'cddf_gv_poisson'][np.isnan(plot_data[f'cddf_gv_poisson'])] = 0
        plot_data[f'cddf_q_poisson'][np.isnan(plot_data[f'cddf_q_poisson'])] = 0

        plot_data[f'cddf_all_err'] = np.sqrt(plot_data[f'cddf_all_cv_{ncells}']**2. + plot_data[f'cddf_all_poisson']**2.)
        plot_data[f'cddf_sf_err'] = np.sqrt(plot_data[f'cddf_sf_cv_{ncells}']**2. + plot_data[f'cddf_sf_poisson']**2.)
        plot_data[f'cddf_gv_err'] = np.sqrt(plot_data[f'cddf_gv_cv_{ncells}']**2. + plot_data[f'cddf_gv_poisson']**2.)
        plot_data[f'cddf_q_err'] = np.sqrt(plot_data[f'cddf_q_cv_{ncells}']**2. + plot_data[f'cddf_q_poisson']**2.)
      
        plot_data[f'cddf_all_sf_err'] = np.sqrt(plot_data[f'cddf_all_err']**2 + plot_data[f'cddf_sf_err']**2)
        plot_data[f'cddf_all_gv_err'] = np.sqrt(plot_data[f'cddf_all_err']**2 + plot_data[f'cddf_gv_err']**2)
        plot_data[f'cddf_all_q_err'] = np.sqrt(plot_data[f'cddf_all_err']**2 + plot_data[f'cddf_q_err']**2)

        #if line == 'MgII2796':
        #    plot_data[f'cddf_all_q_err'][7] = (plot_data[f'cddf_all_q_err'][6] + plot_data[f'cddf_all_q_err'][8]) * 0.5
        #    plot_data[f'cddf_all_gv_err'][7] = (plot_data[f'cddf_all_gv_err'][5] + plot_data[f'cddf_all_gv_err'][6]) * 0.5

        ax[i].errorbar(plot_data['plot_logN'], plot_data[f'cddf_all'], c=ssfr_colors[0], yerr=plot_data[f'cddf_all_err'], 
                          xerr=xerr, capsize=4, ls=rho_ls[0], lw=1)
        ax[i].axvline(plot_data['completeness'], c='k', ls=':', lw=1)
        ax[i+1].axvline(plot_data['completeness'], c='k', ls=':', lw=1)

        #for k in range(len(labels)):

        #    ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_all_{labels[k]}'], c=ssfr_colors[0], ls=rho_ls[k+1], lw=1.5)
        #    ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_sf_{labels[k]}'], c=ssfr_colors[1], ls=rho_ls[k+1], lw=rho_lw[k+1])
        #    ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_gv_{labels[k]}'], c=ssfr_colors[2], ls=rho_ls[k+1], lw=rho_lw[k+1])
        #    ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_q_{labels[k]}'], c=ssfr_colors[3], ls=rho_ls[k+1], lw=rho_lw[k+1])

        #    ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_sf_{labels[k]}'] - plot_data[f'cddf_all']), 
        #                    c=ssfr_colors[1], ls=rho_ls[k+1], lw=rho_lw[k+1])
        #    ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_gv_{labels[k]}'] - plot_data[f'cddf_all']), 
        #                    c=ssfr_colors[2], ls=rho_ls[k+1], lw=rho_lw[k+1])
        #    ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_q_{labels[k]}'] - plot_data[f'cddf_all']), 
        #                    c=ssfr_colors[3], ls=rho_ls[k+1], lw=rho_lw[k+1])
       
        ax[i].plot(plot_data['plot_logN'], plot_data[f'cddf_sf'], c=ssfr_colors[1], ls='-', lw=1)
        ax[i].plot(plot_data['plot_logN'], plot_data[f'cddf_gv'], c=ssfr_colors[2], ls='-', lw=1)
        ax[i].plot(plot_data['plot_logN'], plot_data[f'cddf_q'], c=ssfr_colors[3], ls='-', lw=1)

        ax[i+1].errorbar(plot_data['plot_logN'] - 0.05, (plot_data[f'cddf_sf'] - plot_data[f'cddf_all']), yerr=plot_data[f'cddf_all_sf_err'],
                            xerr=xerr, c=ssfr_colors[1], capsize=4, ls='-', lw=1)

        ax[i+1].errorbar(plot_data['plot_logN'], (plot_data[f'cddf_gv'] - plot_data[f'cddf_all']), yerr=plot_data[f'cddf_all_gv_err'],
                            xerr=xerr, c=ssfr_colors[2], capsize=4, ls='-', lw=1)

        ax[i+1].errorbar(plot_data['plot_logN'] + 0.05, (plot_data[f'cddf_q'] - plot_data[f'cddf_all']), yerr=plot_data[f'cddf_all_q_err'],
                            xerr=xerr, c=ssfr_colors[3], capsize=4, ls='-', lw=1)

        #ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_sf'] - plot_data[f'cddf_all']),
        #                c=ssfr_colors[1], ls='-', lw=1)
        #ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_gv'] - plot_data[f'cddf_all']),
        #                c=ssfr_colors[2], ls='-', lw=1)
        #ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_q'] - plot_data[f'cddf_all']),
        #                c=ssfr_colors[3], ls='-', lw=1)

        ax_top = ax[i].secondary_xaxis('top')

        ax[i].set_xlim(logN_min, 18)
        ax[i].set_ylim(-19, -9)

        ax[i+1].set_xlim(logN_min, 18)
        ax[i+1].set_ylim(-1.25, 1.25)

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
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_compressed_chisqion_{ncells}.png', format='png')
    #plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_compressed_chisqion_{ncells}_extras.pdf', format='pdf')
    plt.close()
