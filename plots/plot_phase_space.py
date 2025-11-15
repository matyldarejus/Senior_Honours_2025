# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025

import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import h5py
import pygad as pg
import sys

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
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap

def quench_thresh(z): # in units of yr^-1
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = sys.argv[4]

    cmap = plt.get_cmap('Greys')
    cmap = truncate_colormap(cmap, 0.0, .6)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    l = lines.index(line)

    width = 0.01
    height = 0.7
    vertical_position = 0.14
    horizontal_position = 0.9

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    ssfr_labels = ['All', 'Star forming', 'Green valley', 'Quenched']

    chisq_lim = 4.5
    N_min = [12., 11., 12., 11., 12., 12.]
    
    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    phase_space_file = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}_phase_space.h5'
    with h5py.File(phase_space_file, 'r') as hf:
        rho_overdensity_temp_hist2d = hf['rho_delta_temp'][:]
        rho_overdensity_bins = hf['rho_delta_bins'][:]
        temp_bins = hf['temp_bins'][:]

    plot_dir = f'/home/matylda/data/plots/'
    sample_dir = f'/disk04/mrejus/sh/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    fig, ax = plt.subplots(2, 4, figsize=(15, 6), sharey='row', sharex='col')
    
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_OVI1031.h5'

    for i in range(len(inner_outer)):

        for j in range(4):
            ax[i][j].imshow(np.log10(rho_overdensity_temp_hist2d), extent=(rho_overdensity_bins[0], rho_overdensity_bins[-1], temp_bins[0], temp_bins[-1]),
                            cmap=cmap)

        all_T = []
        all_rho = []
        all_N = []
        all_chisq = []
        all_ids = []

        for j in range(len(inner_outer[i])):

            with h5py.File(results_file, 'r') as hf:
                all_T.extend(hf[f'log_T_{inner_outer[i][j]}r200'][:])
                all_rho.extend(hf[f'log_rho_{inner_outer[i][j]}r200'][:])
                all_N.extend(hf[f'log_N_{inner_outer[i][j]}r200'][:])
                all_chisq.extend(hf[f'chisq_{inner_outer[i][j]}r200'][:])
                all_ids.extend(hf[f'ids_{inner_outer[i][j]}r200'][:])

        all_T = np.array(all_T)
        all_rho = np.array(all_rho)
        all_N = np.array(all_N)
        all_chisq = np.array(all_chisq)
        all_ids = np.array(all_ids)

        mask = (all_N > N_min[l]) * (all_chisq < chisq_lim)
        all_T = all_T[mask]
        all_delta_rho = all_rho[mask] - np.log10(cosmic_rho)
        all_ids = all_ids[mask]
        all_N = all_N[mask]

        idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
        all_mass = mass[idx]
        all_ssfr = ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

        if line == 'H1215':
            im = ax[i][0].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=16)
            im = ax[i][1].scatter(all_delta_rho[sf_mask], all_T[sf_mask], c=all_N[sf_mask], cmap='magma', s=1, vmin=N_min[l], vmax=16)
            im = ax[i][2].scatter(all_delta_rho[gv_mask], all_T[gv_mask], c=all_N[gv_mask], cmap='magma', s=1, vmin=N_min[l], vmax=16)
            im = ax[i][3].scatter(all_delta_rho[q_mask], all_T[q_mask], c=all_N[q_mask], cmap='magma', s=1, vmin=N_min[l], vmax=16)
        else:
            im = ax[i][0].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=15)
            im = ax[i][1].scatter(all_delta_rho[sf_mask], all_T[sf_mask], c=all_N[sf_mask], cmap='magma', s=1, vmin=N_min[l], vmax=15)
            im = ax[i][2].scatter(all_delta_rho[gv_mask], all_T[gv_mask], c=all_N[gv_mask], cmap='magma', s=1, vmin=N_min[l], vmax=15)
            im = ax[i][3].scatter(all_delta_rho[q_mask], all_T[q_mask], c=all_N[q_mask], cmap='magma', s=1, vmin=N_min[l], vmax=15)
     
        for j in range(4):
            ax[i][j].set_xlim(-1, 5)
            ax[i][j].set_ylim(3, 7)

        if i == 0:
            for j in range(4):
                ax[i][j].set_title(ssfr_labels[j])
        if i == 1:
            for j in range(4):
                ax[i][j].set_xlabel(r'${\rm log }\delta$')
        ax[i][0].set_ylabel(r'${\rm log } (T / {\rm K})$')

    cax = plt.axes([horizontal_position, vertical_position, width, height])
    fig.colorbar(im, cax=cax, label=r'${\rm log }(N / {\rm cm}^{-2})$')

    ax[0][0].annotate('Inner CGM', xy=(0.65, 0.85), xycoords='axes fraction')
    ax[1][0].annotate('Outer CGM', xy=(0.65, 0.85), xycoords='axes fraction')

    #plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_deltaTN_{line}.png', format)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_deltaTN_{line}.pdf', format='pdf')
    plt.close()
