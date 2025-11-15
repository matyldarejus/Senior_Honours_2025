# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib import cm
import matplotlib.patheffects as mpe
import numpy as np
import h5py
import pygad as pg
from scipy.optimize import curve_fit
import sys

sys.path.append('/home/matylda/sh/tools/')
import plotmedian as pm

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14.5)

def linear(x, a, b):
    return a*x + b

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

    outline = mpe.withStroke(linewidth=3, foreground='white')
    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.1, 1.0)
    
    """
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    """

    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']
    
    x = [0.74, 0.68, 0.715, 0.69, 0.695, 0.7]
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    deltath = 2.046913

    logN = np.arange(12.7, 18, 0.1)
    dave1999 = 0.7 * logN - 8.5
    inner_outer = [[0.25, 0.5], [0.75, 1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    nmin = 15
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    plot_dir = f'/home/matylda/data/plots/'
    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    #sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        all_ssfr = sf['ssfr'][:]

    cb_blue = '#5289C7'
    cb_green = '#90C987'
    cb_red = '#E26F72'

    cmap = colors.ListedColormap([cb_blue, cb_green, cb_red])
    norm = colors.BoundaryNorm(np.arange(0., 4., 1.), 3.)

    """
    fig, ax = plt.subplots(len(lines), len(inner_outer), figsize=(10, 13), sharey='row', sharex='col')

    ssfr_lines = []
    ssfr_lines.append(Line2D([0,1],[0,1], color='dimgrey', ls='--', lw=2))
    ssfr_lines.append(Line2D([0,1],[0,1], color='dimgrey', ls='-', lw=2))
    leg = ax[0][0].legend(ssfr_lines, ['Star forming', 'Quenched'], loc=4, fontsize=12)
    ax[0][0].add_artist(leg)

    ssfr_lines = []
    for c in cmap.colors:
        ssfr_lines.append(Line2D([0,1],[0,1], color=c, ls='', marker='o'))
    leg = ax[0][1].legend(ssfr_lines, ['Star forming', 'Green valley', 'Quenched'], loc=4, fontsize=12)
    ax[0][1].add_artist(leg)

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(inner_outer)):
            
            N = []
            rho = []
            chisq = []
            ids = []

            for j in range(len(inner_outer[i])):
                with h5py.File(results_file, 'r') as hf:
                    N.extend(hf[f'log_N_{inner_outer[i][j]}r200'][:])
                    rho.extend(hf[f'log_rho_{inner_outer[i][j]}r200'][:])
                    chisq.extend(hf[f'chisq_{inner_outer[i][j]}r200'][:])
                    ids.extend(hf[f'ids_{inner_outer[i][j]}r200'][:])
            
            N = np.array(N)
            rho = np.array(rho)
            chisq = np.array(chisq)
            ids = np.array(ids)

            mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
            delta_rho = rho[mask] - np.log10(cosmic_rho)
            ids = ids[mask]
            N = N[mask]

            idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
            ssfr = all_ssfr[idx]
            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr)

            ssfr_code = np.zeros(len(ssfr))
            ssfr_code[gv_mask] = 1.
            ssfr_code[q_mask] = 2.

            plot_order = np.arange(len(N))
            np.random.shuffle(plot_order)
            im = ax[l][i].scatter(N[plot_order], delta_rho[plot_order], c=ssfr_code[plot_order], cmap=cmap, s=1.5, norm=norm)
            #im = ax[l][i].scatter(N[gv_mask], delta_rho[gv_mask], c=ssfr_code[gv_mask], cmap=cmap, s=1.5, norm=norm)
            #ax[l][i].scatter(N[gv_mask], delta_rho[gv_mask], c=cb_green, s=1.5)
            #ax[l][i].scatter(N[q_mask], delta_rho[q_mask], c=cb_red, s=1.5)
            #ax[l][i].scatter(N[sf_mask], delta_rho[sf_mask], c=cb_blue, s=1.5)

            bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[sf_mask], delta_rho[sf_mask], stat='median')
            ax[l][i].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='dimgrey', lw=2, ls='--')
            bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[q_mask], delta_rho[q_mask], stat='median')
            ax[l][i].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='dimgrey', lw=2, ls='-')

            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
            
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm log }\delta$')
                ax[l][i].annotate(plot_lines[lines.index(line)], xy=(0.04, 0.1), xycoords='axes fraction', 
                                  bbox=dict(boxstyle="round", fc="w", lw=0.75))
            
            ax[l][i].set_xlim(np.min(N_min), 18)
            ax[l][i].set_ylim(-1, 5)
            ax[l][i].axhline(deltath, ls=':', c='k', lw=1)
    
    for i in range(2):
        ax[0][i].set_title(rho_labels[i])

    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    #fig.colorbar(im, cax=cbar_ax, label=r'$\log\ ({\rm sSFR} / {\rm Gyr}^{-1})$')
    
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Ndelta_ssfr.pdf', format='pdf')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Ndelta_ssfr.png', format='png')
    plt.close()

    """

    cmap = cm.get_cmap('viridis')
    cmap = truncate_colormap(cmap, 0.1, 0.9)
    norm = colors.BoundaryNorm(np.arange(0.125, 1.625, 0.25), cmap.N)

    fig, ax = plt.subplots(1, 3, figsize=(14, 13), sharey='row', sharex='col')

    rho_lines = []
    rho_lines.append(Line2D([0,1],[0,1], color='mediumorchid', ls='-', lw=1.25))
    rho_lines.append(Line2D([0,1],[0,1], color='mediumorchid', ls='--', lw=1.5))
    rho_lines.append(Line2D([0,1],[0,1], color='mediumorchid', ls='-.', lw=1.5))
    leg = ax[0].legend(rho_lines, rho_labels, loc=4, fontsize=12)
    ax[0].add_artist(leg)

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'
        #results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}_extras.h5'
        
        N = []
        rho = []
        chisq = []
        ids = []
        all_r = []

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                N.extend(hf[f'log_N_{fr200[i]}r200'][:])
                rho.extend(hf[f'log_rho_{fr200[i]}r200'][:])
                chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
                ids.extend(hf[f'ids_{fr200[i]}r200'][:])
                all_r.extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

        N = np.array(N)
        rho = np.array(rho)
        chisq = np.array(chisq)
        ids = np.array(ids)
        all_r = np.array(all_r)

        mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
        delta_rho = rho[mask] - np.log10(cosmic_rho)
        N = N[mask]
        all_r = all_r[mask]
        ids = ids[mask]

        idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
        ssfr = all_ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr)

        inner_mask = all_r < 0.75
        silly_mask = N < 18.
        fit_mask = N < 15. 

        plot_order = np.arange(len(N[sf_mask]))
        np.random.shuffle(plot_order)
        im = ax[0].scatter(N[sf_mask][plot_order], delta_rho[sf_mask][plot_order], c=all_r[sf_mask][plot_order], cmap=cmap, norm=norm, s=1, alpha=0.6)
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[sf_mask*inner_mask*silly_mask], delta_rho[sf_mask*inner_mask*silly_mask], stat='median')
        ax[0].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='mediumorchid', lw=1.5, ls='--', path_effects=[outline])
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[sf_mask*~inner_mask*silly_mask], delta_rho[sf_mask*~inner_mask*silly_mask], stat='median')
        ax[0].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='mediumorchid', lw=1.5, ls='-.', path_effects=[outline])

        plot_order = np.arange(len(N[gv_mask]))
        np.random.shuffle(plot_order)
        im = ax[1].scatter(N[gv_mask][plot_order], delta_rho[gv_mask][plot_order], c=all_r[gv_mask][plot_order], cmap=cmap, norm=norm, s=1, alpha=0.6)
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[gv_mask*inner_mask*silly_mask], delta_rho[gv_mask*inner_mask*silly_mask], stat='median')
        ax[1].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='mediumorchid', lw=1.5, ls='--', path_effects=[outline])
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[gv_mask*~inner_mask*silly_mask], delta_rho[gv_mask*~inner_mask*silly_mask], stat='median')
        ax[1].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='mediumorchid', lw=1.5, ls='-.', path_effects=[outline])
        
        plot_order = np.arange(len(N[q_mask]))
        np.random.shuffle(plot_order)
        im = ax[2].scatter(N[q_mask][plot_order], delta_rho[q_mask][plot_order], c=all_r[q_mask][plot_order], cmap=cmap, norm=norm, s=1, alpha=0.6)
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[q_mask*inner_mask*silly_mask], delta_rho[q_mask*inner_mask*silly_mask], stat='median')
        ax[2].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='mediumorchid', lw=1.5, ls='--', path_effects=[outline])
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[q_mask*~inner_mask*silly_mask], delta_rho[q_mask*~inner_mask*silly_mask], stat='median')
        ax[2].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='mediumorchid', lw=1.5, ls='-.', path_effects=[outline])

        if l == 0:
            ax[0].set_title('Star forming')
            ax[1].set_title('Green valley')
            ax[2].set_title('Quenched')

        if l == len(lines)-1:
            for i in range(3):
                ax[i].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
        
        ax[0].set_ylabel(r'${\rm log }\delta$')
        
        ax[2].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.1), xycoords='axes fraction',
                          fontsize=14, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        for i in range(3):
            ax[i].set_xlim(np.min(N_min), 18)
            ax[i].set_ylim(0, 4)
            ax[i].axhline(deltath, ls=':', c='k', lw=1)

        if l == 0:
            ax[0].set_yticks(np.arange(0, 5, 1))
        else:
            ax[0].set_yticks(np.arange(0, 4, 1))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.24, 0.02, 0.51])
    fig.colorbar(im, cax=cbar_ax, ticks=fr200, label=r'$r_\perp / r_{200}$')
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Ndelta_r200.png', format='png')
    #plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Ndelta_r200_extras.pdf', format='pdf')
    plt.close()
