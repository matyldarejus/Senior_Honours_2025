import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import AutoMinorLocator
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
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", cmap_list)
    return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    ion = "OVI1031"
    plot_label = "OVI"
    ion_ev = np.log10(138.1)  # in eV
    adjust_x = 0.025
    N_min = 13.2
    zsolar = 5.79e-3

    chisq_lim_dict = {
        'snap_151': 4.5,
        'snap_137': 4.5,
        'snap_125': 5.6,
        'snap_105': 7.1,
    }
    chisq_lim = chisq_lim_dict[f'snap_{snap}']

    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    deltath = 2.046913
    Tth = 5.0

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200 + 1) * delta_fr200, delta_fr200)

    idelta = 0.8 / (len(fr200) - 1)
    icolor = np.arange(0.1, 0.9 + idelta, idelta)
    cmap = cm.get_cmap('viridis')
    cmap = truncate_colormap(cmap, 0.1, 0.9)
    color_list = [cmap(i) for i in icolor]
    norm = colors.BoundaryNorm(np.arange(0.125, 1.625, 0.25), cmap.N)

    plot_dir = '/home/matylda/data/plots/'

    fig, ax = plt.subplots(3, 1, figsize=(7, 6.5), sharey='row', sharex='col')
    ax = ax.flatten()

    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{ion}.h5'

    weighted_D = np.full(len(fr200), np.nan)
    weighted_D_25 = np.full(len(fr200), np.nan)
    weighted_D_75 = np.full(len(fr200), np.nan)
    weighted_T = np.full(len(fr200), np.nan)
    weighted_T_25 = np.full(len(fr200), np.nan)
    weighted_T_75 = np.full(len(fr200), np.nan)
    weighted_Z = np.full(len(fr200), np.nan)
    weighted_Z_25 = np.full(len(fr200), np.nan)
    weighted_Z_75 = np.full(len(fr200), np.nan)

    for i, fr in enumerate(fr200):

        with h5py.File(results_file, 'r') as hf:
            all_Z = hf[f'log_Z_{fr}r200'][:] - np.log10(zsolar)
            all_T = hf[f'log_T_{fr}r200'][:]
            all_D = hf[f'log_rho_{fr}r200'][:] - np.log10(cosmic_rho)
            all_N = hf[f'log_N_{fr}r200'][:]
            all_chisq = hf[f'chisq_{fr}r200'][:]

        mask = (all_N > N_min) & (all_chisq < chisq_lim)
        all_Z, all_T, all_D, all_N = all_Z[mask], all_T[mask], all_D[mask], all_N[mask]

        if len(all_N) == 0 or np.nansum(all_N) == 0:
            print(f"Empty selection for r = {fr:.2f}r200, skipping.")
            continue

        def weighted_percentile(x, w, q):
            """Compute the weighted percentile given data x, weights w, and percentile q (0-1)."""
            order = np.argsort(x)
            cumsum = np.nancumsum(w[order]) / np.nansum(w)
            return x[order][np.argmin(np.abs(cumsum - q))]

        weighted_D[i] = weighted_percentile(all_D, all_N, 0.5)
        weighted_D_25[i] = weighted_percentile(all_D, all_N, 0.25)
        weighted_D_75[i] = weighted_percentile(all_D, all_N, 0.75)

        weighted_T[i] = weighted_percentile(all_T, all_N, 0.5)
        weighted_T_25[i] = weighted_percentile(all_T, all_N, 0.25)
        weighted_T_75[i] = weighted_percentile(all_T, all_N, 0.75)

        weighted_Z[i] = weighted_percentile(all_Z, all_N, 0.5)
        weighted_Z_25[i] = weighted_percentile(all_Z, all_N, 0.25)
        weighted_Z_75[i] = weighted_percentile(all_Z, all_N, 0.75)

        # plot single points with errorbars
        ax[0].errorbar(ion_ev, weighted_D[i],
                       yerr=np.array([[weighted_D[i] - weighted_D_25[i],
                                       weighted_D_75[i] - weighted_D[i]]]).T,
                       color=color_list[i], lw=1.5, ls='None', capsize=2, alpha=0.6)
        ax[1].errorbar(ion_ev, weighted_T[i],
                       yerr=np.array([[weighted_T[i] - weighted_T_25[i],
                                       weighted_T_75[i] - weighted_T[i]]]).T,
                       color=color_list[i], lw=1.5, ls='None', capsize=2, alpha=0.6)
        ax[2].errorbar(ion_ev, weighted_Z[i],
                       yerr=np.array([[weighted_Z[i] - weighted_Z_25[i],
                                       weighted_Z_75[i] - weighted_Z[i]]]).T,
                       color=color_list[i], lw=1.5, ls='None', capsize=2, alpha=0.6)

    # scatter all points for the colorbar
    im = ax[0].scatter(np.repeat(ion_ev, len(fr200)), weighted_D, c=fr200, cmap=cmap, norm=norm, marker='o', alpha=0.6)
    ax[1].scatter(np.repeat(ion_ev, len(fr200)), weighted_T, c=fr200, cmap=cmap, norm=norm, marker='o', alpha=0.6)
    ax[2].scatter(np.repeat(ion_ev, len(fr200)), weighted_Z, c=fr200, cmap=cmap, norm=norm, marker='o', alpha=0.6)

    ax[0].annotate(plot_label, xy=(ion_ev - adjust_x, np.nanmin(weighted_D - 0.375)), fontsize=13)

    # horizontal reference lines
    ax[0].axhline(deltath, ls=':', c='k', lw=1)
    ax[1].axhline(Tth, ls=':', c='k', lw=1)

    # axis limits
    ax[0].set_ylim(1, 4.)
    ax[1].set_ylim(4, 5.7)
    ax[2].set_ylim(-1, )

    # labels
    ax[2].set_xlabel(r'${\rm log }(E / {\rm eV})$')
    ax[0].set_ylabel(r'${\rm log }\delta$')
    ax[1].set_ylabel(r'${\rm log } (T / {\rm K})$')
    ax[2].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')

    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(4))

    # colorbar
    fig.subplots_adjust(top=0.8)
    cbar_ax = fig.add_axes([0.125, 0.8, 0.775, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=fr200, orientation='horizontal')
    cbar.set_label(r'$r_\perp / r_{200}$', labelpad=8)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Nweighted_deltaTZ_{ion}.png', format='png')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Nweighted_deltaTZ_{ion}.pdf', format='pdf')
    plt.close()