import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.lines import Line2D
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

def quench_thresh(z):
    return -1.8 + 0.3*z - 9.

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    line = "OVI1031"
    chisq_lim_dict = {'snap_151': 4.5}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = 13.2
    Tphoto_ovi = 5.0

    linestyle = ':'
    marker = '^'

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200 + 1) * delta_fr200, delta_fr200)

    plot_dir = '/home/matylda/data/plots/'
    sample_dir = '/disk04/mrejus/sh/samples/'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        ssfr = sf['ssfr'][:]

    s = pg.Snapshot(f'{sample_dir}{model}_{wind}_{snap}.hdf5')
    redshift = s.redshift
    ssfr_th = quench_thresh(redshift)

    ssfr_labels = ["All", "Star-forming", "Green Valley", "Quenched"]
    ssfr_colors = ["black", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(1, 1)

    type_lines = [Line2D([0, 1], [0, 1], color=ssfr_colors[i])
                  for i in range(len(ssfr_labels))]
    leg1 = ax.legend(type_lines, ssfr_labels, loc=3, fontsize=20)
    ax.add_artist(leg1)

    fcol = np.zeros((4, len(fr200)))
    ncol = np.zeros((4, len(fr200)))
    ntotal = np.zeros((4, len(fr200)))

    with h5py.File(results_file, 'r') as hf:
        for i in range(len(fr200)):

            r = fr200[i]

            try:
                all_T = np.array(hf[f'log_T_{r}r200'][:])
                all_N = np.array(hf[f'log_N_{r}r200'][:])
                all_chisq = np.array(hf[f'chisq_{r}r200'][:])
                all_ids = np.array(hf[f'ids_{r}r200'][:])
            except KeyError:
                continue

            mask = (all_N > N_min) & (all_chisq < chisq_lim)
            all_T = all_T[mask]
            all_N = all_N[mask]
            all_ids = all_ids[mask]

            if len(all_ids) == 0:
                continue

            idx = np.array([np.where(gal_ids == gid)[0][0]
                            for gid in all_ids if gid in gal_ids])

            if len(idx) == 0:
                continue

            all_ssfr = ssfr[idx]
            collisional = all_T > Tphoto_ovi

            all_mask = np.ones_like(all_ssfr, dtype=bool)
            sf_mask = all_ssfr >= ssfr_th
            gv_mask = (all_ssfr < ssfr_th) & (all_ssfr > ssfr_th - 1.0)
            q_mask = all_ssfr == -14.0

            type_masks = [all_mask, sf_mask, gv_mask, q_mask]

            for j in range(4):
                m = type_masks[j]
                total_abs = np.nansum(10**all_N[m])
                if total_abs > 0:
                    fcol[j][i] = np.nansum(10**all_N[m & collisional]) / total_abs
                else:
                    fcol[j][i] = np.nan
                ncol[j][i] = np.sum(m & collisional)
                ntotal[j][i] = np.sum(m)

    for j in range(4):
        plt.plot(fr200, fcol[j], color=ssfr_colors[j],
                 ls=linestyle, marker=marker, lw=1.5)

    plt.ylim(-0.05, 1)
    plt.ylabel(r'$\sum N_{\rm col} / \sum N_{\rm total}$')
    plt.xlabel(r'$r_\perp / r_{200}$')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcol_OVI.png', format='png')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcol_OVI.pdf', format='pdf')
    plt.close()
