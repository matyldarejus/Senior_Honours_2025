import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

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

    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']
    chisq_lim_dict = {'snap_151': [4.5],}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = [13.2]

    Tphoto_ovi = 5
    linestyles = ['--', ':']
    markers = ['s', '^']

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} M_\star < $' + f'{mass_bins[i+1]}')

    idelta = 1. / (len(mass_bins) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('plasma')
    cmap = truncate_colormap(cmap, 0.2, .8)
    mass_colors = [cmap(i) for i in icolor]

    plot_dir = '/home/matylda/data/plots/'
    sample_dir = f'/disk04/mrejus/sh/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]

    fig, ax = plt.subplots(1, 1)

    mass_lines = []
    for i in range(len(mass_colors)):
        mass_lines.append(Line2D([0,1],[0,1], color=mass_colors[i]))
    leg = ax.legend(mass_lines, mass_plot_titles, loc=3, fontsize=13)
    ax.add_artist(leg)

    ion_lines = []
    for i in range(len(lines)):
        ion_lines.append(Line2D([0,1],[0,1], ls=linestyles[i], marker=markers[i], color='dimgrey'))
    leg = ax.legend(ion_lines, plot_lines, loc=4, fontsize=14)
    ax.add_artist(leg)

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'

        fcol = np.zeros((len(mass_plot_titles), len(fr200)))
        ncol = np.zeros((len(mass_plot_titles), len(fr200)))
        ntotal = np.zeros((len(mass_plot_titles), len(fr200)))

        for i in range(len(fr200)):

            all_T = []
            all_N = []
            all_chisq = []
            all_ids = []

            with h5py.File(results_file, 'r') as hf:
                all_T.extend(hf[f'log_T_{fr200[i]}r200'][:])
                all_N.extend(hf[f'log_N_{fr200[i]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
                all_ids.extend(hf[f'ids_{fr200[i]}r200'][:])

            all_T = np.array(all_T)
            all_N = np.array(all_N)
            all_chisq = np.array(all_chisq)
            all_ids = np.array(all_ids)

            mask = (all_N > N_min[lines.index(line)]) * (all_chisq < chisq_lim[lines.index(line)])
            all_T = all_T[mask]
            all_N = all_N[mask]
            all_ids = all_ids[mask]

            idx = np.concatenate([np.where(gal_ids == gid)[0] for gid in all_ids if gid in gal_ids], axis=0)
            if len(idx) == 0:
                # Skip this radius bin if no matches
                continue
            all_mass = mass[idx.astype(int)]

            collisional = all_T > Tphoto_ovi # note: only for OVI
           
            for j in range(len(mass_plot_titles)):
                
                mass_mask = (all_mass > mass_bins[j]) & (all_mass < mass_bins[j+1])

                total_absorption = np.nansum(10**all_N[mass_mask])
                fcol[j][i] = np.nansum(10**all_N[mass_mask * collisional]) / total_absorption
                ncol[j][i] = len(all_N[mass_mask*collisional])
                ntotal[j][i] = len(all_N[mass_mask])
        
        for j in range(len(mass_plot_titles)):
            #ax.errorbar(fr200, fcol[j], yerr=poisson[j], color=mass_colors[j], ls=linestyles[l], marker=markers[l], lw=1.5, capsize=4)
            plt.plot(fr200, fcol[j], color=mass_colors[j], ls=linestyles[l], marker=markers[l], lw=1.5)

        print(fcol)

    plt.ylim(-0.05, 1)
    plt.ylabel(r'$\sum N_{\rm col} / \sum N_{\rm total}$')
    plt.xlabel(r'$r_\perp / r_{200}$')

    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcol.png', format='png')
    plt.close()