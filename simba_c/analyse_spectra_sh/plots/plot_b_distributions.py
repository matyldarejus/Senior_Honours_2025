# Plot b-parameter distributions for different ssfr types
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14.5)

def quench_thresh(z):  # sSFR threshold in yr^-1
    return -1.8 + 0.3*z - 9.

def ssfr_type_check(ssfr_thresh, ssfr):
    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh - 1)
    q_mask = (ssfr == -14.0)
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = "OVI1031"

    plot_dir = '/home/matylda/data/plots/'
    sample_dir = '/disk04/mrejus/sh/samples/'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    snapfile = f'{sample_dir}{model}_{wind}_{snap}.hdf5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        ssfr = sf['ssfr'][:]

    s = pg.Snapshot(snapfile)
    z = s.redshift
    quench = quench_thresh(z)
    sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr)

    cb_blue = '#5289C7'
    cb_green = '#90C987'
    cb_red = '#E26F72'
    cb_grey = '0.3'

    b_all, b_sf, b_gv, b_q = [], [], [], []

    # get all the b values
    with h5py.File(results_file, 'r') as f:
        for key in f.keys():
            if not key.startswith('b_'):
                continue
            b_vals = f[key][:]
            ids_key = key.replace('b_', 'ids_')
            if ids_key not in f:
                continue
            ids = f[ids_key][:]
            if len(b_vals) == 0:
                continue

            idx = np.array([np.where(gal_ids == g)[0][0] for g in ids])
            this_ssfr = ssfr[idx]
            sf_mask_i, gv_mask_i, q_mask_i = ssfr_type_check(quench, this_ssfr)

            b_all.extend(b_vals)
            b_sf.extend(b_vals[sf_mask_i])
            b_gv.extend(b_vals[gv_mask_i])
            b_q.extend(b_vals[q_mask_i])

    b_all, b_sf, b_gv, b_q = map(np.array, (b_all, b_sf, b_gv, b_q))

    fig, ax = plt.subplots(figsize=(7,5))
    bins = np.linspace(0, 150, 40)

    ax.hist(b_all, bins=bins, histtype='step', color=cb_grey, lw=2, label='Total')
    ax.hist(b_sf, bins=bins, histtype='step', color=cb_blue, lw=1.8, label='Star forming')
    ax.hist(b_gv, bins=bins, histtype='step', color=cb_green, lw=1.8, label='Green valley')
    ax.hist(b_q, bins=bins, histtype='step', color=cb_red, lw=1.8, label='Quenched')

    ax.set_xlabel(r'$b\ [{\rm km\ s^{-1}}]$')
    ax.set_ylabel('Count')
    ax.legend(fontsize=11)
    ax.set_title(f'{line} linewidth distributions', fontsize=15)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_bparam_dist_total.png', format='png')
    #plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_bparam_dist_total.pdf', format='pdf')
    plt.close()