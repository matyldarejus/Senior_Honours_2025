# Generate the sm and sfr bins for plotting


import caesar, yt, numpy as np, h5py, sys
model, wind, snap = sys.argv[1], sys.argv[2], sys.argv[3]
data_dir = f'/disk04/rad/sim/{model}/{wind}/'
sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

# get galaxy stellar mass and SFR in log space (match other scripts)
gal_sm = np.log10(np.array([g.masses['stellar'].in_units('Msun') for g in sim.galaxies]))
gal_sfr = np.log10(np.array([g.sfr.in_units('Msun/yr') for g in sim.galaxies]) + 1e-6)  # small floor to avoid -inf

# define bins (choose same as plotting script expects)
mass_bins = np.arange(9.5, 12.5, 0.1)    # adjust binning as you prefer
sfr_bins  = np.arange(-4.0, 2.0, 0.1)

# 2D histogram (mass on x, sfr on y) - note histogram2d returns (H, xedges, yedges)
H, xedges, yedges = np.histogram2d(gal_sm, gal_sfr, bins=[mass_bins, sfr_bins])

out_file = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}_sm_sfr.h5'
with h5py.File(out_file, 'w') as hf:
    hf.create_dataset('sm_sfr', data=H.T)        # transpose if plotting expects shape (sfr_bins, mass_bins)
    hf.create_dataset('mass_bins', data=xedges)
    hf.create_dataset('sfr_bins',f'Wata=d{ges)

}.'p