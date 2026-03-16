import h5py
import numpy as np

sample_file = '/disk04/mrejus/sh/samples/m100n1024_s50_151_galaxy_sample.h5'
zsolar = 5.79e-3

with h5py.File(sample_file, 'r') as sf:
    mass = sf['mass'][:]
    Z_sfr = sf['Z_sfr_weighted'][:]
    Z_cgm = sf['Z_cgm_mw'][:]

Z_ism = np.log10(Z_sfr) - np.log10(zsolar)
Z_cgm = np.log10(Z_cgm) - np.log10(zsolar)

mask = (mass >= 10.0) & (mass < 10.5)
print('N galaxies in bin:', np.sum(mask))
print('Z_ism range:', np.nanmin(Z_ism[mask]), np.nanmax(Z_ism[mask]))
print('Z_cgm range:', np.nanmin(Z_cgm[mask]), np.nanmax(Z_cgm[mask]))

