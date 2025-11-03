# Very simple line fitting method


import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
import caesar
import yt
np.random.seed(1)

model = 'm25n256'
wind = 's50'
snap = '151'


data_dir = f'/disk04/mrejus/sh/samples/'
sim_dir = f'/disk04/rad/sim/{model}/{wind}/'
spectrum_dir = f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}_hm12/'



def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z - 9.

# load in the galaxy and halo catalog with caesar:

sim = caesar.load(f'{sim_dir}Groups/{model}_{snap}.hdf5')
redshift = sim.simulation.redshift
quench = quench_thresh(redshift)

gal_sm = yt.YTArray([i.masses['stellar'].in_units('Msun') for i in sim.galaxies], 'Msun')
gal_sfr = yt.YTArray([i.sfr.in_units('Msun/yr') for i in sim.galaxies], 'Msun/yr')
gal_central = np.array([i.central for i in sim.galaxies])
gal_ssfr = gal_sfr / gal_sm

# get ids of star forming galaxies:
sf_ids = np.arange(len(sim.galaxies))[(gal_ssfr > quench) & (gal_sm > 1e10) & gal_central]
gal = sim.galaxies[np.random.choice(sf_ids, 1)[0]]


with h5py.File(f'{data_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sample:
    gal_sm = sample['mass'][:]
    gal_sfr = sample['sfr'][:]
    #gal_fgas = sample['gas_frac'][:]
    gal_vpos = sample['vgal_position'][:]
    gal_ids = sample['gal_ids'][:]
    print(sample.keys())
#gal_fgas[gal_fgas == 0] = 1e-3

gal = gal_ids[3]
vgal = gal_vpos[3][2]
spectrum_file = f'{spectrum_dir}sample_galaxy_{gal}_OVI1031_0_deg_0.25r200.h5'

sf = h5py.File(spectrum_file, 'r')
sf.keys()


line = 'OVI1031'
vel_range = 600. # km/s

velocity = sf['velocity'][:]
wave = sf[f'{line}_wavelength'][:]
flux = sf[f'{line}_flux'][:]
noise = sf[f'{line}_noise'][:]
tau = sf[f'{line}_tau'][:]

fig, ax = plt.subplots(2, 1, figsize=(15, 6))
ax[0].plot(velocity, tau)
ax[0].set_ylabel(r'$\tau$')
ax[0].axvline(vgal, c='m')
ax[1].plot(velocity, flux)
ax[1].set_ylabel('Flux')
ax[1].set_xlabel('V (km/s)')
ax[1].axvline(vgal, c='m')
plt.savefig(f'/home/matylda/tmp/plot1.png', format='png')


vel_mask = (velocity > vgal - vel_range) & (velocity < vgal + vel_range) 

fig, ax = plt.subplots(2, 1, figsize=(15, 6))
ax[0].plot(velocity[vel_mask], tau[vel_mask])
ax[0].set_ylabel(r'$\tau$')
ax[0].axvline(vgal, c='m')
ax[1].plot(velocity[vel_mask], flux[vel_mask])
ax[1].set_ylabel('Flux')
ax[1].set_xlabel('V (km/s)')
ax[1].axvline(vgal, c='m')
plt.savefig(f'/home/matylda/tmp/plot2.png', format='png')

line_list = pg.analysis.fit_profiles(line, wave[vel_mask], flux[vel_mask]+noise[vel_mask], noise[vel_mask], chisq_lim=2.0, max_lines=7, logN_bounds=[11,17], b_bounds=[3,100], mode='Voigt')


print(line_list)


fig, ax = plt.subplots(1, 1)
model_flux, N, dN, b, db, l, dl, EW = pg.analysis.plot_fit(ax, wave[vel_mask], flux[vel_mask]+noise[vel_mask], noise[vel_mask], line_list, line)
plt.savefig(f'/home/matylda/tmp/plot2.png', format='png')