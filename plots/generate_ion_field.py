import caesar
import numpy as np
import h5py
import yt
import trident
import sys 

# Set the arguments
model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]
gal_id = sys.argv[4]

# Load the dataset
fn = f'/disk04/rad/sim/m100n1024/simba-c/snap_{model}_{snap}.hdf5'
ds = yt.load(fn)

# Cut out a sphere out of the dataset
# Load in the galaxy data and find the center of the galaxy

data_dir = f'/disk04/rad/sim/m100n1024/simba-c/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

gal_cent = np.array([i.central for i in sim.galaxies])[gal_id]
radius = np.array([i.halo.virial_quantities['r200c'].in_units('kpc/h') for i in sim.galaxies])[gal_id]

sp = ds.sphere(gal_cent, (radius, "kpc"))

# Add an ion field for OVI
trident.add_ion_fields(ds, ions=['O VI'])

# Initialise the projection plot
proj = yt.ProjectionPlot(ds, "z", "O_p5_number_density")
proj.set_cmap("O_p5_number_density", "viridis")

# Save the projection plot
plot_dir = '/home/matylda/data/plots/'
output_filename = f'{plot_dir}ion_field_{model}_{wind}_{snap}'
proj.save(f'{output_filename}.png', format='png', dpi=400)
proj.save(f'{output_filename}.pdf', format='pdf')
