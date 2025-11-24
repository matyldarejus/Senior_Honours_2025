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
gal_id = int(sys.argv[4])

# Load the dataset
fn = f'/disk04/rad/sim/m100n1024/simba-c/snap_{model}_{snap}.hdf5'
ds = yt.load(fn)


# Load in the galaxy data and find the center of the galaxy

data_dir = f'/disk04/rad/sim/m100n1024/simba-c/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

gal_cent = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')[gal_id]
print(gal_cent)
radius = np.array([i.halo.virial_quantities['r200c'].in_units('kpc/h') for i in sim.galaxies])[gal_id]
print(radius)



# Add an ion field for OVI
trident.add_ion_fields(ds, ions=['O VI'])

# Cut out a sphere out of the dataset
sp = ds.sphere(gal_cent, (radius, "kpc/h"))

# Initialise the projection plot
proj = yt.ProjectionPlot(ds, "z", ("gas", "density"), data_source=sp, center=gal_cent, width=(1.4, "Mpc"))
proj.set_cmap(("gas", "density"), "magma")

# Do the slice plot
slice_plot = yt.SlicePlot(ds, "z", ("gas", "density"), center=gal_cent)
slice_plot.set_cmap(("gas", "density"), "magma")

# Save the plots
plot_dir = '/home/matylda/data/plots/'
output_filename = f'{plot_dir}ion_field_{model}_{wind}_{snap}'
proj.save(f'{output_filename}.png')
proj.save(f'{output_filename}.pdf')

slice_plot.save(f"{output_filename}_slice_z0.png")