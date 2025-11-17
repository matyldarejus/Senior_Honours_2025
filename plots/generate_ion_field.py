import yt
import trident
import sys 

# Set the arguments
model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

# Load the dataset
fn = "/disk04/rad/sim/{model}/{wind}/"
ds = yt.load(fn)

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
