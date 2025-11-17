import yt
import trident

density = [1e-28, 1e-29]
temperature = [1e5, 1e5]
redshift = [0, 0]

trident.ion_balance.calculate_ion_fraction('O VI', density, temperature, redshift)