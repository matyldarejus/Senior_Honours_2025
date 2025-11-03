# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025


import os
import sys
import numpy as np
from spectrum import Spectrum

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    i = int(sys.argv[4])
    ion = sys.argv[5]

    vel_range = 600.
    chisq_asym_thresh = -3
    chisq_unacceptable = 25

    #spec_dir = f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}_hm12/'
    spec_dir = f'./test/'
    listdir = os.listdir(spec_dir)
    spec_file = [i for i in listdir if ion in i]

    for my_file in spec_file:
        spec = Spectrum(f'{spec_dir}{my_file}')
        print('Fitting lines in: %s' % my_file)

        spec.main(
            vel_range=vel_range,
            do_fit=True,
            write_lines=True,
            chisq_unacceptable=chisq_unacceptable,
            chisq_asym_thresh=chisq_asym_thresh, 
            plot_fit=False,
            )
