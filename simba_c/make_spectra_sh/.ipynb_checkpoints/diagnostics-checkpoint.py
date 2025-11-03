import caesar
import yt
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt

sf = h5py.File("/disk04/mrejus/sh/normal/results/m25n256_s50_151_hm12_fit_lines_OVI1031.h5", "r")
print(sf.keys())

print(sf["chisq_0.25r200"])




