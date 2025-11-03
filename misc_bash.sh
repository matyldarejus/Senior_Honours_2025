#!/bin/bash

MODEL=$1
WIND=$2
SNAP=$3
NLOS=$4

FR_VALUES=(0.25 0.5 0.75 1.0 1.25)
for FR in "${FR_VALUES[@]}"; do
    python ~/sh/analyse_spectra_sh/gather_line_results.py $MODEL $WIND $SNAP $FR $NLOS ||
done
