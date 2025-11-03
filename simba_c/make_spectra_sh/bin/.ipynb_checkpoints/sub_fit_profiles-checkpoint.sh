#!/bin/bash

pipeline_path=/home/matylda/sh/make_spectra_sh/fit_profiles.py
model=$1
wind=$2
snap=$3
line=$4

for i in {0..1}; do
    echo "Running galaxy $i..."
    python $pipeline_path $model $wind $snap $i $line
done
