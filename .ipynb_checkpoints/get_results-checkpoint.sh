#!/bin/bash

# Usage: bash get_results.sh MODEL WIND SNAP NLOS (one galaxy only)

# ====== Arguments ======
MODEL=$1
WIND=$2
SNAP=$3
NLOS=$4

# ====== Paths ======
MAKE_SPECTRA=~/sh/make_spectra_sh
ANALYSE_SPECTRA=~/sh/analyse_spectra_sh
SUBMISSION=$MAKE_SPECTRA/submission
PLOTS_DIR=/home/matylda/plots

echo "============================================================"
echo "   Compiling results for $MODEL $WIND $SNAP $NLOS"
echo "============================================================"

# Step 1: Plot galaxy sample
echo "[1/5] Running plot_galaxy_sample..."
python $ANALYSE_SPECTRA/plots/plot_galaxy_sample.py $MODEL $WIND $SNAP ||

# Step 2: Get cddf
echo "[2/5] Running get_cddf..."
python $ANALYSE_SPECTRA/get_cddf.py $MODEL $WIND $SNAP $NLOS ||

# Step 3: Get cddf mass
echo "[3/5] Running get_cddf_mass..."
python $ANALYSE_SPECTRA/get_cddf_mass.py $MODEL $WIND $SNAP $NLOS ||

# Step 4: Plot cddf
echo "[4/5] Running plot_cddf..."
python $ANALYSE_SPECTRA/plots/plot_cddf.py $MODEL $WIND $SNAP ||

# Step 5: Plot cddf_mass
echo "[5/5] Running plot_cddf_mass..."
python $ANALYSE_SPECTRA/plots/plot_cddf_mass.py $MODEL $WIND $SNAP ||

echo ""
echo "============================================================"
echo " Results pipeline finished for $MODEL $WIND $SNAP $NLOS"
echo "============================================================"