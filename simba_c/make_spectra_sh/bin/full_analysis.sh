#!/bin/bash

# ====== Arguments ======
MODEL=$1
WIND=$2
SNAP=$3
NLOS=$4
GAL_NO=$5

echo "============================================================"
echo " Gathering and analysing data for:"
echo " Model: $MODEL | Wind: $WIND | Snapshot: $SNAP | NLOS: $NLOS"
echo "============================================================"
echo ""

# ====== Run for all galaxies ======
for GALID in $(seq 0 $GAL_NO); do
    echo ">>> Starting galaxy $GALID"
    bash ~/sh/make_spectra_sh/run_full_pipeline.sh $MODEL $WIND $SNAP $GALID $NLOS
    echo ">>> Finished galaxy $GALID"
done

# ====== Compile final results ======
echo "============================================================"
echo " All galaxy pipelines complete — compiling results..."
echo "============================================================"

bash ~/sh/make_spectra_sh/get_results.sh $MODEL $WIND $SNAP $NLOS

echo ""
echo "============================================================"
echo " All processing and result compilation finished!"
echo "============================================================"NLOS
done