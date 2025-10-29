#!/bin/bash


pipeline_path=~/sh/make_spectra_sh/pipeline.py
model=$1
wind=$2
snap=$3
line=$4

for ii in {0..216}
do
   echo Submitting job $ii
   python $pipeline_path $model $wind $snap $ii $line
   echo Finished job $ii
done
