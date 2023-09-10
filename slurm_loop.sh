#!/bin/bash

for i in $(seq 12 23)
do
   sbatch slurm_script.sh $i
done