#!/bin/bash

for i in $(seq 0 3)
do
   sbatch slurm_script.sh $i
done