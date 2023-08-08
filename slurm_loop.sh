#!/bin/bash

for i in $(seq 0 11)
do
   sbatch slurm_script.sh $i
done