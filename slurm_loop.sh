#!/bin/bash

for i in $(seq 15 17)
do
   sbatch slurm_script.sh $i
done