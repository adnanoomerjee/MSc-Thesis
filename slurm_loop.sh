#!/bin/bash

for i in $(seq 0)
do
   sbatch slurm_script.sh $i
done