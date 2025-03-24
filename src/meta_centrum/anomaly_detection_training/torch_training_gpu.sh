#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=8:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb:scratch_local=64gb
#PBS -N anomaly_detection_torch_training

# Setup environment
CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
DATADIR=/storage/plzen1/home/zapped99/dp
API_KEY=$(cat $DATADIR/.wandb_api_key)
PYTHON_SCRIPT=TODO/TODO.py

# Prepare scratch directory
cd $SCRATCHDIR
cp -r $DATADIR/DP_2024_2025_Zappe/* .

# TODO: Copy data files to scratch/data

# Prepare the container
singularity run $CONTAINER pip3 install -r requirements_metacentrum.txt --user
singularity run $CONTAINER python3 -m wandb login --relogin API_KEY

# TODO: Run the parsing of .json -> lobster.csv

# Run the Python script
singularity run --nv $CONTAINER python3 $PYTHON_SCRIPT --TODO_flags

# Cleanup
clean_scratch
