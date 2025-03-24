#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=8:0:0
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=64gb
#PBS -N anomaly_detection_torch_training

# Setup environment
CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
#DATADIR=/storage/plzen1/home/zapped99/s3/a7/witching_days/keep/XEUR_2021_witching_days
DATADIR=/storage/plzen1/home/zapped99/dp/mock_data
PROJECT_DIR=/storage/plzen1/home/zapped99/dp
API_KEY=$(cat $PROJECT_DIR/.wandb_api_key)
PYTHON_SCRIPT=src/anomaly_detection/models/if_ocsvm_lof.py

# Prepare scratch directory
cd $SCRATCHDIR
cp -r $PROJECT_DIR/DP_2024_2025_Zappe/* .

# Copy data files to scratch/data
data_file="${market_id}_${date}_${market_segment_id}_${security_id}_detailed.zip"
cp $DATADIR/$data_file ./data
# Decompress the data file
cd data
unzip $data_file
cd ..

# Prepare the container
singularity run $CONTAINER pip3 install -r requirements_metacentrum.txt --user
singularity run $CONTAINER python3 -m wandb login --relogin API_KEY

# Run the parsing of .json to .csv augmented lobster format
singularity run $CONTAINER python3 src/data_preprocess/json-detailed2lobster.py $market_id $date $market_segment_id $security_id
singularity run $CONTAINER python3 src/data_preprocess/augment_lobster.py $market_id $date $market_segment_id $security_id

# Run the Python script
singularity run $CONTAINER python3 $PYTHON_SCRIPT --model_type $model_type --kfolds $kfolds

# Cleanup
clean_scratch
