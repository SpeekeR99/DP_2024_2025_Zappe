#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=1:0:0
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=64gb:cl_elbi1=False:cl_elmu1=False:cl_eluo1=False:cl_elum1=False:cl_elwe=False:cl_elmo5=False:cl_elmo4=False:cl_eltu=False:cl_elmo3=False:cl_elmo2=False:cl_elmo1=False
#PBS -N anomaly_detection_torch_training

# Setup environment
CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
# DATADIR=/storage/plzen1/home/zapped99/s3/a7/witching_days/keep/XEUR_2021_witching_days
DATADIR=/storage/plzen1/home/zapped99/dp/mock_data
PROJECT_DIR=/storage/plzen1/home/zapped99/dp
API_KEY=$(cat $PROJECT_DIR/.wandb_api_key)
JSON_TO_LOBSTER=src/data_preprocess/json-detailed2lobster.py
AUGMENT_LOBSTER=src/data_preprocess/augment_lobster.py
MAIN_SCRIPT=src/anomaly_detection/models/autoencoder.py

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
singularity run $CONTAINER python3 -m pip install --user wandb
singularity run $CONTAINER python3 -m wandb login --relogin $API_KEY

# Run the parsing of .json to .csv augmented lobster format
singularity run $CONTAINER python3 $JSON_TO_LOBSTER $market_id $date $market_segment_id $security_id
singularity run $CONTAINER python3 $AUGMENT_LOBSTER $market_id $date $market_segment_id $security_id

# Run the Python script
singularity run $CONTAINER python3 $MAIN_SCRIPT --market_id $market_id --date $date --market_segment_id $market_segment_id --security_id $security_id --model_type $model_type --epochs $epochs --kfolds $kfolds --batch_size $batch_size --lr $lr --seq_len $seq_len --latent_dim $latent_dim

# Cleanup
clean_scratch
