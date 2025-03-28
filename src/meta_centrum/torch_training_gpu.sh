#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=12:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb:scratch_local=64gb:cl_elbi1=False:cl_elmu1=False:cl_eluo1=False:cl_elum1=False:cl_elwe=False:cl_elmo5=False:cl_elmo4=False:cl_eltu=False:cl_elmo3=False:cl_elmo2=False:cl_elmo1=False
#PBS -N anomaly_detection_gpu_torch_training

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Variables                                                                    |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Setup environment
CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
DATADIR=/storage/plzen1/home/zapped99/s3/a7/witching_days/keep/XEUR_2021_witching_days
# DATADIR=/storage/plzen1/home/zapped99/dp/mock_data
PROJECT_DIR=/storage/plzen1/home/zapped99/dp
API_KEY=$(cat $PROJECT_DIR/.wandb_api_key)
JSON_TO_LOBSTER=src/data_preprocess/json-detailed2lobster.py
AUGMENT_LOBSTER=src/data_preprocess/augment_lobster.py
MAIN_SCRIPT=src/anomaly_detection/models/autoencoder.py

# Expected input environmental variables: market_id, date, market_segment_id, security_id, and model parameters

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Preparation                                                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Prepare scratch directory
cd $SCRATCHDIR
cp -r $PROJECT_DIR/DP_2024_2025_Zappe/* .

# Clean the results and models directory on scratch
cd res
rm -rf ./*
cd ..
cd models
rm -rf ./*
cd ..

# Copy data files to scratch/data
data_file="${market_id}_${date}_${market_segment_id}_${security_id}_detailed.zip"
cp $DATADIR/$data_file ./data
# Decompress the data file
cd data
unzip $data_file
cd ..

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Venv (Singularity)                                                           |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Prepare the container
singularity run $CONTAINER pip3 install -r requirements.txt --user
singularity run $CONTAINER python3 -m pip install --user wandb
singularity run $CONTAINER python3 -m wandb login --relogin $API_KEY

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Prepare data and analyze it using the model                                  |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Run the parsing of .json to .csv augmented lobster format
singularity run $CONTAINER python3 $JSON_TO_LOBSTER $market_id $date $market_segment_id $security_id
singularity run $CONTAINER python3 $AUGMENT_LOBSTER $market_id $date $market_segment_id $security_id

# Run the Python script
singularity run --nv $CONTAINER python3 $MAIN_SCRIPT --market_id $market_id --date $date --market_segment_id $market_segment_id --security_id $security_id --model_type $model_type --epochs $epochs --kfolds $kfolds --batch_size $batch_size --lr $lr --seq_len $seq_len --latent_dim $latent_dim

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Save the results                                                             |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
# Copy out the results
CONFIG_STRING="${market_id}_${date}_${market_segment_id}_${security_id}_${model_type}_epochs_${epochs}_kfolds_${kfolds}_batch_size_${batch_size}_lr_${lr}_seq_len_${seq_len}_latent_dim_${latent_dim}"
mkdir -p $PROJECT_DIR/DP_2024_2025_Zappe/res/${PBS_JOBID}_${CONFIG_STRING}
cp -r res $PROJECT_DIR/DP_2024_2025_Zappe/res/${PBS_JOBID}_${CONFIG_STRING}
mkdir -p $PROJECT_DIR/DP_2024_2025_Zappe/models/${PBS_JOBID}_${CONFIG_STRING}
cp -r models $PROJECT_DIR/DP_2024_2025_Zappe/models/${PBS_JOBID}_${CONFIG_STRING}

# ┌─────────────────────────────────────────────────────────────────────────────────────────┐
# |            Cleanup                                                                      |
# └─────────────────────────────────────────────────────────────────────────────────────────┘
clean_scratch
