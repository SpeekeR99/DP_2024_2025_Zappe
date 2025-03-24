#!/bin/bash

# Check the number of arguments
if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 market_id date market_segment_id security_id"
    echo "Example: $0 XEUR 20191202 688 4128839"
    exit 1
fi

# Assign the arguments to variables
market_id=$1
date=$2
market_segment_id=$3
security_id=$4

# TODO: Grid search for loops

# These variables would be set by the grid search in the for loops normally, hardcoded for now
model_type="cnn"
epochs=500
kfolds=5
batch_size=32
lr=1e-3
seq_len=64
latent_dim=4

# Run the training on a given market, date, market_segment_id, security_id with the given model and its params
# CPU
# qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_type",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$seq_len",latent_dim="$latent_dim" torch_training.sh
# GPU
qsub -v market_id="$market_id",date="$date",market_segment_id="$market_segment_id",security_id="$security_id",model_type="$model_type",epochs="$epochs",kfolds="$kfolds",batch_size="$batch_size",lr="$lr",seq_len="$seq_len",latent_dim="$latent_dim" torch_training_gpu.sh
